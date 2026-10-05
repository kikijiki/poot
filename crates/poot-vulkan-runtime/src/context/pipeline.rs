use crate::*;

impl Context {
    /// Validate one dispatch's bindings against the kernel's argument `schema` (card 608, SC-002) and
    /// this context, and validate its grid. Every dispatch path goes through here, so a kernel is only ever
    /// bound to exactly the number and native kind of buffers its schema names, all of them children of
    /// this context's device, each told no more elements than its allocation holds.
    pub(crate) fn check_bindings(
        &self,
        schema: &[poot_runtime_common::ArgSchema],
        wg: [u32; 3],
        threads: [u32; 3],
        args: &[Binding<'_>],
    ) -> Result<[u32; 3], RuntimeError> {
        self.owner.check_live("dispatch")?;
        let elements: Vec<ElementKind> = args.iter().map(|a| a.buffer.element()).collect();
        poot_runtime_common::check_element_schema(schema, &elements)?;
        if args.iter().any(|a| !a.buffer.belongs_to(&self.owner)) {
            return Err(RuntimeError::ForeignObject("buffer"));
        }
        for (index, arg) in args.iter().enumerate() {
            let element = arg.buffer.element();
            let bytes = u64::from(arg.elems) * element.byte_width() as u64;
            if bytes > arg.buffer.byte_len() as u64 {
                return Err(RuntimeError::ArgExceedsBuffer {
                    index,
                    elems: arg.elems,
                    element,
                    byte_len: arg.buffer.byte_len(),
                });
            }
        }
        // Fold first so the length block carries the X-thread extent for `thread_index` reconstruction.
        crate::resolve_groups(threads, wg, self.max_workgroup_count)
    }

    /// Build the (module, descriptor-set-layout, pipeline-layout, pipeline) quartet for `kernel`: its
    /// data buffers at bindings `0..n`, the length buffer at binding `n`, and for a trapping kernel the
    /// error word at `n + 1`. Used uncached by [`Context::dispatch`] and through the cache by
    /// [`Context::pipeline`]. On failure, everything built so far is torn down before returning the
    /// error.
    pub(crate) fn build_pipeline(&self, kernel: &CompiledKernel) -> Result<Pipeline, RuntimeError> {
        let shape = PipelineShape::of(kernel)?;
        let facts = crate::spirv::scan(&shape.words);
        // A module with cooperative matrices needs full subgroups of the size its workgroup was written
        // for (`REQUIRE_FULL_SUBGROUPS` and a required size equal to its `LocalSize` X): pre-1.6 modules
        // cannot use the extension otherwise, and a workgroup that is not a multiple of the default
        // subgroup size would not be one.
        let required_subgroup_size = if facts.cooperative_matrix {
            match (self.coopmat_subgroup_sizes, facts.local_size_x) {
                (Some((min, max)), Some(x)) if x.is_power_of_two() && (min..=max).contains(&x) => {
                    Some(x)
                }
                (supported, local_size_x) => {
                    return Err(RuntimeError::CooperativeMatrixUnsupported {
                        local_size_x,
                        supported,
                    });
                }
            }
        } else {
            None
        };
        let n = shape.args.len();
        let bound = n + 1 + usize::from(shape.has_trap);
        let words = &*shape.words;
        let module_info = vk::ShaderModuleCreateInfo::default().code(words);
        // SAFETY: `module_info` borrows the kernel's words, which outlive the call; `self.device` is
        // valid. `kernel` is a `CompiledKernel`, constructible only by poot-codegen's `unsafe`
        // constructor, whose contract makes the words a valid SPIR-V module.
        let module = unsafe { self.device.create_shader_module(&module_info, None)? };

        let bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..bound as u32)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let dsl_info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        // SAFETY: `self.device` is valid; `bindings` outlives the call.
        let dsl = match unsafe { self.device.create_descriptor_set_layout(&dsl_info, None) } {
            Ok(d) => d,
            Err(e) => {
                // SAFETY: `module` has no other owner yet.
                unsafe { self.device.destroy_shader_module(module, None) };
                return Err(e.into());
            }
        };

        let set_layouts = [dsl];
        let pl_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        // SAFETY: `self.device`/`dsl` are valid; `pl_info` outlives the call.
        let pipeline_layout = match unsafe { self.device.create_pipeline_layout(&pl_info, None) } {
            Ok(p) => p,
            Err(e) => {
                // SAFETY: neither has any other owner yet.
                unsafe {
                    self.device.destroy_descriptor_set_layout(dsl, None);
                    self.device.destroy_shader_module(module, None);
                }
                return Err(e.into());
            }
        };

        let entry_point = c"main";
        let mut required_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(required_subgroup_size.unwrap_or(0));
        let mut stage_info = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(entry_point);
        if required_subgroup_size.is_some() {
            stage_info = stage_info
                .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                .push_next(&mut required_size);
        }
        let pipeline_info = vk::ComputePipelineCreateInfo::default()
            .stage(stage_info)
            .layout(pipeline_layout);
        // SAFETY: `self.device` is valid; `pipeline_info` (and the stage/module/layout it references)
        // outlive the call; a null pipeline cache is a valid "no cache" argument.
        let pipeline = match unsafe {
            self.device
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
        } {
            Ok(p) => p[0],
            Err((_, e)) => {
                // SAFETY: none of these have any other owner yet.
                unsafe {
                    self.device.destroy_pipeline_layout(pipeline_layout, None);
                    self.device.destroy_descriptor_set_layout(dsl, None);
                    self.device.destroy_shader_module(module, None);
                }
                return Err(e.into());
            }
        };

        Ok(Pipeline {
            module,
            dsl,
            pipeline_layout,
            pipeline,
            shape,
            owner: Arc::clone(&self.owner),
        })
    }
}

impl PipelineShape {
    /// Whether `kernel` is the kernel this shape was built from (the cache's identity check, with no
    /// copy of the module), or a typed error if `kernel` was compiled for another backend.
    pub(crate) fn matches(&self, kernel: &CompiledKernel) -> Result<bool, RuntimeError> {
        Ok(*self.args == *kernel.args()
            && self.has_trap == kernel.has_trap()
            && *self.words == *spirv_words(kernel)?)
    }

    /// The identity of `kernel`'s pipeline: its SPIR-V module and binding shape, or a typed error if it
    /// was compiled for another backend.
    pub(crate) fn of(kernel: &CompiledKernel) -> Result<Self, RuntimeError> {
        Ok(PipelineShape {
            args: kernel.args().into(),
            has_trap: kernel.has_trap(),
            words: spirv_words(kernel)?.into(),
        })
    }
}
