use std::fmt::Write as _;

use super::{
    EmitError, Emitter, IndexAxis, Rvalue, Statement, Target, Terminator, Ty, axis_index,
    spirv_index_intrinsic,
};

impl<'a> Emitter<'a> {
    /// Emit a dispatch-id read for `dim`, returning the value (usize-width).
    pub(super) fn emit_thread_index(
        &mut self,
        dim: IndexAxis,
        w: &mut String,
    ) -> Result<String, EmitError> {
        match self.target {
            // The AIE-core target runs one core, so there is no SPMD dispatch id. A kernel for it must be a sequential
            // loop (the matmul/elementwise loop forms), not one-thread-per-element. Reject by name (FR-004)
            // instead of inventing an index.
            Target::AieCore => Err(EmitError::Unsupported(
                "thread_index on AIE-core: a single AIE core has no SPMD dispatch id; author the kernel \
                 as a sequential loop over the tile (the IRON harness handles tiling + data movement)".into(),
            )),
            Target::SpirvVulkan => {
                // Shared 2-D fold reconstruction: the host may launch `[x, y, 1]` workgroups after
                // folding a 1-D grid over the device cap (`poot_runtime_common::fold_grid`). The linear
                // thread id is `group_y * x_extent + global_id.x`, where `x_extent = x * workgroup_size[0]`
                // is appended to the length buffer at slot `param_count`. When the launch stays 1-D,
                // `group_y` is 0 and this reduces to plain `global_id.x`. Only `IndexAxis::X` is folded
                // (Y/Z thread ids stay raw); kernels that already reconstruct via `group_index_y`
                // do not call `thread_index` for the linear id.
                if dim == IndexAxis::X {
                    let (intrin, axis) = spirv_index_intrinsic(dim);
                    self.declares.insert(format!("declare i32 @{intrin}(i32)"));
                    let gx = self.fresh();
                    let _ = writeln!(w, "  {gx} = tail call i32 @{intrin}(i32 {axis})");
                    let (gintrin, gaxis) = spirv_index_intrinsic(IndexAxis::GroupY);
                    self.declares.insert(format!("declare i32 @{gintrin}(i32)"));
                    let gy = self.fresh();
                    let _ = writeln!(w, "  {gy} = tail call i32 @{gintrin}(i32 {gaxis})");
                    // lengths[param_count] is the X-thread extent (see poot-runtime's length ABI).
                    let slot = self.body.param_count;
                    let (bufty, mangle) = self.spirv_len_buf();
                    let p = self.fresh();
                    let _ = writeln!(
                        w,
                        "  {p} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} %hlen, i32 {slot})"
                    );
                    self.declares.insert(format!(
                        "declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty}, i32)"
                    ));
                    let ext = self.fresh();
                    let _ = writeln!(w, "  {ext} = load i32, ptr addrspace(11) {p}, align 4");
                    let scaled = self.fresh();
                    let _ = writeln!(w, "  {scaled} = mul i32 {gy}, {ext}");
                    let lin = self.fresh();
                    let _ = writeln!(w, "  {lin} = add i32 {scaled}, {gx}");
                    return Ok(lin); // i32 == usize on spirv
                }
                let (intrin, axis) = spirv_index_intrinsic(dim);
                self.declares.insert(format!("declare i32 @{intrin}(i32)"));
                let v = self.fresh();
                let _ = writeln!(w, "  {v} = tail call i32 @{intrin}(i32 {axis})");
                Ok(v) // i32 == usize on spirv
            }
            Target::Nvptx => {
                let r = self.emit_nvptx_index(dim, w)?;
                // zext i32 -> i64 (usize on nvptx).
                let z = self.fresh();
                let _ = writeln!(w, "  {z} = zext i32 {r} to i64");
                Ok(z)
            }
            // AMDGPU: same IR contract as NVPTX: `X/Y/Z` is the global work-item id, `LocalX/Y/Z` the per-wavefront thread
            // id (lane), `GroupX/Y/Z` the workgroup id. The backend lowers `@llvm.amdgcn.workitem.id` to v0 (lane id,
            // 0..workgroup_size-1), so the global id is assembled in IR as `workgroup_id * workgroup_size + workitem_id`.
            // The workgroup id comes from `@llvm.amdgcn.workgroup.id`, lowered to s_getreg with no SGPR preload needed;
            // both are preloaded for `amdgpu_kernel` CC (`SIMachineFunctionInfo` sets `WorkItemIDX` and `WorkGroupIDX`),
            // and the kd's `kernel_code_properties = 0x0408` covers both reads.
            Target::AmdGcn(_) => {
                let r = self.emit_amdgcn_index(dim, w)?;
                // zext i32 -> i64 (usize on amdgcn, same as nvptx).
                let z = self.fresh();
                let _ = writeln!(w, "  {z} = zext i32 {r} to i64");
                Ok(z)
            }
        }
    }

    pub(super) fn emit_nvptx_index(
        &mut self,
        dim: IndexAxis,
        w: &mut String,
    ) -> Result<String, EmitError> {
        let axis = |a: IndexAxis| match a {
            IndexAxis::X | IndexAxis::LocalX | IndexAxis::GroupX => "x",
            IndexAxis::Y | IndexAxis::LocalY | IndexAxis::GroupY => "y",
            IndexAxis::Z | IndexAxis::LocalZ | IndexAxis::GroupZ => "z",
        };
        let a = axis(dim);
        let read = |this: &mut Self, sreg: &str, w: &mut String| -> String {
            this.declares
                .insert(format!("declare i32 @llvm.nvvm.read.ptx.sreg.{sreg}.{a}()"));
            let v = this.fresh();
            let _ = writeln!(
                w,
                "  {v} = tail call i32 @llvm.nvvm.read.ptx.sreg.{sreg}.{a}()"
            );
            v
        };
        match dim {
            IndexAxis::LocalX | IndexAxis::LocalY | IndexAxis::LocalZ => Ok(read(self, "tid", w)),
            IndexAxis::GroupX | IndexAxis::GroupY | IndexAxis::GroupZ => Ok(read(self, "ctaid", w)),
            // global: ctaid * ntid + tid
            IndexAxis::X | IndexAxis::Y | IndexAxis::Z => {
                let tid = read(self, "tid", w);
                let ntid = read(self, "ntid", w);
                let ctaid = read(self, "ctaid", w);
                let base = self.fresh();
                let _ = writeln!(w, "  {base} = mul i32 {ctaid}, {ntid}");
                let gid = self.fresh();
                let _ = writeln!(w, "  {gid} = add i32 {base}, {tid}");
                Ok(gid)
            }
        }
    }

    /// Emit an AMDGPU dispatch-id read for `dim`, returning the value (i32). Mirrors `emit_nvptx_index` so the IR
    /// contract is backend-neutral: `LocalX/Y/Z` is the per-wavefront thread id (0..workgroup_size-1),
    /// `GroupX/Y/Z` the workgroup id, `X/Y/Z` the global work-item id (`workgroup_id * workgroup_size +
    /// workitem_id`). The workgroup size comes from `self.body.workgroup_size`.
    ///
    /// Intrinsics: `@llvm.amdgcn.workitem.id.<axis>()` (lowered to v0) and `@llvm.amdgcn.workgroup.id.<axis>()`
    /// (lowered to `s_getreg` HW_REG_HW_ID). Both are preloaded by `SIMachineFunctionInfo` for `amdgpu_kernel` CC,
    /// so the kd's `kernel_code_properties = 0x0408` (KERNARG + WAVEFRONT_SIZE32) suffices.
    pub(super) fn emit_amdgcn_index(
        &mut self,
        dim: IndexAxis,
        w: &mut String,
    ) -> Result<String, EmitError> {
        let axis = match dim {
            IndexAxis::X | IndexAxis::LocalX | IndexAxis::GroupX => "x",
            IndexAxis::Y | IndexAxis::LocalY | IndexAxis::GroupY => "y",
            IndexAxis::Z | IndexAxis::LocalZ | IndexAxis::GroupZ => "z",
        };
        // Read one of the two AMDGPU intrinsics into a fresh i32 SSA value.
        let read = |this: &mut Self, kind: &str, w: &mut String| -> String {
            let intrin = format!("llvm.amdgcn.{kind}.id.{axis}");
            this.declares.insert(format!("declare i32 @{intrin}()"));
            let v = this.fresh();
            let _ = writeln!(w, "  {v} = tail call i32 @{intrin}()");
            v
        };
        Ok(match dim {
            IndexAxis::LocalX | IndexAxis::LocalY | IndexAxis::LocalZ => read(self, "workitem", w),
            IndexAxis::GroupX | IndexAxis::GroupY | IndexAxis::GroupZ => read(self, "workgroup", w),
            // global: workgroup_id * workgroup_size[axis] + workitem_id
            IndexAxis::X | IndexAxis::Y | IndexAxis::Z => {
                let wid = read(self, "workitem", w);
                let wgid = read(self, "workgroup", w);
                let wgsize = self.body.workgroup_size[axis_index(dim)];
                let base = self.fresh();
                let _ = writeln!(w, "  {base} = mul i32 {wgid}, {wgsize}");
                let gid = self.fresh();
                let _ = writeln!(w, "  {gid} = add i32 {base}, {wid}");
                gid
            }
        })
    }

    // ---- SPIR-V buffer type helpers ------------------------------------------------------------

    /// (buffer target type, intrinsic mangle) for a data buffer of `elem`. `writable` selects the
    /// `(...,12,1)` shape (poot always emits writable data buffers; only the length buffer is read-only).
    pub(super) fn spirv_buf(
        &self,
        elem: &Ty,
        writable: bool,
    ) -> Result<(String, String), EmitError> {
        let (llty, mangle_elem): (String, String) = match elem {
            Ty::F32 => ("float".into(), "f32".into()),
            Ty::F64 => ("double".into(), "f64".into()),
            Ty::F16 => ("half".into(), "f16".into()),
            // bf16 buffers are NVPTX-only (see scalar_llty); spirv_buf is the SPIR-V path, so reject.
            Ty::BF16 => {
                return Err(EmitError::Unsupported(
                    "bf16 compute is NVPTX-only (SPIR-V needs SPV_KHR_bfloat16, unsupported on the Arc)"
                        .into(),
                ));
            }
            Ty::I32 | Ty::U32 | Ty::Usize => ("i32".into(), "i32".into()),
            // spec 134 P1 / FR-004: a vec4 buffer is declared with the `<lanes x elem>` element type directly (the A1
            // probe fixture path), never a bitcast over a scalar-element resource (the A2 probe fixture crashes llc).
            // `lanes` is folded into the mangle (`v4f32`) so distinct widths get distinct declared intrinsics, matching
            // the fixture's `tspirv.VulkanBuffer_a0v4f32_12_1t` naming.
            Ty::Vec { elem, lanes } => {
                let (inner_llty, inner_mangle): (String, String) = match elem.as_ref() {
                    Ty::F32 => ("float".into(), "f32".into()),
                    Ty::F16 => ("half".into(), "f16".into()),
                    Ty::I32 | Ty::U32 | Ty::Usize => ("i32".into(), "i32".into()),
                    other => {
                        return Err(EmitError::Unsupported(format!(
                            "vector buffer element {other:?}"
                        )));
                    }
                };
                (
                    format!("<{lanes} x {inner_llty}>"),
                    format!("v{lanes}{inner_mangle}"),
                )
            }
            other => return Err(EmitError::Unsupported(format!("slice element {other:?}"))),
        };
        let wbit = if writable { 1 } else { 0 };
        let bufty = format!("target(\"spirv.VulkanBuffer\", [0 x {llty}], 12, {wbit})");
        let mangle = format!("tspirv.VulkanBuffer_a0{mangle_elem}_12_{wbit}t");
        Ok((bufty, mangle))
    }

    pub(super) fn spirv_len_buf(&self) -> (String, String) {
        (
            "target(\"spirv.VulkanBuffer\", [0 x i32], 12, 0)".to_string(),
            "tspirv.VulkanBuffer_a0i32_12_0t".to_string(),
        )
    }

    pub(super) fn body_uses_len(&self) -> bool {
        self.body
            .blocks
            .iter()
            .flat_map(|b| &b.statements)
            .any(|s| matches!(s, Statement::Assign(_, Rvalue::Len(_))))
    }

    /// Whether this body's SpirvVulkan form must bind the length buffer: it either reads param
    /// lengths (`Rvalue::Len`) or reconstructs a folded `thread_index(X)` from the X-thread extent
    /// the runtime appends at slot `param_count`.
    pub(super) fn body_needs_length_buffer(&self) -> bool {
        if self.body_uses_len() {
            return true;
        }
        self.target == Target::SpirvVulkan
            && self.body.blocks.iter().any(|b| {
                matches!(
                    b.terminator,
                    Terminator::ThreadIndexCall {
                        dim: IndexAxis::X,
                        ..
                    }
                )
            })
    }

    // ---- footer --------------------------------------------------------------------------------

    pub(super) fn emit_footer(&mut self) {
        let decls: Vec<String> = self.declares.iter().cloned().collect();
        for d in decls {
            self.out.push('\n');
            self.out.push_str(&d);
        }
        self.out.push('\n');
        if self.target == Target::SpirvVulkan {
            let [x, y, z] = self.body.workgroup_size;
            let _ = writeln!(
                self.out,
                "\nattributes #0 = {{ \"hlsl.numthreads\"=\"{x},{y},{z}\" \"hlsl.shader\"=\"compute\" }}"
            );
            // #1: the workgroup barrier's convergence attributes (see the Barrier emit).
            let _ = writeln!(self.out, "attributes #1 = {{ convergent nounwind }}");
        } else if self.strictfp_needed {
            // card 628: Nvptx/AmdGcn's `#0` is unused otherwise (see `strictfp_attr_suffix`).
            let _ = writeln!(self.out, "\nattributes #0 = {{ strictfp }}");
        }
    }
}
