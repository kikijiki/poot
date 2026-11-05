use std::fmt::Write as _;

use poot_kernel_ir::{
    FragmentMajor, FragmentRole, FragmentTy, Operand, Ty, WmmaMat, WmmaShape, accumulator_map,
};

use super::{EmitError, Emitter, Local, Place, ProjectionElem, Target};

impl<'a> Emitter<'a> {
    // ---- WMMA tensor-core intrinsics (spec 025, NVPTX + AMDGPU) ------------------------------------
    // m16n16k16, row-major, f32 accumulate. Fragments are register-group aggregates/vectors, alloca'd like
    // any other local at the physical type `fragment_llty` picks (see `emit_entry`); the load/store
    // intrinsics handle the fragment<->lane layout via (ptr, stride). See the llc-22 feasibility probe in
    // specs/025. Card 110 adds AMDGPU WMMA via `llvm.amdgcn.wmma.*` intrinsics; card 530 makes F16 the
    // dtype every target shares (NVPTX/AMDGCN also keep Bf16 for the production tiled dequant/matmul path;
    // SPIR-V/RADV has no BF16 coopmat).

    /// Gate a WMMA/coopmat statement on (target, operand dtype) (card 154, extended target-neutral by card
    /// 530). `F16` lowers on every target implemented today; `Bf16` lowers only on NVPTX/AMDGCN (RADV has no
    /// `VK_KHR_shader_bfloat16`). Every other combination is a typed "unsupported" error instead of the
    /// wrong fragment layout or intrinsic.
    pub(super) fn wmma_check_target(
        &self,
        dtype: poot_kernel_ir::WmmaDtype,
    ) -> Result<(), EmitError> {
        use poot_kernel_ir::WmmaDtype;
        match (self.target, dtype) {
            (Target::Nvptx | Target::AmdGcn(_) | Target::SpirvVulkan, WmmaDtype::F16) => Ok(()),
            (Target::Nvptx | Target::AmdGcn(_), WmmaDtype::Bf16) => Ok(()),
            (Target::SpirvVulkan, WmmaDtype::Bf16) => Err(EmitError::Unsupported(
                "SPIR-V/Vulkan cooperative-matrix has no BF16 fragment support (RADV has no \
                 VK_KHR_shader_bfloat16 - config 13 is f16xf16->f32); use F16 operands"
                    .into(),
            )),
            (Target::AieCore, _) => Err(EmitError::Unsupported(
                "WMMA/cooperative-matrix tensor-core ops require NVPTX, AMDGPU, or SpirvVulkan \
                 (no AIE-core equivalent)"
                    .into(),
            )),
        }
    }

    /// Gate a WMMA/coopmat statement on `shape` (card 530). Every lowering in this file
    /// hard-codes m16n16k16 (register counts, intrinsic names, per-lane offset arithmetic); a `WmmaShape`
    /// this codegen does not implement must be a typed refusal, not a silent m16n16k16 lowering of the
    /// wrong tile size - the same "no silently wrong fragment layout" contract `wmma_check_target` gives
    /// `dtype`. Call this before any shape-dependent codegen; when a second shape exists, this is where its
    /// per-target support matrix goes.
    pub(super) fn wmma_check_shape(
        &self,
        shape: poot_kernel_ir::WmmaShape,
    ) -> Result<(), EmitError> {
        if shape == poot_kernel_ir::WmmaShape::M16N16K16 {
            Ok(())
        } else {
            Err(EmitError::Unsupported(format!(
                "WMMA/cooperative-matrix shape {shape:?} is not implemented (every target's lowering in \
                 this codegen hard-codes m16n16k16); only WmmaShape::M16N16K16 lowers today"
            )))
        }
    }

    /// Cross-check a `WmmaLoad`/`WmmaLoadLds` statement's `dtype` against the fragment source's actual
    /// element type (card 530). AMDGCN's load derives its per-lane element type from the
    /// tile place and ignores `dtype`; NVPTX's load derives the intrinsic (and so the bit width it reads)
    /// from `dtype` and never looks at the tile. Before this card, that combination was unreachable
    /// (NVPTX/AMDGCN only ever saw `Bf16`); `wmma_check_target` now legalises `F16` on those targets too,
    /// so a future generator could produce `WmmaLoad { dtype: F16 }` over a bf16 slice (or the reverse) and
    /// silently reinterpret bits with no error. Comparing `dtype` against the element `Ty` this codegen
    /// already resolves for the tile turns that into a typed refusal.
    pub(super) fn check_wmma_dtype_matches(
        &self,
        elem_ty: &poot_kernel_ir::Ty,
        dtype: poot_kernel_ir::WmmaDtype,
    ) -> Result<(), EmitError> {
        use poot_kernel_ir::{Ty, WmmaDtype};
        let expected = match dtype {
            WmmaDtype::Bf16 => Ty::BF16,
            WmmaDtype::F16 => Ty::F16,
        };
        if *elem_ty == expected {
            Ok(())
        } else {
            Err(EmitError::Unsupported(format!(
                "WmmaLoad dtype {dtype:?} does not match the fragment source's element type {elem_ty:?} \
                 (expected {expected:?}); a mismatched dtype silently reinterprets bits instead of \
                 converting them"
            )))
        }
    }

    /// [`Self::check_wmma_dtype_matches`] for a global tile place (`WmmaLoad`'s `tile`).
    pub(super) fn check_wmma_load_dtype(
        &self,
        tile: &Place,
        dtype: poot_kernel_ir::WmmaDtype,
    ) -> Result<(), EmitError> {
        let elem_ty = self.place_ty(tile)?;
        self.check_wmma_dtype_matches(&elem_ty, dtype)
    }

    /// [`Self::check_wmma_dtype_matches`] for a workgroup-local (LDS) array (`WmmaLoadLds`'s `array`): an
    /// A/B operand tile holds `dtype`, an accumulator tile holds f32.
    pub(super) fn check_wmma_load_lds_dtype(
        &self,
        array: u8,
        role: FragmentRole,
        dtype: poot_kernel_ir::WmmaDtype,
    ) -> Result<(), EmitError> {
        let elem_ty = self.body.workgroup_locals[array as usize].elem_ty.clone();
        match role {
            FragmentRole::Matrix(_) => self.check_wmma_dtype_matches(&elem_ty, dtype),
            FragmentRole::Accumulator => self.check_wmma_lds_accumulator_array(array),
        }
    }

    /// An accumulator tile staged in LDS is f32 (`WmmaStoreLds` writes it, an accumulator `WmmaLoadLds`
    /// reads it): any other element type would convert silently or reinterpret bits.
    pub(super) fn check_wmma_lds_accumulator_array(&self, array: u8) -> Result<(), EmitError> {
        let elem_ty = &self.body.workgroup_locals[array as usize].elem_ty;
        if *elem_ty == Ty::F32 {
            Ok(())
        } else {
            Err(EmitError::Unsupported(format!(
                "an accumulator fragment tile in LDS must be f32, array {array} holds {elem_ty:?}"
            )))
        }
    }

    /// The AMD arch descriptor when the target is AMDGPU, else `None`; WMMA emit uses it to gate the tensor-core
    /// intrinsic on the detected capability family (spec 120).
    pub(super) fn amd_arch(&self) -> Option<crate::AmdArch> {
        match self.target {
            Target::AmdGcn(arch) => Some(arch),
            _ => None,
        }
    }

    /// Guard the AMDGPU WMMA path on the arch's matrix units. Only the RDNA3/3.5 (gfx11xx) WMMA 16x16x16
    /// layout is emitted; RDNA4 (128b) and CDNA (MFMA) report no unit, so they get a typed "unsupported target"
    /// error instead of the wrong intrinsic or a backend "cannot select" crash (FR-003).
    pub(super) fn require_amd_wmma(&self) -> Result<(), EmitError> {
        if let Some(arch) = self.amd_arch()
            && arch.matrix_units().is_empty()
        {
            return Err(EmitError::Unsupported(format!(
                "WMMA tensor-core codegen unsupported for AMD target {} (wave {}); only \
                 RDNA3/3.5 (gfx11xx) WMMA 16x16x16 is implemented",
                arch.mcpu(),
                arch.wave,
            )));
        }
        Ok(())
    }

    // ---- fragment storage (cards 530/663) -----------------------------------------------------------
    // A fragment `Local` is an ordinary alloca'd local on every target: `emit_entry` allocas it at the
    // physical type its declared `Ty::Fragment` names (below) instead of a scalar size, and every op
    // reads/writes it through the normal `load_place`/`store_place` path. Being a plain variable is what lets a
    // fragment, the accumulator above all, carry across a loop back-edge on every target with no phi
    // machinery.

    /// The alloca'd LLVM type of a fragment local. The accumulator/D fragment is always f32; an A/B operand
    /// fragment carries `dtype`.
    ///
    /// - AMDGCN packs every dtype into the same `<8 x i32>` raw-bits vector (16 lanes of a 2-byte element,
    ///   bitcast to the dtype-typed vector only at the point of use - see `emit_wmma_load_amdgcn`); the
    ///   accumulator is `<8 x float>`.
    /// - NVPTX's fragment shape is dtype-sized (bf16: 4 lanes of i32; f16: 8 lanes of `<2 x half>`, per the
    ///   NVVM WMMA fragment table), so its aggregate type must match exactly what the load/mma intrinsics
    ///   return/accept; the accumulator is eight floats.
    /// - SPIR-V's is the opaque `OpTypeCooperativeMatrixKHR` of the fragment's `Use`, held in a `Function`
    ///   variable.
    pub(super) fn fragment_llty(&self, fragment: FragmentTy) -> Result<String, EmitError> {
        self.wmma_check_target(fragment.dtype)?;
        self.wmma_check_shape(fragment.shape)?;
        let is_acc = fragment.role == FragmentRole::Accumulator;
        match self.target {
            Target::AmdGcn(_) => Ok(if is_acc {
                "<8 x float>".into()
            } else {
                "<8 x i32>".into()
            }),
            Target::Nvptx => Ok(if is_acc {
                Self::NVPTX_ACC_STRUCT_TY.into()
            } else {
                Self::nvptx_ab_struct_ty(fragment.dtype)
            }),
            Target::SpirvVulkan => Ok(match fragment.role {
                FragmentRole::Matrix(WmmaMat::A) => self.coopmat_ty("half", 0),
                FragmentRole::Matrix(WmmaMat::B) => self.coopmat_ty("half", 1),
                FragmentRole::Accumulator => self.coopmat_ty("float", 2),
            }),
            Target::AieCore => Err(EmitError::Unsupported(format!(
                "matrix fragments have no {:?} representation",
                self.target
            ))),
        }
    }

    // ---- SPIR-V cooperative-matrix (card 154) --------------------------------------------------
    // RADV config 13: F16xF16->F32, 16x16x16, Subgroup scope. A coopmat value is one opaque
    // `OpTypeCooperativeMatrixKHR` with no addressable components; a fragment local holds it in a `Function`
    // variable (see `fragment_llty`), so a K-loop carries the accumulator through that variable.
    // `spirv_postprocess::fix_coopmat_calls` rewrites the Import-linkage stub calls that LLVM's Vulkan-mode
    // SPIR-V backend emits for these `__spirv_*` builtins into the real
    // `OpCooperativeMatrix{Load,Store,MulAdd}KHR`/`OpCompositeConstruct` instructions (in `compile()`'s
    // post-process pipeline). The mangled name does not matter under Vulkan mode (llc recognizes none of these
    // builtins there, mangled correctly or not, and always emits a generic Import-linked declaration), so the
    // `_N` suffixes below only need to be distinct per call site with a different result type.

    /// The LLVM type text for a `16x16x16` Subgroup-scope coopmat value of component type
    /// `component_llty` (`"half"` for A/B, `"float"` for the accumulator) and `Use` (0=MatrixA,
    /// 1=MatrixB, 2=MatrixAccumulator).
    pub(super) fn coopmat_ty(&self, component_llty: &str, use_: u32) -> String {
        format!("target(\"spirv.CooperativeMatrixKHR\", {component_llty}, 3, 16, 16, {use_})")
    }

    /// A `[0 x i8]` (byte-array) `spirv.VulkanBuffer` type + its `llvm.spv.resource.*` mangle suffix, following
    /// `spirv_buf`'s naming for `half`/`float`/`i32` (`tspirv.VulkanBuffer_a0<elem>_12_<writable>t`) with `i8` as
    /// the element.
    pub(super) fn coopmat_byte_buf(&self, writable: bool) -> (String, String) {
        let wbit = if writable { 1 } else { 0 };
        (
            format!("target(\"spirv.VulkanBuffer\", [0 x i8], 12, {wbit})"),
            format!("tspirv.VulkanBuffer_a0i8_12_{wbit}t"),
        )
    }

    /// A byte-addressed `StorageBuffer` pointer to `(*tile.local)[idx] * elem_bytes` for a coopmat Load/Store's
    /// Pointer operand.
    ///
    /// This does not reuse `element_ptr`, which returns a `half`/`float`-typed pointer via the param's normal
    /// resource binding. Under Vulkan/Shader mode these coopmat builtins are unrecognized external declarations
    /// (see the section comment), and an opaque `ptr` parameter of such a declaration defaults to a `uchar` (byte)
    /// pointee regardless of the argument's type or the mangled name (an Itanium-mangled name encoding `half
    /// addrspace(11)*` made no difference). A `half`/`float`-typed pointer forces an `OpBitcast`, which `spirv-val`
    /// rejects for `OpCooperativeMatrix{Load,Store}KHR`'s Pointer operand ("is not a logical pointer": Vulkan's
    /// Logical addressing requires the pointer to trace back through only `OpVariable`/`OpAccessChain`). So bind a
    /// separate `[0 x i8]` view of the same descriptor slot as the param's typed resource handle (a legal aliased
    /// binding that validates alongside the one `emit_entry` creates for every param) and GEP into that at the byte
    /// offset; its `uchar` pointee matches the callee's default, so no bitcast is inserted.
    pub(super) fn coopmat_byte_ptr(
        &mut self,
        tile: &Place,
        elem_bytes: u32,
        w: &mut String,
    ) -> Result<String, EmitError> {
        let idx_local = match tile.projection.as_slice() {
            [ProjectionElem::Deref, ProjectionElem::Index(i)] => *i,
            other => {
                return Err(EmitError::Unsupported(format!(
                    "coopmat tile projection {other:?}"
                )));
            }
        };
        let (idx_val, _) = self.load_place(&Place::local(idx_local), w)?;
        let byte_idx = self.fresh();
        let _ = writeln!(w, "  {byte_idx} = mul i32 {idx_val}, {elem_bytes}");
        let (bufty, mangle) = self.coopmat_byte_buf(true);
        let bind = tile.local.index - 1;
        let h = self.fresh();
        let _ = writeln!(
            w,
            "  {h} = tail call {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32 0, i32 {bind}, i32 1, i32 0, ptr nonnull @.pn{bind})"
        );
        self.declares.insert(format!(
            "declare {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32, i32, i32, i32, ptr)"
        ));
        self.declares.insert(format!(
            "declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty}, i32)"
        ));
        let p = self.fresh();
        let _ = writeln!(
            w,
            "  {p} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} {h}, i32 {byte_idx})"
        );
        Ok(p)
    }

    /// `OpCooperativeMatrixLoadKHR`, RowMajor. `which` selects the component type (`half`) and `Use`
    /// (MatrixA/MatrixB); accumulators are seeded by `WmmaZero`, not loaded, so this loads only A/B. `stride`
    /// is in elements (as in the NVPTX/AMDGCN WMMA convention); because the pointer is byte-addressed (see
    /// `coopmat_byte_ptr`), the element index and the Stride operand are both scaled to bytes (2 bytes/half).
    pub(super) fn emit_wmma_load_coopmat(
        &mut self,
        which: poot_kernel_ir::WmmaMat,
        tile: &Place,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let use_ = match which {
            poot_kernel_ir::WmmaMat::A => 0,
            poot_kernel_ir::WmmaMat::B => 1,
        };
        let ty = self.coopmat_ty("half", use_);
        const ELEM_BYTES: u32 = 2; // half
        let ptr = self.coopmat_byte_ptr(tile, ELEM_BYTES, w)?;
        let stride_bytes = stride * ELEM_BYTES;
        let mangled = match which {
            poot_kernel_ir::WmmaMat::A => "_Z32__spirv_CooperativeMatrixLoadKHR_1",
            poot_kernel_ir::WmmaMat::B => "_Z32__spirv_CooperativeMatrixLoadKHR_2",
        };
        self.declares.insert(format!(
            "declare dso_local spir_func {ty} @{mangled}(ptr addrspace(11), i32, i64, i32)"
        ));
        let d = self.fresh();
        let _ = writeln!(
            w,
            "  {d} = tail call spir_func {ty} @{mangled}(ptr addrspace(11) {ptr}, i32 0, i64 {stride_bytes}, i32 0)"
        );
        self.store_place(&Place::local(dst), &d, &ty, w)
    }

    /// Seed a zero-valued `float` accumulator fragment via `OpCompositeConstruct` (the coopmat
    /// scalar-broadcast constructor - see `WmmaZero`'s doc).
    pub(super) fn emit_wmma_zero_coopmat(
        &mut self,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let ty = self.coopmat_ty("float", 2);
        let mangled = "_Z27__spirv_CompositeConstruct";
        self.declares.insert(format!(
            "declare dso_local spir_func {ty} @{mangled}(float)"
        ));
        let d = self.fresh();
        let _ = writeln!(
            w,
            "  {d} = tail call spir_func {ty} @{mangled}(float 0.000000e+00)"
        );
        self.store_place(&Place::local(dst), &d, &ty, w)
    }

    /// `OpCooperativeMatrixMulAddKHR`: `dst = a * b + c` (fused).
    pub(super) fn emit_wmma_mma_coopmat(
        &mut self,
        a: Local,
        b: Local,
        c: Local,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (a_reg, a_ty) = self.load_place(&Place::local(a), w)?;
        let (b_reg, b_ty) = self.load_place(&Place::local(b), w)?;
        let (c_reg, c_ty) = self.load_place(&Place::local(c), w)?;
        let d_ty = c_ty.clone(); // the accumulator shape is unchanged by MulAdd
        let mangled = "_Z34__spirv_CooperativeMatrixMulAddKHR";
        self.declares.insert(format!(
            "declare dso_local spir_func {d_ty} @{mangled}({a_ty}, {b_ty}, {c_ty}, i32)"
        ));
        let d = self.fresh();
        let _ = writeln!(
            w,
            "  {d} = tail call spir_func {d_ty} @{mangled}({a_ty} {a_reg}, {b_ty} {b_reg}, {c_ty} {c_reg}, i32 0)"
        );
        self.store_place(&Place::local(dst), &d, &d_ty, w)
    }

    /// `OpCooperativeMatrixStoreKHR`, RowMajor.
    pub(super) fn emit_wmma_store_coopmat(
        &mut self,
        tile: &Place,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (v_reg, v_ty) = self.load_place(&Place::local(src), w)?;
        const ELEM_BYTES: u32 = 4; // float (the accumulator/D fragment is always F32 - config 13)
        let ptr = self.coopmat_byte_ptr(tile, ELEM_BYTES, w)?;
        let stride_bytes = stride * ELEM_BYTES;
        let mangled = "_Z33__spirv_CooperativeMatrixStoreKHR";
        self.declares.insert(format!(
            "declare dso_local spir_func void @{mangled}(ptr addrspace(11), {v_ty}, i32, i64, i32)"
        ));
        let _ = writeln!(
            w,
            "  tail call spir_func void @{mangled}(ptr addrspace(11) {ptr}, {v_ty} {v_reg}, i32 0, i64 {stride_bytes}, i32 0)"
        );
        Ok(())
    }

    /// The `CooperativeMatrixLayout` operand of a coopmat load/store: `RowMajorKHR` is 0, `ColumnMajorKHR` 1.
    fn coopmat_layout(major: FragmentMajor) -> u32 {
        match major {
            FragmentMajor::Row => 0,
            FragmentMajor::Col => 1,
        }
    }

    /// `OpCooperativeMatrixLoadKHR` from workgroup (LDS) storage: an A/B operand tile of f16, or an f32
    /// accumulator tile, at element `base` of `array`, in tile layout `major`. `stride` is in elements of the
    /// array's element type (the pointer addresses elements, not bytes: the LDS arrays are typed). The
    /// cooperative matrix is logical, so a column-major tile loads into the same fragment a row-major one
    /// would.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn emit_wmma_load_lds_coopmat(
        &mut self,
        role: FragmentRole,
        array: u8,
        base: &Operand,
        major: FragmentMajor,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (ty, mangled) = match role {
            FragmentRole::Matrix(WmmaMat::A) => (
                self.coopmat_ty("half", 0),
                "_Z32__spirv_CooperativeMatrixLoadKHR_4",
            ),
            FragmentRole::Matrix(WmmaMat::B) => (
                self.coopmat_ty("half", 1),
                "_Z32__spirv_CooperativeMatrixLoadKHR_5",
            ),
            FragmentRole::Accumulator => (
                self.coopmat_ty("float", 2),
                "_Z32__spirv_CooperativeMatrixLoadKHR_3",
            ),
        };
        let (ptr, _) = self.lds_tile_ptr(array, base, w)?;
        let layout = Self::coopmat_layout(major);
        self.declares.insert(format!(
            "declare dso_local spir_func {ty} @{mangled}(ptr addrspace(3), i32, i64, i32)"
        ));
        let d = self.fresh();
        let _ = writeln!(
            w,
            "  {d} = tail call spir_func {ty} @{mangled}(ptr addrspace(3) {ptr}, i32 {layout}, i64 {stride}, i32 0)"
        );
        self.store_place(&Place::local(dst), &d, &ty, w)
    }

    /// `OpCooperativeMatrixStoreKHR` of the f32 accumulator to workgroup (LDS) storage at element `base` of
    /// the f32 `array`, in tile layout `major`.
    pub(super) fn emit_wmma_store_lds_coopmat(
        &mut self,
        array: u8,
        base: &Operand,
        major: FragmentMajor,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (v_reg, v_ty) = self.load_place(&Place::local(src), w)?;
        let (ptr, _) = self.lds_tile_ptr(array, base, w)?;
        let layout = Self::coopmat_layout(major);
        let mangled = "_Z33__spirv_CooperativeMatrixStoreKHR_1";
        self.declares.insert(format!(
            "declare dso_local spir_func void @{mangled}(ptr addrspace(3), {v_ty}, i32, i64, i32)"
        ));
        let _ = writeln!(
            w,
            "  tail call spir_func void @{mangled}(ptr addrspace(3) {ptr}, {v_ty} {v_reg}, i32 {layout}, i64 {stride}, i32 0)"
        );
        Ok(())
    }

    // ---- dispatch (target-neutral entry points, card 530) -------------------------------------------

    #[expect(clippy::too_many_arguments)]
    pub(super) fn emit_wmma_load(
        &mut self,
        which: poot_kernel_ir::WmmaMat,
        dtype: poot_kernel_ir::WmmaDtype,
        shape: WmmaShape,
        tile: &Place,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_target(dtype)?;
        self.wmma_check_shape(shape)?;
        self.check_wmma_load_dtype(tile, dtype)?;
        if self.target == Target::SpirvVulkan {
            return self.emit_wmma_load_coopmat(which, tile, stride, dst, w);
        }
        if self.amd_arch().is_some() {
            self.require_amd_wmma()?;
            return self.emit_wmma_load_amdgcn(which, tile, stride, dst, w);
        }
        let (ptr, _, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&ptr, addr, w);
        self.emit_wmma_load_from_nvptx(which, dtype, FragmentMajor::Row, &gptr, stride, dst, w)
    }

    /// The element pointer of workgroup-local array `array` at index `base`, and the array's element LLVM type.
    /// The pointer is in LDS (`addrspace(3)`); the `.p0` NVPTX intrinsics take it addrspacecast to generic.
    fn lds_tile_ptr(
        &mut self,
        array: u8,
        base: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let (base_idx, _) = self.operand(base, w)?;
        let name = self.body.name.clone();
        let decl = &self.body.workgroup_locals[array as usize];
        let (n, elem) = (decl.len, self.scalar_llty(&decl.elem_ty)?);
        let us = self.usize_llty();
        let ptr = self.fresh();
        let _ = writeln!(
            w,
            "  {ptr} = getelementptr [{n} x {elem}], ptr addrspace(3) @{name}_lds{array}, {us} 0, {us} {base_idx}"
        );
        Ok((ptr, elem))
    }

    /// Load a fragment from a workgroup-local (LDS) array: the tile's first element at index `base`, tile
    /// layout `major` (the fragment-IR contract: LDS fragment ops with a base and a major, on every target).
    /// An A/B role fills an operand fragment; the accumulator role reloads an f32 tile, which is how an
    /// `Opaque`-layout target applies a per-row scale to an accumulator.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn emit_wmma_load_lds(
        &mut self,
        role: FragmentRole,
        dtype: poot_kernel_ir::WmmaDtype,
        shape: WmmaShape,
        array: u8,
        base: &Operand,
        major: FragmentMajor,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_target(dtype)?;
        self.wmma_check_shape(shape)?;
        self.check_wmma_load_lds_dtype(array, role, dtype)?;
        if self.target == Target::SpirvVulkan {
            return self.emit_wmma_load_lds_coopmat(role, array, base, major, stride, dst, w);
        }
        let (ptr, elem) = self.lds_tile_ptr(array, base, w)?;
        if self.amd_arch().is_some() {
            self.require_amd_wmma()?;
            return match role {
                FragmentRole::Matrix(which) => self.emit_wmma_load_ab_amdgcn(
                    which,
                    major,
                    stride,
                    AmdTilePtr::lds(&ptr, &elem),
                    dst,
                    w,
                ),
                FragmentRole::Accumulator => self.emit_wmma_load_acc_amdgcn(
                    major,
                    stride,
                    AmdTilePtr::lds(&ptr, &elem),
                    dst,
                    w,
                ),
            };
        }
        let gptr = self.wmma_generic_ptr(&ptr, 3, w);
        match role {
            FragmentRole::Matrix(which) => {
                self.emit_wmma_load_from_nvptx(which, dtype, major, &gptr, stride, dst, w)
            }
            FragmentRole::Accumulator => {
                self.emit_wmma_load_acc_nvptx(major, &gptr, stride, dst, w)
            }
        }
    }

    /// NVPTX `wmma.load.c`: the f32 accumulator fragment from `gptr` (generic) in tile layout `major`.
    fn emit_wmma_load_acc_nvptx(
        &mut self,
        major: FragmentMajor,
        gptr: &str,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let layout = nvptx_layout(major);
        let fty = Self::NVPTX_ACC_STRUCT_TY;
        let intrin = format!("llvm.nvvm.wmma.m16n16k16.load.c.{layout}.stride.f32.p0");
        self.declares
            .insert(format!("declare {fty} @{intrin}(ptr, i32)"));
        let f = self.fresh();
        let _ = writeln!(w, "  {f} = call {fty} @{intrin}(ptr {gptr}, i32 {stride})");
        self.store_place(&Place::local(dst), &f, fty, w)
    }

    /// The per-`dtype` NVPTX WMMA A/B fragment field list: the LLVM element type each aggregate field of the
    /// `load.{a,b}` intrinsic's return struct (and the `mma` intrinsic's matching operand list) has, per the
    /// NVVM WMMA fragment table (`m16n16k16:a/b:bf16` = 4 x i32; `m16n16k16:a/b:f16` = 8 x `<2 x half>`).
    fn nvptx_ab_fields(dtype: poot_kernel_ir::WmmaDtype) -> &'static [&'static str] {
        match dtype {
            poot_kernel_ir::WmmaDtype::Bf16 => &["i32", "i32", "i32", "i32"],
            poot_kernel_ir::WmmaDtype::F16 => &["<2 x half>"; 8],
        }
    }

    fn nvptx_ab_struct_ty(dtype: poot_kernel_ir::WmmaDtype) -> String {
        format!("{{{}}}", Self::nvptx_ab_fields(dtype).join(", "))
    }

    const NVPTX_ACC_STRUCT_TY: &'static str =
        "{float, float, float, float, float, float, float, float}";

    /// NVPTX WMMA fragment load: call the `load.{a,b}.row.stride.<dtype>.p0` intrinsic on `gptr` and store the
    /// returned aggregate straight into `dst`'s alloca (one store, no per-lane spill - `dst`'s alloca is
    /// exactly this aggregate type, see `fragment_llty`). `gptr` is a generic (addrspace 0) pointer.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn emit_wmma_load_from_nvptx(
        &mut self,
        which: poot_kernel_ir::WmmaMat,
        dtype: poot_kernel_ir::WmmaDtype,
        major: FragmentMajor,
        gptr: &str,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let mat = match which {
            poot_kernel_ir::WmmaMat::A => "a",
            poot_kernel_ir::WmmaMat::B => "b",
        };
        let dty_name = match dtype {
            poot_kernel_ir::WmmaDtype::Bf16 => "bf16",
            poot_kernel_ir::WmmaDtype::F16 => "f16",
        };
        let fty = Self::nvptx_ab_struct_ty(dtype);
        // Generic (.p0) addrspace: the only wmma load/store intrinsic variant llc lowers for all spaces (the .p1/.p3
        // variants are not all present); a generic pointer covers global + shared.
        let layout = nvptx_layout(major);
        let intrin = format!("llvm.nvvm.wmma.m16n16k16.load.{mat}.{layout}.stride.{dty_name}.p0");
        self.declares
            .insert(format!("declare {fty} @{intrin}(ptr, i32)"));
        let f = self.fresh();
        let _ = writeln!(w, "  {f} = call {fty} @{intrin}(ptr {gptr}, i32 {stride})");
        self.store_place(&Place::local(dst), &f, &fty, w)
    }

    /// AMDGPU WMMA fragment load from a global tile.
    pub(super) fn emit_wmma_load_amdgcn(
        &mut self,
        which: poot_kernel_ir::WmmaMat,
        tile: &Place,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (base_ptr, elem_llty, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&base_ptr, addr, w);
        self.emit_wmma_load_ab_amdgcn(
            which,
            FragmentMajor::Row,
            stride,
            AmdTilePtr::generic(&gptr, &elem_llty),
            dst,
            w,
        )
    }

    /// The lane id within its wave32 (`workitem.id.x & 31`).
    fn amdgcn_lane(&mut self, w: &mut String) -> String {
        self.declares
            .insert("declare i32 @llvm.amdgcn.workitem.id.x()".into());
        let lane_raw = self.fresh();
        let _ = writeln!(w, "  {lane_raw} = call i32 @llvm.amdgcn.workitem.id.x()");
        let lane = self.fresh();
        let _ = writeln!(w, "  {lane} = and i32 {lane_raw}, 31");
        lane
    }

    /// AMDGPU WMMA A/B fragment load (GFX11 wave32, 16x16x16) from `tile`, in tile layout `major`. Each lane
    /// holds 16 values of the tile's element type, those of one row of A (`m = lane % 16`, all 16 `k`) or one
    /// column of B (`n = lane % 16`, all 16 `k`); the two half-waves replicate. Which memory axis a lane's
    /// 16 values step along follows from the layout: `k` is contiguous for A row-major and B column-major
    /// (`r * stride + j`), strided for A column-major and B row-major (`j * stride + r`). Builds the native
    /// `<16 x elem>` vector, bitcasts it once to the fragment's raw-bits alloca type `<8 x i32>` and stores
    /// that: one store, no per-lane spill.
    pub(super) fn emit_wmma_load_ab_amdgcn(
        &mut self,
        which: WmmaMat,
        major: FragmentMajor,
        stride: u32,
        tile: AmdTilePtr<'_>,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let AmdTilePtr {
            ptr_ty,
            ptr,
            elem: elem_llty,
        } = tile;
        let lane = self.amdgcn_lane(w);
        let r = self.fresh();
        let _ = writeln!(w, "  {r} = and i32 {lane}, 15");
        let us = self.usize_llty();
        let k_contiguous = matches!(
            (which, major),
            (WmmaMat::A, FragmentMajor::Row) | (WmmaMat::B, FragmentMajor::Col)
        );
        // `k` contiguous: one `r * stride` (the lane's row/column start), then `+ j`. `k` strided: `r + j *
        // stride`, a constant per element.
        let r_stride = if k_contiguous {
            let o = self.fresh();
            let _ = writeln!(w, "  {o} = mul i32 {r}, {stride}");
            o
        } else {
            String::new()
        };

        let vec_ty = format!("<16 x {elem_llty}>");
        let mut cur_vec = String::new();
        for j in 0..16u32 {
            let offset = if k_contiguous {
                let o = self.fresh();
                let _ = writeln!(w, "  {o} = add i32 {r_stride}, {j}");
                o
            } else {
                let o = self.fresh();
                let _ = writeln!(w, "  {o} = add i32 {r}, {}", j * stride);
                o
            };
            let offset_us = self.fresh();
            let _ = writeln!(w, "  {offset_us} = zext i32 {offset} to {us}");
            let addr_j = self.fresh();
            let _ = writeln!(
                w,
                "  {addr_j} = getelementptr {elem_llty}, {ptr_ty} {ptr}, {us} {offset_us}"
            );
            let val = self.fresh();
            let _ = writeln!(w, "  {val} = load {elem_llty}, {ptr_ty} {addr_j}");
            let new_vec = self.fresh();
            let prev = if j == 0 { "poison" } else { cur_vec.as_str() };
            let _ = writeln!(
                w,
                "  {new_vec} = insertelement {vec_ty} {prev}, {elem_llty} {val}, i32 {j}"
            );
            cur_vec = new_vec;
        }

        let frag_ty = "<8 x i32>";
        let frag = self.fresh();
        let _ = writeln!(w, "  {frag} = bitcast {vec_ty} {cur_vec} to {frag_ty}");
        self.store_place(&Place::local(dst), &frag, frag_ty, w)
    }

    /// The memory offset (in elements) of accumulator element `i` of this lane in a 16x16 tile of layout
    /// `major` and `stride`, under the declared [`accumulator_map`]: row `2 i + lane / 16`, column
    /// `lane % 16`. `half` is `lane / 16`, `col` is `lane % 16`.
    fn amdgcn_acc_offset(
        &mut self,
        i: u32,
        half: &str,
        col: &str,
        major: FragmentMajor,
        stride: u32,
        w: &mut String,
    ) -> String {
        let row = self.fresh();
        let _ = writeln!(
            w,
            "  {row} = add i32 {half}, {}",
            accumulator_map::position(0, i).0
        );
        let (major_idx, minor_idx) = match major {
            FragmentMajor::Row => (row.as_str(), col),
            FragmentMajor::Col => (col, row.as_str()),
        };
        let scaled = self.fresh();
        let _ = writeln!(w, "  {scaled} = mul i32 {major_idx}, {stride}");
        let offset = self.fresh();
        let _ = writeln!(w, "  {offset} = add i32 {scaled}, {minor_idx}");
        offset
    }

    /// `(lane / 16, lane % 16)`: the row-set and the column every [`accumulator_map`] element of this lane uses.
    fn amdgcn_acc_lane_parts(&mut self, w: &mut String) -> (String, String) {
        let lane = self.amdgcn_lane(w);
        let col = self.fresh();
        let _ = writeln!(
            w,
            "  {col} = and i32 {lane}, {}",
            accumulator_map::LANES_PER_ROW_SET - 1
        );
        let half = self.fresh();
        let _ = writeln!(w, "  {half} = lshr i32 {lane}, 4");
        (half, col)
    }

    /// AMDGPU accumulator load: lane l's `<8 x float>` registers from `tile` under the declared
    /// [`accumulator_map`], tile layout `major`.
    pub(super) fn emit_wmma_load_acc_amdgcn(
        &mut self,
        major: FragmentMajor,
        stride: u32,
        tile: AmdTilePtr<'_>,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let AmdTilePtr {
            ptr_ty,
            ptr,
            elem: elem_llty,
        } = tile;
        let (half, col) = self.amdgcn_acc_lane_parts(w);
        let us = self.usize_llty();
        let mut cur = "poison".to_string();
        for i in 0..accumulator_map::ELEMENTS_PER_LANE {
            let offset = self.amdgcn_acc_offset(i, &half, &col, major, stride, w);
            let offset_us = self.fresh();
            let _ = writeln!(w, "  {offset_us} = zext i32 {offset} to {us}");
            let addr_i = self.fresh();
            let _ = writeln!(
                w,
                "  {addr_i} = getelementptr {elem_llty}, {ptr_ty} {ptr}, {us} {offset_us}"
            );
            let val = self.fresh();
            let _ = writeln!(w, "  {val} = load {elem_llty}, {ptr_ty} {addr_i}");
            let next = self.fresh();
            let _ = writeln!(
                w,
                "  {next} = insertelement <8 x float> {cur}, float {val}, i32 {i}"
            );
            cur = next;
        }
        self.store_place(&Place::local(dst), &cur, "<8 x float>", w)
    }

    /// Accumulator element read: `dst = acc[idx]` of this lane, under the declared [`accumulator_map`].
    /// AMDGCN only: the other targets' fragment registers have no public layout.
    pub(super) fn emit_wmma_elem_read(
        &mut self,
        src: Local,
        idx: u8,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.require_declared_fragment_layout("WmmaElemRead")?;
        let (acc, _) = self.load_place(&Place::local(src), w)?;
        let v = self.fresh();
        let _ = writeln!(w, "  {v} = extractelement <8 x float> {acc}, i32 {idx}");
        self.store_place(&Place::local(dst), &v, "float", w)
    }

    /// Accumulator element write: `acc[idx] = value` of this lane, under the declared [`accumulator_map`].
    pub(super) fn emit_wmma_elem_write(
        &mut self,
        dst: Local,
        idx: u8,
        value: &Operand,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.require_declared_fragment_layout("WmmaElemWrite")?;
        let (v, _) = self.operand(value, w)?;
        let (acc, _) = self.load_place(&Place::local(dst), w)?;
        let next = self.fresh();
        let _ = writeln!(
            w,
            "  {next} = insertelement <8 x float> {acc}, float {v}, i32 {idx}"
        );
        self.store_place(&Place::local(dst), &next, "<8 x float>", w)
    }

    /// Element access names `(lane, element)`, which only a `Declared` fragment layout defines. AMDGCN gfx11
    /// publishes one; SPIR-V cooperative matrices and NVPTX `wmma` fragments are opaque, so they refuse
    /// (typed) and the schedule stages the accumulator through LDS.
    fn require_declared_fragment_layout(&self, op: &str) -> Result<(), EmitError> {
        if self.amd_arch().is_some() {
            self.require_amd_wmma()
        } else {
            Err(EmitError::Unsupported(format!(
                "{op} needs a declared accumulator layout (AMDGCN gfx11); {:?} fragments are opaque: stage                  the accumulator through WmmaStoreLds / an accumulator WmmaLoadLds instead",
                self.target
            )))
        }
    }

    #[expect(clippy::too_many_arguments)]
    pub(super) fn emit_wmma_mma(
        &mut self,
        dtype: poot_kernel_ir::WmmaDtype,
        shape: WmmaShape,
        a: Local,
        b: Local,
        c: Local,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_target(dtype)?;
        self.wmma_check_shape(shape)?;
        if self.target == Target::SpirvVulkan {
            return self.emit_wmma_mma_coopmat(a, b, c, dst, w);
        }
        if self.amd_arch().is_some() {
            self.require_amd_wmma()?;
            return self.emit_wmma_mma_amdgcn(dtype, a, b, c, dst, w);
        }
        self.emit_wmma_mma_nvptx(dtype, a, b, c, dst, w)
    }

    /// NVPTX WMMA MMA: `dtype` inputs / f32 accumulate, row-major. Loads `a`/`b`/`c` from their allocas,
    /// `extractvalue`s each field to build the intrinsic's flat operand list, then stores the returned f32
    /// aggregate straight into `dst`'s alloca.
    pub(super) fn emit_wmma_mma_nvptx(
        &mut self,
        dtype: poot_kernel_ir::WmmaDtype,
        a: Local,
        b: Local,
        c: Local,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let ab_ty = Self::nvptx_ab_struct_ty(dtype);
        let ab_fields = Self::nvptx_ab_fields(dtype);
        let (a_val, _) = self.load_place(&Place::local(a), w)?;
        let (b_val, _) = self.load_place(&Place::local(b), w)?;
        let (c_val, _) = self.load_place(&Place::local(c), w)?;
        let mut args = Vec::new();
        for (src_reg, src_ty) in [(&a_val, &ab_ty), (&b_val, &ab_ty)] {
            for (i, fty) in ab_fields.iter().enumerate() {
                let e = self.fresh();
                let _ = writeln!(w, "  {e} = extractvalue {src_ty} {src_reg}, {i}");
                args.push(format!("{fty} {e}"));
            }
        }
        for i in 0..8 {
            let e = self.fresh();
            let _ = writeln!(
                w,
                "  {e} = extractvalue {} {c_val}, {i}",
                Self::NVPTX_ACC_STRUCT_TY
            );
            args.push(format!("float {e}"));
        }
        // Per NVVM's WMMA_NAME signature rule: an f16 A dtype is disambiguated by (D, C) type
        // ("f32.f32" here, both f32-accumulate), matching PTX ISA's
        // `wmma.mma.sync.aligned.row.row.m16n16k16.f32.f16.f16.f32`; bf16 has only one accumulate type, so
        // the name carries no C/D suffix.
        // The multiply is specialised on the layouts the operands were loaded in (`.alayout.blayout`), which
        // the operands' declared fragment types carry.
        let layouts = format!(
            "{}.{}",
            nvptx_layout(self.operand_fragment_major(a)),
            nvptx_layout(self.operand_fragment_major(b))
        );
        let intrin = match dtype {
            poot_kernel_ir::WmmaDtype::Bf16 => {
                format!("llvm.nvvm.wmma.m16n16k16.mma.{layouts}.bf16")
            }
            poot_kernel_ir::WmmaDtype::F16 => {
                format!("llvm.nvvm.wmma.m16n16k16.mma.{layouts}.f32.f32")
            }
        };
        let arg_sig: Vec<&str> = ab_fields
            .iter()
            .copied()
            .chain(ab_fields.iter().copied())
            .chain(std::iter::repeat_n("float", 8))
            .collect();
        self.declares.insert(format!(
            "declare {} @{intrin}({})",
            Self::NVPTX_ACC_STRUCT_TY,
            arg_sig.join(", ")
        ));
        let d = self.fresh();
        let _ = writeln!(
            w,
            "  {d} = call {} @{intrin}({})",
            Self::NVPTX_ACC_STRUCT_TY,
            args.join(", ")
        );
        self.store_place(&Place::local(dst), &d, Self::NVPTX_ACC_STRUCT_TY, w)
    }

    /// The layout an A/B operand fragment local was declared as loaded from.
    fn operand_fragment_major(&self, operand: Local) -> FragmentMajor {
        match self.body.locals[operand.index as usize].ty {
            Ty::Fragment(fragment) => fragment.major,
            _ => FragmentMajor::Row, // the verifier only admits fragment locals here
        }
    }

    /// AMDGPU WMMA MMA: `a`/`b` are already alloca'd as the raw-bits `<8 x i32>` fragment type; bitcast each
    /// to the dtype-typed vector the intrinsic wants (`<16 x i16>` for bf16, `<16 x half>` for f16 - both
    /// 256 bits, same as the `<8 x i32>` storage, so the bitcast is a pure reinterpret), call
    /// `llvm.amdgcn.wmma.f32.16x16x16.<dtype>`, and store the `<8 x float>` result straight into `dst`.
    pub(super) fn emit_wmma_mma_amdgcn(
        &mut self,
        dtype: poot_kernel_ir::WmmaDtype,
        a: Local,
        b: Local,
        c: Local,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (op_ty, intrin) = match dtype {
            poot_kernel_ir::WmmaDtype::Bf16 => (
                "<16 x i16>",
                "llvm.amdgcn.wmma.f32.16x16x16.bf16.v8f32.v16i16",
            ),
            poot_kernel_ir::WmmaDtype::F16 => (
                "<16 x half>",
                "llvm.amdgcn.wmma.f32.16x16x16.f16.v8f32.v16f16",
            ),
        };
        let (a_raw, _) = self.load_place(&Place::local(a), w)?;
        let (b_raw, _) = self.load_place(&Place::local(b), w)?;
        let (c_val, _) = self.load_place(&Place::local(c), w)?;
        let a_vec = self.fresh();
        let _ = writeln!(w, "  {a_vec} = bitcast <8 x i32> {a_raw} to {op_ty}");
        let b_vec = self.fresh();
        let _ = writeln!(w, "  {b_vec} = bitcast <8 x i32> {b_raw} to {op_ty}");

        let ret_ty = "<8 x float>";
        self.declares.insert(format!(
            "declare {ret_ty} @{intrin}({op_ty}, {op_ty}, <8 x float>)"
        ));
        let d = self.fresh();
        let _ = writeln!(
            w,
            "  {d} = call {ret_ty} @{intrin}({op_ty} {a_vec}, {op_ty} {b_vec}, <8 x float> {c_val})"
        );
        self.store_place(&Place::local(dst), &d, ret_ty, w)
    }

    pub(super) fn emit_wmma_store(
        &mut self,
        dtype: poot_kernel_ir::WmmaDtype,
        shape: WmmaShape,
        tile: &Place,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_target(dtype)?;
        self.wmma_check_shape(shape)?;
        if self.target == Target::SpirvVulkan {
            return self.emit_wmma_store_coopmat(tile, stride, src, w);
        }
        if self.amd_arch().is_some() {
            self.require_amd_wmma()?;
            return self.emit_wmma_store_amdgcn(tile, stride, src, w);
        }
        self.emit_wmma_store_nvptx(tile, stride, src, w)
    }

    /// NVPTX WMMA store: `store.d.row.stride.f32.p0` intrinsic. Dtype-independent (the D fragment is always
    /// f32 in this codegen, on every target).
    pub(super) fn emit_wmma_store_nvptx(
        &mut self,
        tile: &Place,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (ptr, _, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&ptr, addr, w);
        self.emit_wmma_store_to_nvptx(FragmentMajor::Row, &gptr, stride, src, w)
    }

    /// NVPTX `wmma.store.d` of the f32 accumulator `src` to the generic pointer `gptr`, tile layout `major`.
    fn emit_wmma_store_to_nvptx(
        &mut self,
        major: FragmentMajor,
        gptr: &str,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (src_val, _) = self.load_place(&Place::local(src), w)?;
        let mut vals = Vec::new();
        for i in 0..8 {
            let e = self.fresh();
            let _ = writeln!(
                w,
                "  {e} = extractvalue {} {src_val}, {i}",
                Self::NVPTX_ACC_STRUCT_TY
            );
            vals.push(format!("float {e}"));
        }
        let layout = nvptx_layout(major);
        let intrin = format!("llvm.nvvm.wmma.m16n16k16.store.d.{layout}.stride.f32.p0");
        self.declares.insert(format!(
            "declare void @{intrin}(ptr, float, float, float, float, float, float, float, float, i32)"
        ));
        let _ = writeln!(
            w,
            "  call void @{intrin}(ptr {gptr}, {}, i32 {stride})",
            vals.join(", ")
        );
        Ok(())
    }

    /// AMDGPU WMMA store to a global tile (row-major): each wave32 lane stores its 8 f32 accumulator elements
    /// at the position the declared [`accumulator_map`] gives (lanes 0-15 the even rows, lanes 16-31 the odd).
    pub(super) fn emit_wmma_store_amdgcn(
        &mut self,
        tile: &Place,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (base_ptr, elem_llty, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&base_ptr, addr, w);
        self.emit_wmma_store_acc_amdgcn(
            FragmentMajor::Row,
            stride,
            AmdTilePtr::generic(&gptr, &elem_llty),
            src,
            w,
        )
    }

    /// Store the f32 accumulator `src` to `tile` in tile layout `major`, element `i` of this lane at the
    /// [`accumulator_map`] position.
    fn emit_wmma_store_acc_amdgcn(
        &mut self,
        major: FragmentMajor,
        stride: u32,
        tile: AmdTilePtr<'_>,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let AmdTilePtr {
            ptr_ty,
            ptr,
            elem: elem_llty,
        } = tile;
        let (src_val, _) = self.load_place(&Place::local(src), w)?;
        let (half, col) = self.amdgcn_acc_lane_parts(w);
        let us = self.usize_llty();
        for i in 0..accumulator_map::ELEMENTS_PER_LANE {
            let v = self.fresh();
            let _ = writeln!(w, "  {v} = extractelement <8 x float> {src_val}, i32 {i}");
            let offset = self.amdgcn_acc_offset(i, &half, &col, major, stride, w);
            let offset_us = self.fresh();
            let _ = writeln!(w, "  {offset_us} = zext i32 {offset} to {us}");
            let addr_i = self.fresh();
            let _ = writeln!(
                w,
                "  {addr_i} = getelementptr {elem_llty}, {ptr_ty} {ptr}, {us} {offset_us}"
            );
            let _ = writeln!(w, "  store {elem_llty} {v}, {ptr_ty} {addr_i}");
        }
        Ok(())
    }

    /// Store the f32 accumulator fragment to a workgroup-local (LDS) f32 `array`: the tile's first element at
    /// index `base`, tile layout `major`. The narrow epilogue and an `Opaque` target's element access stage an
    /// accumulator in LDS this way.
    #[expect(clippy::too_many_arguments)]
    pub(super) fn emit_wmma_store_lds(
        &mut self,
        dtype: poot_kernel_ir::WmmaDtype,
        shape: WmmaShape,
        array: u8,
        base: &Operand,
        major: FragmentMajor,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_target(dtype)?;
        self.wmma_check_shape(shape)?;
        self.check_wmma_lds_accumulator_array(array)?;
        if self.target == Target::SpirvVulkan {
            return self.emit_wmma_store_lds_coopmat(array, base, major, stride, src, w);
        }
        let (ptr, elem) = self.lds_tile_ptr(array, base, w)?;
        if self.amd_arch().is_some() {
            self.require_amd_wmma()?;
            return self.emit_wmma_store_acc_amdgcn(
                major,
                stride,
                AmdTilePtr::lds(&ptr, &elem),
                src,
                w,
            );
        }
        let gptr = self.wmma_generic_ptr(&ptr, 3, w);
        self.emit_wmma_store_to_nvptx(major, &gptr, stride, src, w)
    }

    /// Seed a zero-valued f32 accumulator fragment (card 154; every target, card 530). SPIR-V builds it
    /// through `OpCompositeConstruct` (no addressable components to zero directly); AMDGCN/NVPTX store a
    /// plain `zeroinitializer` aggregate/vector constant into the alloca - no device instruction, the
    /// fragment starts as literal zero bits.
    pub(super) fn emit_wmma_zero(
        &mut self,
        dtype: poot_kernel_ir::WmmaDtype,
        shape: WmmaShape,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_target(dtype)?;
        self.wmma_check_shape(shape)?;
        if self.target == Target::SpirvVulkan {
            return self.emit_wmma_zero_coopmat(dst, w);
        }
        let llty = if self.amd_arch().is_some() {
            "<8 x float>".to_string()
        } else {
            Self::NVPTX_ACC_STRUCT_TY.to_string()
        };
        self.store_place(&Place::local(dst), "zeroinitializer", &llty, w)
    }

    /// addrspacecast a (global/shared) pointer to generic (addrspace 0) for the wmma `.p0` intrinsics; a
    /// no-op string if already generic.
    pub(super) fn wmma_generic_ptr(&mut self, ptr: &str, addr: u32, w: &mut String) -> String {
        if addr == 0 {
            return ptr.to_string();
        }
        let g = self.fresh();
        let _ = writeln!(
            w,
            "  {g} = addrspacecast ptr addrspace({addr}) {ptr} to ptr"
        );
        g
    }
}

/// A tile's base pointer for the AMDGCN fragment loads and stores: the pointer value, its address space
/// (`ptr` for generic/global, `ptr addrspace(3)` for LDS) and the tile's element LLVM type.
#[derive(Clone, Copy)]
pub(super) struct AmdTilePtr<'p> {
    ptr_ty: &'static str,
    ptr: &'p str,
    elem: &'p str,
}

impl<'p> AmdTilePtr<'p> {
    /// A generic (flat) pointer, as the global tile paths use.
    pub(super) fn generic(ptr: &'p str, elem: &'p str) -> Self {
        Self {
            ptr_ty: "ptr",
            ptr,
            elem,
        }
    }

    /// A workgroup-local (LDS, `addrspace(3)`) pointer.
    pub(super) fn lds(ptr: &'p str, elem: &'p str) -> Self {
        Self {
            ptr_ty: "ptr addrspace(3)",
            ptr,
            elem,
        }
    }
}

/// The NVVM intrinsic spelling of a tile layout.
fn nvptx_layout(major: FragmentMajor) -> &'static str {
    match major {
        FragmentMajor::Row => "row",
        FragmentMajor::Col => "col",
    }
}
