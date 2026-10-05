use std::collections::HashMap;
use std::fmt::Write as _;

use poot_kernel_ir::{Body, Statement, WmmaDtype, WmmaShape};

use super::{EmitError, Emitter, Local, Place, ProjectionElem, Target};

/// Every `Local` a body uses as a matrix-fragment handle, with its dtype and whether it is the f32
/// accumulator (a `WmmaZero`/`WmmaMma` `dst`) or an A/B operand fragment (a `WmmaLoad`/`WmmaLoadLds` `dst`,
/// carrying `dtype`). Run once before `emit_entry` so AMDGCN/NVPTX can alloca each fragment local at its
/// real physical size (see `Emitter::frag_layout`'s doc) instead of its marker `Ty`'s scalar size. A
/// fragment `Local` is always some statement's own `dst`: `WmmaMma`'s `a`/`b`/`c` and `WmmaStore`/
/// `WmmaStoreLds`'s `src` read a definition recorded here when it was produced, so they need no entry of
/// their own.
pub(super) fn scan_frag_layout(body: &Body) -> HashMap<u32, (WmmaDtype, bool)> {
    let mut layout = HashMap::new();
    for block in &body.blocks {
        for stmt in &block.statements {
            match stmt {
                Statement::WmmaLoad { dtype, dst, .. }
                | Statement::WmmaLoadLds { dtype, dst, .. } => {
                    layout.insert(dst.index, (*dtype, false));
                }
                Statement::WmmaZero { dtype, dst, .. } | Statement::WmmaMma { dtype, dst, .. } => {
                    layout.insert(dst.index, (*dtype, true));
                }
                _ => {}
            }
        }
    }
    layout
}

impl<'a> Emitter<'a> {
    // ---- WMMA tensor-core intrinsics (spec 025, NVPTX + AMDGPU) ------------------------------------
    // m16n16k16, row-major, f32 accumulate. Fragments are register-group aggregates/vectors, alloca'd like
    // any other local at the physical size `frag_alloca_llty` picks (see `emit_entry`); the load/store
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

    /// [`Self::check_wmma_dtype_matches`] for a workgroup-local (LDS) array (`WmmaLoadLds`'s `array`).
    pub(super) fn check_wmma_load_lds_dtype(
        &self,
        array: u8,
        dtype: poot_kernel_ir::WmmaDtype,
    ) -> Result<(), EmitError> {
        let elem_ty = self.body.workgroup_locals[array as usize].elem_ty.clone();
        self.check_wmma_dtype_matches(&elem_ty, dtype)
    }

    /// The AMD arch descriptor when the target is AMDGPU, else `None`; WMMA emit uses it to gate the tensor-core
    /// intrinsic on the detected capability family (spec 120).
    pub(super) fn amd_arch(&self) -> Option<crate::AmdArch> {
        match self.target {
            Target::AmdGcn(arch) => Some(arch),
            _ => None,
        }
    }

    /// Guard the AMDGPU WMMA path on the detected tensor-core family. Only the RDNA3/3.5 (gfx11xx) WMMA 16x16x16
    /// layout is emitted; RDNA4 (128b) and CDNA (MFMA) use different fragment packings, so they get a typed
    /// "unsupported target" error instead of the wrong intrinsic or a backend "cannot select" crash (FR-003).
    pub(super) fn require_amd_wmma(&self) -> Result<(), EmitError> {
        if let Some(arch) = self.amd_arch()
            && arch.tensor_core != poot_target::TensorCoreSupport::Wmma16x16x16Rdna3
        {
            return Err(EmitError::Unsupported(format!(
                "WMMA tensor-core codegen unsupported for AMD target {} ({:?}); only \
                 RDNA3/3.5 (gfx11xx) WMMA 16x16x16 is implemented",
                arch.mcpu(),
                arch.tensor_core,
            )));
        }
        Ok(())
    }

    // ---- AMDGCN/NVPTX fragment storage (card 530) --------------------------------------------------
    // A fragment `Local` is an ordinary alloca'd value on these two targets (unlike SPIR-V's opaque coopmat
    // handle, see `frag_regs`'s doc): `emit_entry` allocas it at the size below instead of its marker `Ty`'s
    // scalar size, and every op reads/writes it through the normal `load_place`/`store_place` path. This is
    // what lets a fragment loop-carry (the tiled `matmul_tensorcore` K-loop) with no special machinery.

    /// The alloca'd LLVM type for one fragment local on AMDGCN/NVPTX: `is_acc` (a `WmmaZero`/`WmmaMma` `dst`)
    /// is always the f32 accumulator/D fragment (8 lanes); otherwise an A/B operand fragment of `dtype`.
    /// AMDGCN packs every dtype into the same `<8 x i32>` raw-bits vector (16 lanes of a 2-byte element,
    /// bitcast to the dtype-typed vector only at the point of use - see `emit_wmma_load_amdgcn`); NVPTX's
    /// fragment shape is dtype-sized (bf16: 4 lanes of i32; f16: 8 lanes of `<2 x half>`, per the NVVM WMMA
    /// fragment table), so its aggregate type must match exactly what the load/mma intrinsics return/accept.
    pub(super) fn frag_alloca_llty(
        &self,
        dtype: WmmaDtype,
        is_acc: bool,
    ) -> Result<String, EmitError> {
        match self.target {
            Target::AmdGcn(_) => Ok(if is_acc {
                "<8 x float>".into()
            } else {
                "<8 x i32>".into()
            }),
            Target::Nvptx => Ok(if is_acc {
                "{float, float, float, float, float, float, float, float}".into()
            } else {
                match dtype {
                    WmmaDtype::Bf16 => "{i32, i32, i32, i32}".into(),
                    WmmaDtype::F16 => {
                        "{<2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, <2 x half>, \
                         <2 x half>, <2 x half>}"
                            .into()
                    }
                }
            }),
            Target::SpirvVulkan | Target::AieCore => Err(EmitError::Unsupported(
                "frag_alloca_llty is AMDGCN/NVPTX-only (SPIR-V fragments are opaque SSA values, see \
                 frag_regs)"
                    .into(),
            )),
        }
    }

    // ---- SPIR-V cooperative-matrix (card 154) --------------------------------------------------
    // RADV config 13: F16xF16->F32, 16x16x16, Subgroup scope. Unlike NVPTX/AMDGCN's alloca'd register-group
    // fragments, a coopmat value is one opaque `OpTypeCooperativeMatrixKHR` SSA value with no addressable
    // components, so it stays in `frag_regs` (an SSA-register cache) instead of memory.
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

    /// The live SSA register + LLVM type text for a `Local` holding a coopmat value - see
    /// `frag_regs`'s doc on the `Emitter` struct.
    pub(super) fn coopmat_reg(&self, l: Local) -> Result<(String, String), EmitError> {
        self.frag_regs.get(&l.index).cloned().ok_or_else(|| {
            EmitError::Unsupported(format!(
                "SPIR-V cooperative-matrix local %l{} read before a WmmaLoad/WmmaMma/WmmaZero \
                 populated it",
                l.index
            ))
        })
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
        self.frag_regs.insert(dst.index, (d, ty));
        Ok(())
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
        self.frag_regs.insert(dst.index, (d, ty));
        Ok(())
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
        let (a_reg, a_ty) = self.coopmat_reg(a)?;
        let (b_reg, b_ty) = self.coopmat_reg(b)?;
        let (c_reg, c_ty) = self.coopmat_reg(c)?;
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
        self.frag_regs.insert(dst.index, (d, d_ty));
        Ok(())
    }

    /// `OpCooperativeMatrixStoreKHR`, RowMajor.
    pub(super) fn emit_wmma_store_coopmat(
        &mut self,
        tile: &Place,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (v_reg, v_ty) = self.coopmat_reg(src)?;
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

    // ---- dispatch (target-neutral entry points, card 530) -------------------------------------------

    #[allow(clippy::too_many_arguments)]
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
            return self.emit_wmma_load_amdgcn(which, dtype, tile, stride, dst, w);
        }
        let (ptr, _, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&ptr, addr, w);
        self.emit_wmma_load_from_nvptx(which, dtype, &gptr, stride, dst, w)
    }

    /// WMMA-load an A/B fragment from a workgroup-local (LDS) array (the dequant gemm stages a bf16 B-tile
    /// there); same intrinsic as `emit_wmma_load` with the LDS global addrspacecast to generic. NVPTX-only.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn emit_wmma_load_lds(
        &mut self,
        which: poot_kernel_ir::WmmaMat,
        dtype: poot_kernel_ir::WmmaDtype,
        shape: WmmaShape,
        array: u8,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_shape(shape)?;
        self.check_wmma_load_lds_dtype(array, dtype)?;
        if self.target != Target::Nvptx {
            return Err(EmitError::Unsupported(
                "WmmaLoadLds is NVPTX-only (no AMDGPU LDS WMMA path yet)".into(),
            ));
        }
        let name = self.body.name.clone();
        let decl = &self.body.workgroup_locals[array as usize];
        let (n, elem) = (decl.len, self.scalar_llty(&decl.elem_ty)?);
        let us = self.usize_llty();
        let base = self.fresh();
        let _ = writeln!(
            w,
            "  {base} = getelementptr [{n} x {elem}], ptr addrspace(3) @{name}_lds{array}, {us} 0, {us} 0"
        );
        let gptr = self.fresh();
        let _ = writeln!(w, "  {gptr} = addrspacecast ptr addrspace(3) {base} to ptr");
        self.emit_wmma_load_from_nvptx(which, dtype, &gptr, stride, dst, w)
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
    /// exactly this aggregate type, see `frag_alloca_llty`). `gptr` is a generic (addrspace 0) pointer.
    pub(super) fn emit_wmma_load_from_nvptx(
        &mut self,
        which: poot_kernel_ir::WmmaMat,
        dtype: poot_kernel_ir::WmmaDtype,
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
        let intrin = format!("llvm.nvvm.wmma.m16n16k16.load.{mat}.row.stride.{dty_name}.p0");
        self.declares
            .insert(format!("declare {fty} @{intrin}(ptr, i32)"));
        let f = self.fresh();
        let _ = writeln!(w, "  {f} = call {fty} @{intrin}(ptr {gptr}, i32 {stride})");
        self.store_place(&Place::local(dst), &f, &fty, w)
    }

    /// AMDGPU WMMA fragment load (GFX11 wave32, 16x16x16). Each lane holds 16 values of `dtype`'s element
    /// (16 bf16/f16 lanes, matching the register width WMMA reads). A (m16 x k16): loads A[row=lane%16,
    /// 0..15] -> A[i, k]. B (k16 x n16): loads B[0..15, col=lane%16] -> B[k, j] (transposed). Builds the
    /// native `<16 x elem>` vector (`elem` from `dtype`, not a bitcast trick: `half`/`bfloat` are both
    /// ordinary LLVM scalar types this backend accepts, see `scalar_llty`), then bitcasts it once to the
    /// fragment's raw-bits alloca type `<8 x i32>` (same total width either way) and stores that - one
    /// store, no per-lane spill.
    pub(super) fn emit_wmma_load_amdgcn(
        &mut self,
        which: poot_kernel_ir::WmmaMat,
        dtype: poot_kernel_ir::WmmaDtype,
        tile: &Place,
        stride: u32,
        dst: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (base_ptr, elem_llty, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&base_ptr, addr, w);
        let _ = dtype; // elem_llty already reflects the tile's declared element type (bf16 or f16).

        self.declares
            .insert("declare i32 @llvm.amdgcn.workitem.id.x()".into());
        let lane_raw = self.fresh();
        let _ = writeln!(w, "  {lane_raw} = call i32 @llvm.amdgcn.workitem.id.x()");
        let lane = self.fresh();
        let _ = writeln!(w, "  {lane} = and i32 {lane_raw}, 31");
        let row = self.fresh();
        let _ = writeln!(w, "  {row} = and i32 {lane}, 15");
        let us = self.usize_llty();

        // For A: address = tile + row * stride + j  (row-major, j = 0..15)
        // For B: address = tile + j * stride + row  (transposed: B[j, row])
        let row_stride = match which {
            poot_kernel_ir::WmmaMat::A => {
                let rs = self.fresh();
                let _ = writeln!(w, "  {rs} = mul i32 {row}, {stride}");
                rs
            }
            poot_kernel_ir::WmmaMat::B => String::new(),
        };

        // Load 16 values of `elem_llty`, build `<16 x elem_llty>`, bitcast to the `<8 x i32>` storage type.
        let vec_ty = format!("<16 x {elem_llty}>");
        let mut cur_vec = String::new();

        for j in 0..16u32 {
            let offset = match which {
                poot_kernel_ir::WmmaMat::A => {
                    if j == 0 {
                        row_stride.clone()
                    } else {
                        let o = self.fresh();
                        let _ = writeln!(w, "  {o} = add i32 {row_stride}, {j}");
                        o
                    }
                }
                poot_kernel_ir::WmmaMat::B => {
                    // B: offset = j * stride + row
                    if j == 0 {
                        row.clone()
                    } else {
                        let j_stride = self.fresh();
                        let _ = writeln!(w, "  {j_stride} = mul i32 {j}, {stride}");
                        let o = self.fresh();
                        let _ = writeln!(w, "  {o} = add i32 {j_stride}, {row}");
                        o
                    }
                }
            };
            let offset_us = self.fresh();
            let _ = writeln!(w, "  {offset_us} = zext i32 {offset} to {us}");
            let addr_j = self.fresh();
            let _ = writeln!(
                w,
                "  {addr_j} = getelementptr {elem_llty}, ptr {gptr}, {us} {offset_us}"
            );
            let val = self.fresh();
            let _ = writeln!(w, "  {val} = load {elem_llty}, ptr {addr_j}");
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

    #[allow(clippy::too_many_arguments)]
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
        let intrin = match dtype {
            poot_kernel_ir::WmmaDtype::Bf16 => {
                "llvm.nvvm.wmma.m16n16k16.mma.row.row.bf16".to_string()
            }
            poot_kernel_ir::WmmaDtype::F16 => {
                "llvm.nvvm.wmma.m16n16k16.mma.row.row.f32.f32".to_string()
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
        let (ptr, _, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&ptr, addr, w);
        let intrin = "llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0";
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

    /// AMDGPU WMMA store: each wave32 lane stores 8 f32 values (`extractelement`d from the `<8 x float>`
    /// accumulator fragment) to the 16x16 tile.
    /// GFX11 wave32 WMMA output layout (from rocWMMA / rocKE):
    ///   lane l, slot i -> (row = 2*i + l/16, col = l%16)
    /// Lanes 0-15 write even rows (0,2,4,...,14), lanes 16-31 write odd rows (1,3,...,15).
    pub(super) fn emit_wmma_store_amdgcn(
        &mut self,
        tile: &Place,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        let (base_ptr, elem_llty, addr) = self.element_ptr(tile, w)?;
        let gptr = self.wmma_generic_ptr(&base_ptr, addr, w);
        let (src_val, _) = self.load_place(&Place::local(src), w)?;

        // Get lane ID.
        self.declares
            .insert("declare i32 @llvm.amdgcn.workitem.id.x()".into());
        let lane_raw = self.fresh();
        let _ = writeln!(w, "  {lane_raw} = call i32 @llvm.amdgcn.workitem.id.x()");
        let lane = self.fresh();
        let _ = writeln!(w, "  {lane} = and i32 {lane_raw}, 31");

        // frag = lane % 16 (column index)
        let frag = self.fresh();
        let _ = writeln!(w, "  {frag} = and i32 {lane}, 15");
        // half = lane / 16 (0 for even rows, 1 for odd rows)
        let half = self.fresh();
        let _ = writeln!(w, "  {half} = lshr i32 {lane}, 4");

        let us = self.usize_llty();

        for i in 0..8usize {
            let v = self.fresh();
            let _ = writeln!(w, "  {v} = extractelement <8 x float> {src_val}, i32 {i}");
            // row = 2*i + half
            let row = self.fresh();
            let _ = writeln!(w, "  {row} = add i32 {half}, {i_twice}", i_twice = i * 2);
            // col = frag
            // byte_offset = (row * stride + col) * sizeof(elem)
            let row_stride = self.fresh();
            let _ = writeln!(w, "  {row_stride} = mul i32 {row}, {stride}");
            let offset = self.fresh();
            let _ = writeln!(w, "  {offset} = add i32 {row_stride}, {frag}");
            let offset_us = self.fresh();
            let _ = writeln!(w, "  {offset_us} = zext i32 {offset} to {us}");
            let addr_j = self.fresh();
            let _ = writeln!(
                w,
                "  {addr_j} = getelementptr {elem_llty}, ptr {gptr}, {us} {offset_us}"
            );
            let _ = writeln!(w, "  store {elem_llty} {v}, ptr {addr_j}");
        }
        Ok(())
    }

    /// WMMA store the f32 D fragment to a workgroup-local (LDS, `addrspace(3)`) array (base index 0) with row
    /// `stride`. The same `store.d.row.stride.f32.p0` intrinsic as `emit_wmma_store`, but the destination is
    /// the LDS global addrspacecast to generic (the bf16 gemm narrow epilogue stages f32 in LDS here).
    /// NVPTX-only.
    pub(super) fn emit_wmma_store_lds(
        &mut self,
        shape: WmmaShape,
        array: u8,
        stride: u32,
        src: Local,
        w: &mut String,
    ) -> Result<(), EmitError> {
        self.wmma_check_shape(shape)?;
        if self.target != Target::Nvptx {
            return Err(EmitError::Unsupported(
                "WmmaStoreLds is NVPTX-only (no AMDGPU LDS WMMA path yet)".into(),
            ));
        }
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
        let name = self.body.name.clone();
        let decl = &self.body.workgroup_locals[array as usize];
        let (n, elem) = (decl.len, self.scalar_llty(&decl.elem_ty)?);
        let us = self.usize_llty();
        let base = self.fresh();
        let _ = writeln!(
            w,
            "  {base} = getelementptr [{n} x {elem}], ptr addrspace(3) @{name}_lds{array}, {us} 0, {us} 0"
        );
        let gptr = self.fresh();
        let _ = writeln!(w, "  {gptr} = addrspacecast ptr addrspace(3) {base} to ptr");
        let intrin = "llvm.nvvm.wmma.m16n16k16.store.d.row.stride.f32.p0";
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
