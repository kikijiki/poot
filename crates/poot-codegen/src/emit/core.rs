use std::fmt::Write as _;

use super::{
    BTreeSet, BlockId, Body, EmitError, Emitter, Local, MemoryOrdering, MemoryScope, Operand,
    Place, Statement, Target, Terminator, Ty, align_of,
};

impl<'a> Emitter<'a> {
    pub(super) fn new(body: &'a Body, target: Target, error_accum: Option<Local>) -> Self {
        Emitter {
            body,
            target,
            out: String::new(),
            tmp: 0,
            declares: BTreeSet::new(),
            uses_len: false,
            frag_regs: std::collections::HashMap::new(),
            frag_layout: super::wmma::scan_frag_layout(body),
            error_accum,
            float_binop_ordinal: 0,
            no_contract_ordinals: Vec::new(),
            strictfp_needed: false,
        }
    }

    pub(super) fn fresh(&mut self) -> String {
        let t = format!("%t{}", self.tmp);
        self.tmp += 1;
        t
    }

    // ---- type lowering -------------------------------------------------------------------------

    /// LLVM scalar type for a poot scalar `Ty`. `usize` is i32 on SPIR-V, i64 on NVPTX.
    pub(super) fn scalar_llty(&self, ty: &Ty) -> Result<String, EmitError> {
        Ok(match ty {
            Ty::Bool => "i1".into(),
            Ty::I32 | Ty::U32 => "i32".into(),
            Ty::Usize => match self.target {
                // AIE is a 32-bit core, so usize is i32 (as on SPIR-V). NVPTX is the 64-bit outlier; AMDGPU follows NVPTX
                // (the flat kernarg ABI uses i64 lengths, see `emit_signature`).
                Target::SpirvVulkan | Target::AieCore => "i32".into(),
                Target::Nvptx | Target::AmdGcn(_) => "i64".into(),
            },
            Ty::F16 => "half".into(),
            // bf16 is NVPTX-only: the LLVM SPIR-V backend rejects `bfloat` without SPV_KHR_bfloat16, which the
            // Vulkan target does not implement (spec 024 probe). Reject with a diagnostic instead of an opaque
            // llc error; the SPIR-V/wgpu path stays f32/f16.
            Ty::BF16 => match self.target {
                Target::Nvptx => "bfloat".into(),
                Target::SpirvVulkan => {
                    return Err(EmitError::Unsupported(
                        "bf16 compute is NVPTX-only (SPIR-V needs SPV_KHR_bfloat16, unsupported on the Arc)"
                            .into(),
                    ));
                }
                // The AIE-core PoC is f32-only (spec 057); bf16 on aie2p needs vector/MAC intrinsics. Reject by name
                // rather than emit IR Peano cannot lower.
                Target::AieCore => {
                    return Err(EmitError::Unsupported(
                        "bf16 is not in the AIE-core PoC subset (f32-only; vector/bf16 is rung c)"
                            .into(),
                    ));
                }
                // AMDGPU (gfx1151) supports bf16 via WMMA (card 110). The backend accepts `bfloat` on GFX11+; ocml bf16
                // arithmetic is absent, but WMMA does not need it.
                Target::AmdGcn(_) => "bfloat".into(),
            },
            Ty::F32 => "float".into(),
            Ty::F64 => "double".into(),
            // spec 134 P1: `Ty::Vec` is SpirvVulkan-only for now (the `<lanes x elem>`-typed-resource path, `spirv_buf`).
            // NVPTX/AMDGCN vector lowering (`ld.global.v4` / `global_load_dwordx4`) is a later phase; reject by name (as
            // for bf16 above) rather than let an un-lowered vector rvalue reach llc.
            Ty::Vec { elem, lanes } => match self.target {
                Target::SpirvVulkan => {
                    let inner = self.scalar_llty(elem)?;
                    format!("<{lanes} x {inner}>")
                }
                Target::Nvptx | Target::AmdGcn(_) | Target::AieCore => {
                    return Err(EmitError::Unsupported(format!(
                        "Ty::Vec{{elem={elem:?}, lanes={lanes}}} on {:?}: vector codegen is \
                         SpirvVulkan-only in this increment (spec 134 P1); NVPTX/AMDGCN lowering is a \
                         later phase",
                        self.target
                    )));
                }
            },
            other => return Err(EmitError::Unsupported(format!("scalar type {other:?}"))),
        })
    }

    pub(super) fn usize_llty(&self) -> &'static str {
        match self.target {
            Target::SpirvVulkan | Target::AieCore => "i32",
            // NVPTX and AMDGPU both use i64 for `usize`; the flat kernarg ABI declares lengths as i64 (see
            // `emit_signature`; the AMDGPU signature mirrors NVPTX).
            Target::Nvptx | Target::AmdGcn(_) => "i64",
        }
    }

    /// The `atomicrmw` syncscope prefix (with trailing space) spliced in before the ordering keyword, per target.
    /// SpirvVulkan needs an explicit `syncscope("device")`: without one, LLVM's SPIR-V backend emits `OpAtomic*`'s
    /// Memory Scope as `CrossDevice` (0), which `spirv-val --target-env vulkan1.3` rejects
    /// (VUID-StandaloneSpirv-None-04638). `syncscope("device")` emits `Device` (1), which validates and is the
    /// right scope (card 147 probe: integer `atomicrmw add` and float `atomicrmw fadd`, via `llc` + `spirv-dis`).
    /// NVPTX/AMDGCN keep the default (SYSTEM scope), which `atom.global.*` / `global_atomic_*` already lower
    /// correctly.
    pub(super) fn atomic_syncscope(&self) -> &'static str {
        match self.target {
            Target::SpirvVulkan => "syncscope(\"device\") ",
            Target::Nvptx | Target::AmdGcn(_) | Target::AieCore => "",
        }
    }

    /// The `fence` syncscope name (bare, no quotes) for `scope` on this target (spec 138 Phase 2; directed `llc`
    /// probes on LLVM 22.1.5):
    ///
    /// - NVPTX names scopes by hardware granularity: `"block"` (not `"workgroup"`, which is a fatal `LLVM ERROR:
    ///   NVPTX backend does not support syncscope "workgroup"`) lowers to `fence.acq_rel.cta`; `"device"` lowers to
    ///   `fence.acq_rel.gpu`, tighter than the unscoped default (`fence.acq_rel.sys`: host + every peer device), so
    ///   every NVPTX fence gets an explicit syncscope.
    /// - AMDGCN uses `"workgroup"` for the LDS-fence-before-barrier idiom (see the `Barrier` terminator's
    ///   `Target::AmdGcn` arm); the device-scope name is `"agent"` (card 106's ROCm HSA precedent). On a single-GPU
    ///   gfx1151 it lowers to the same `buffer_gl1_inv`+`buffer_gl0_inv` pair as the unscoped default.
    /// - SpirvVulkan: `"workgroup"` makes the backend emit `OpMemoryBarrier`'s Memory Scope as `Workgroup` (2) and
    ///   `"device"` as `Device` (1). Without a syncscope the scope operand is `OpConstantNull` = `CrossDevice` (0),
    ///   invalid on Vulkan without `VulkanMemoryModel` (the same defect and fix as `atomic_syncscope`), so no
    ///   binary-rewrite postprocess is needed. This resolves spec 138's FR-008/S1: always pass an explicit
    ///   syncscope.
    pub(super) fn fence_syncscope(&self, scope: MemoryScope) -> &'static str {
        match (self.target, scope) {
            (Target::Nvptx, MemoryScope::Workgroup) => "block",
            (Target::Nvptx, MemoryScope::Device) => "device",
            (Target::AmdGcn(_), MemoryScope::Workgroup) => "workgroup",
            (Target::AmdGcn(_), MemoryScope::Device) => "agent",
            (Target::SpirvVulkan, MemoryScope::Workgroup) => "workgroup",
            (Target::SpirvVulkan, MemoryScope::Device) => "device",
            // AieCore never reaches here - Statement::Fence rejects it before calling this.
            (Target::AieCore, _) => "singlethread",
        }
    }

    /// The alloca'd LLVM type for `l`: a matrix-fragment handle (card 530) on AMDGCN/NVPTX overrides its
    /// marker `Ty`'s scalar size with the real register-group aggregate/vector (see `frag_layout`'s doc and
    /// `wmma::frag_alloca_llty`); every other local, and every fragment on SPIR-V (tracked via `frag_regs`
    /// instead, never alloca'd), uses its declared `Ty` as before.
    pub(super) fn local_llty(&self, l: Local) -> Result<String, EmitError> {
        match (self.frag_layout.get(&l.index), self.target) {
            (Some(&(dtype, is_acc)), Target::AmdGcn(_) | Target::Nvptx) => {
                self.frag_alloca_llty(dtype, is_acc)
            }
            _ => self.scalar_llty(&self.body.locals[l.index as usize].ty),
        }
    }

    pub(super) fn is_param(&self, l: Local) -> bool {
        l.index >= 1 && l.index <= self.body.param_count
    }

    /// The element `Ty` of a slice param local (`Ref{ Slice(elem) }`).
    pub(super) fn slice_elem(&self, l: Local) -> Result<Ty, EmitError> {
        match &self.body.locals[l.index as usize].ty {
            Ty::Ref { pointee, .. } => match &**pointee {
                Ty::Slice(elem) => Ok((**elem).clone()),
                other => Err(EmitError::Unsupported(format!(
                    "param is &{other:?}, expected &[_]"
                ))),
            },
            other => Err(EmitError::Unsupported(format!(
                "param type {other:?}, expected a slice ref"
            ))),
        }
    }

    // ---- driver --------------------------------------------------------------------------------

    pub(super) fn run(&mut self) -> Result<(), EmitError> {
        // params must be slices for now (by-value scalar params are a later addition).
        for p in self.body.params() {
            self.slice_elem(p)?;
        }
        // Emit the body into a temporary so the header/footer, which depend on declares + uses_len found during
        // emission, can be written around it.
        let mut block_ir = String::new();
        // entry block: handles (spirv) + allocas + jump to bb0.
        self.emit_entry(&mut block_ir)?;
        for (i, _) in self.body.blocks.iter().enumerate() {
            self.emit_block(BlockId { index: i as u32 }, &mut block_ir)?;
        }

        self.emit_header();
        self.emit_lds_globals()?;
        self.emit_signature();
        self.out.push_str(&block_ir);
        self.out.push_str("}\n\n");
        self.emit_footer();
        Ok(())
    }

    pub(super) fn emit_header(&mut self) {
        match self.target {
            Target::SpirvVulkan => {
                self.out.push_str(
                    "target datalayout = \"e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10\"\n",
                );
                self.out
                    .push_str("target triple = \"spirv-unknown-vulkan1.3-compute\"\n\n");
                // Resource name constants (handlefrombinding needs a non-null name or llc crashes); they follow
                // datalayout/triple, matching the clang reference order.
                let slots = self.body.param_count
                    + if self.uses_len { 1 } else { 0 }
                    + if self.error_accum.is_some() { 1 } else { 0 };
                for b in 0..slots {
                    let n = b as usize;
                    let len = "buf".len() + n.to_string().len() + 1;
                    let _ = writeln!(
                        self.out,
                        "@.pn{b} = private unnamed_addr constant [{len} x i8] c\"buf{b}\\00\", align 1"
                    );
                }
                self.out.push('\n');
            }
            Target::Nvptx => {
                self.out
                    .push_str("target triple = \"nvptx64-nvidia-cuda\"\n\n");
            }
            // AIE core: no datalayout/triple in the IR text; Peano supplies them from `--march=aie2p` (plain scalar IR
            // with no triple lowers cleanly). A comment marks the target.
            Target::AieCore => {
                self.out
                    .push_str("; poot AIE-core kernel (lowered by Peano, llc --march=aie2p)\n\n");
            }
            // AMDGPU: the standard amdgcn datalayout (what clang and the ROCm LLVM fork emit for
            // `-mtriple=amdgcn-amd-amdhsa`). The `A5` and `G1` tokens set the alloca and global address spaces; without a
            // datalayout the backend rejects pointer GEPs that lack a datalayout-compliant element stride. The triple is
            // matched in `llc_args()`.
            Target::AmdGcn(_) => {
                self.out.push_str("target datalayout = \"e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9\"\n");
                self.out
                    .push_str("target triple = \"amdgcn-amd-amdhsa\"\n\n");
            }
        }
    }

    /// Module-scope workgroup-local (LDS) arrays, one `addrspace(3)` global per declaration. The LLVM SPIR-V
    /// backend lowers `addrspace(3)` to an `OpVariable ... Workgroup`; NVPTX lowers it to `.shared`. Emitted before
    /// `define` so the kernel body can `getelementptr` into them.
    pub(super) fn emit_lds_globals(&mut self) -> Result<(), EmitError> {
        if self.body.workgroup_locals.is_empty() {
            return Ok(());
        }
        if self.target == Target::AieCore {
            return Err(EmitError::Unsupported(
                "workgroup-local (LDS) memory on AIE-core: a single AIE core has no shared workgroup \
                 memory; tile staging is the IRON harness's L1/L2 ObjectFifo job, not the kernel's".into(),
            ));
        }
        // AMDGPU LLVM rejects `zeroinitializer` for addrspace(3) (LDS) globals ("unsupported initializer for address
        // space"), so use `undef` there; SPIR-V and NVPTX accept zeroinitializer.
        let init = match self.target {
            Target::AmdGcn(_) => "undef",
            _ => "zeroinitializer",
        };
        for (i, decl) in self.body.workgroup_locals.iter().enumerate() {
            let elem = self.scalar_llty(&decl.elem_ty)?;
            let _ = writeln!(
                self.out,
                "@{}_lds{i} = internal addrspace(3) global [{} x {elem}] {init}, align 16",
                self.body.name, decl.len
            );
        }
        self.out.push('\n');
        Ok(())
    }

    pub(super) fn emit_signature(&mut self) {
        match self.target {
            Target::SpirvVulkan => {
                self.out.push_str("define void @main() #0 {\n");
            }
            Target::Nvptx => {
                let mut args = Vec::new();
                for p in self.body.params() {
                    args.push(format!("ptr addrspace(1) %p{}, i64 %n{}", p.index, p.index));
                }
                let attr = self.strictfp_attr_suffix();
                let _ = writeln!(
                    self.out,
                    "define ptx_kernel void @{}({}){attr} {{",
                    self.body.name,
                    args.join(", ")
                );
            }
            // AIE core: a plain `extern "C"`-shape function the IRON harness links by symbol name, with flat `ptr` args
            // (default addrspace), i32 lengths (32-bit core), and no `ptx_kernel` cc. IRON's `external_func` binds the
            // ObjectFifo-acquired buffers to it (bare-pointer ABI).
            Target::AieCore => {
                let mut args = Vec::new();
                for p in self.body.params() {
                    args.push(format!("ptr %p{}, i32 %n{}", p.index, p.index));
                }
                let _ = writeln!(
                    self.out,
                    "define void @{}({}) {{",
                    self.body.name,
                    args.join(", ")
                );
            }
            // AMDGPU M1 deliberately mirrors NVPTX: the kernel takes `(ptr addrspace(1), i64)` per slice. AMDGPU's
            // idiomatic ABI is a single flat kernarg buffer pointer (`@llvm.amdgcn.kernarg.segment.ptr`) with each scalar
            // at a fixed byte offset; that is needed once the per-op executor mixes pointers, scalars, and i32 shape
            // metadata (M2). M1 handles only the elementwise add case (3 slices of `ptr addrspace(1)` + `i64` length).
            // The `amdgpu_kernel` calling conv marks the symbol as an HSA kernel (`STT_AMDGPU_HSA_KERNEL` after
            // lowering).
            Target::AmdGcn(_) => {
                let mut args = Vec::new();
                for p in self.body.params() {
                    args.push(format!("ptr addrspace(1) %p{}, i64 %n{}", p.index, p.index));
                }
                let attr = self.strictfp_attr_suffix();
                let _ = writeln!(
                    self.out,
                    "define amdgpu_kernel void @{}({}){attr} {{",
                    self.body.name,
                    args.join(", ")
                );
            }
        }
    }

    /// The ` #0` suffix `emit_signature` and `emit_binop_no_contract` append to a Nvptx/AmdGcn `define`
    /// or constrained-intrinsic call when the body needs `strictfp` (card 628); empty otherwise, and empty
    /// on every other target (SpirvVulkan's `#0`/`#1` are unrelated attribute groups; AieCore never emits
    /// `BinaryOpNoContract`).
    pub(super) fn strictfp_attr_suffix(&self) -> &'static str {
        if self.strictfp_needed { " #0" } else { "" }
    }

    pub(super) fn emit_entry(&mut self, w: &mut String) -> Result<(), EmitError> {
        w.push_str("entry:\n");
        // SPIR-V: bind a resource handle for each slice param (and the length buffer if used).
        if self.target == Target::SpirvVulkan {
            for p in self.body.params() {
                let elem = self.param_bind_ty(p)?;
                let (bufty, mangle) = self.spirv_buf(&elem, true)?;
                let _ = writeln!(
                    w,
                    "  %h{} = tail call {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32 0, i32 {bind}, i32 1, i32 0, ptr nonnull @.pn{bind})",
                    p.index,
                    bind = p.index - 1
                );
                self.declares.insert(format!(
                    "declare {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32, i32, i32, i32, ptr)"
                ));
                self.declares.insert(format!(
                    "declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty}, i32)"
                ));
            }
            // Length handle: either `Rvalue::Len` slots or the X-thread extent for folded
            // `thread_index(X)` reconstruction (see `body_needs_length_buffer`).
            if self.body_needs_length_buffer() {
                self.uses_len = true;
                let bind = self.body.param_count;
                let (bufty, mangle) = self.spirv_len_buf();
                let _ = writeln!(
                    w,
                    "  %hlen = tail call {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32 0, i32 {bind}, i32 1, i32 0, ptr nonnull @.pn{bind})"
                );
                self.declares.insert(format!(
                    "declare {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32, i32, i32, i32, ptr)"
                ));
                self.declares.insert(format!(
                    "declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty}, i32)"
                ));
            }
            // Error-word handle: one writable u32 slot, bound only when `debranch` gave this body a fault
            // accumulator (the pre-debranch body had a Trap: a failed Assert or a reached Unreachable),
            // right after the params and the length buffer if either used a slot (card 531c).
            // `Terminator::Return`'s SpirvVulkan lowering flushes the accumulator through `%herr`.
            if self.error_accum.is_some() {
                let bind = self.body.param_count
                    + if self.body_needs_length_buffer() {
                        1
                    } else {
                        0
                    };
                let (bufty, mangle) = self.spirv_buf(&Ty::U32, true)?;
                let _ = writeln!(
                    w,
                    "  %herr = tail call {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32 0, i32 {bind}, i32 1, i32 0, ptr nonnull @.pn{bind})"
                );
                self.declares.insert(format!(
                    "declare {bufty} @llvm.spv.resource.handlefrombinding.{mangle}(i32, i32, i32, i32, ptr)"
                ));
                self.declares.insert(format!(
                    "declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty}, i32)"
                ));
            }
        } else if self.body_uses_len() {
            self.uses_len = true; // nvptx: len is a kernel arg, nothing to bind here.
        }
        // Allocas for every non-param scalar local (skip _0 unit return, slice params, and reference/slice-typed
        // locals; an imported body's reborrow-alias temporaries are folded to their source param by normalize and
        // left unused here).
        for (idx, decl) in self.body.locals.iter().enumerate() {
            let l = Local { index: idx as u32 };
            if l.index == 0
                || self.is_param(l)
                || decl.ty == Ty::Unit
                || matches!(decl.ty, Ty::Ref { .. } | Ty::Slice(_))
            {
                continue;
            }
            // A fixed-size private array (flash attention's o[D] accumulator) is `alloca [N x elem]`. The LLVM SPIR-V
            // backend crashes on a dynamically-indexed private array (poot-legacy spec-076), so such a kernel is
            // NVPTX-only; reject it on SpirvVulkan with a named diagnostic.
            if let Ty::Array { elem, len } = &decl.ty {
                if matches!(self.target, Target::SpirvVulkan) {
                    return Err(EmitError::Unsupported(format!(
                        "private array local _{idx} ([{len} x _]): the LLVM SPIR-V backend crashes on a \
                         dynamically-indexed private array (poot-legacy spec-076); this kernel is NVPTX-only"
                    )));
                }
                let elem_llty = self.scalar_llty(elem)?;
                if matches!(self.target, Target::AmdGcn(_)) {
                    // AMDGPU IR requires allocas in addrspace(5) (private); the verifier rejects unadorned `alloca`s during
                    // `llvm-as`, so the alloca text needs the explicit space.
                    let _ = writeln!(w, "  %l{idx} = alloca [{len} x {elem_llty}], addrspace(5)");
                } else {
                    let _ = writeln!(w, "  %l{idx} = alloca [{len} x {elem_llty}]");
                }
                continue;
            }
            // A matrix-fragment handle (card 530) gets its real register-group aggregate/vector alloca type
            // here too, via `local_llty` (see its doc) - the same type `load_place`/`store_place` use.
            let llty = self.local_llty(l)?;
            if matches!(self.target, Target::AmdGcn(_)) {
                let _ = writeln!(w, "  %l{} = alloca {llty}, addrspace(5)", idx);
            } else {
                let _ = writeln!(w, "  %l{} = alloca {llty}", idx);
            }
        }
        w.push_str("  br label %bb0\n");
        Ok(())
    }

    pub(super) fn emit_block(&mut self, id: BlockId, w: &mut String) -> Result<(), EmitError> {
        let block = &self.body.blocks[id.index as usize];
        let _ = writeln!(w, "bb{}:", id.index);
        // clone the small bits we need to avoid borrow conflicts with &mut self.
        let stmts = block.statements.clone();
        let term = block.terminator.clone();
        for s in &stmts {
            self.emit_statement(s, w)?;
        }
        self.emit_terminator(&term, w)?;
        Ok(())
    }

    pub(super) fn emit_statement(
        &mut self,
        s: &Statement,
        w: &mut String,
    ) -> Result<(), EmitError> {
        match s {
            Statement::Assign(place, rvalue) => {
                // Most rvalues ignore this; `VectorSplat`/`VectorLoad` read lanes/elem off the assignment target's declared
                // `Ty::Vec` (spec 134 P1), since a splat of a bare scalar operand has no other source.
                let dest_ty = self.body.locals[place.local.index as usize].ty.clone();
                let (val, llty) = self.emit_rvalue(rvalue, &dest_ty, w)?;
                self.store_place(place, &val, &llty, w)?;
                Ok(())
            }
            Statement::StorageLive(_) | Statement::StorageDead(_) => Ok(()), // no-op (allocas live whole fn)
            // spec 134 P1 (FR-001, write side): store a `Ty::Vec` operand as `lanes` contiguous elements at `place`'s
            // vector-group index; `vector_element_ptr` picks the vec4-typed resource declared for this param by
            // `param_bind_ty`.
            Statement::VectorStore { place, value } => {
                let vec_ty = self.operand_ty(value)?;
                let (v, llty) = self.operand(value, w)?;
                let (ptr, elem_llty, addrspace) = self.vector_element_ptr(place, &vec_ty, w)?;
                debug_assert_eq!(
                    llty, elem_llty,
                    "VectorStore operand llty must match the vector buffer element llty"
                );
                let align = align_of(&elem_llty);
                let _ = writeln!(
                    w,
                    "  store {elem_llty} {v}, ptr addrspace({addrspace}) {ptr}, align {align}"
                );
                Ok(())
            }
            Statement::WorkgroupLocalWrite { idx, value, array } => {
                let (i, _) = self.operand(idx, w)?;
                let (v, _) = self.operand(value, w)?;
                let decl = &self.body.workgroup_locals[*array as usize];
                let (n, elem) = (decl.len, self.scalar_llty(&decl.elem_ty)?);
                let us = self.usize_llty();
                let name = self.body.name.clone();
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr [{n} x {elem}], ptr addrspace(3) @{name}_lds{array}, {us} 0, {us} {i}"
                );
                // card 045: a hardcoded `align 4` on 2-byte-elem (bf16) storage is UB and made the AMDGPU backend miscompile
                // bf16 scalar accesses to 0. The atomic LDS paths already use align_of(); this plain write/read pair was the
                // last place hardcoding it.
                let align = align_of(&elem);
                let _ = writeln!(w, "  store {elem} {v}, ptr addrspace(3) {p}, align {align}");
                Ok(())
            }
            Statement::WmmaLoad {
                which,
                dtype,
                shape,
                tile,
                stride,
                dst,
            } => self.emit_wmma_load(*which, *dtype, *shape, tile, *stride, *dst, w),
            Statement::WmmaMma {
                dtype,
                shape,
                a,
                b,
                c,
                dst,
            } => self.emit_wmma_mma(*dtype, *shape, *a, *b, *c, *dst, w),
            Statement::WmmaStore {
                dtype,
                shape,
                tile,
                stride,
                src,
            } => self.emit_wmma_store(*dtype, *shape, tile, *stride, *src, w),
            Statement::WmmaZero { dtype, shape, dst } => {
                self.emit_wmma_zero(*dtype, *shape, *dst, w)
            }
            Statement::WmmaStoreLds {
                shape,
                array,
                stride,
                src,
            } => self.emit_wmma_store_lds(*shape, *array, *stride, *src, w),
            Statement::WmmaLoadLds {
                which,
                dtype,
                shape,
                array,
                stride,
                dst,
            } => self.emit_wmma_load_lds(*which, *dtype, *shape, *array, *stride, *dst, w),
            // Spec 138 Phase 2: a scoped memory-ordering fence with no control-flow effect. AieCore is rejected here (in
            // `emit_statement`, not `emit_terminator`): a single AIE core has no other core to fence against (as for the
            // Barrier/CAS rejections). Every other target gets an explicit syncscope (see
            // `fence_syncscope`); an explicit `syncscope("device")` already emits a valid `Device` scope, so no SPIR-V
            // postprocess rewrite is needed (spec 138 FR-008, as for `atomic_syncscope`).
            Statement::Fence { scope, ordering } => {
                if self.target == Target::AieCore {
                    return Err(EmitError::Unsupported(
                        "memory fence on AIE-core: a single AIE core has no other core to fence \
                         against (the kernel must be a sequential loop; data movement is the IRON \
                         harness's job)"
                            .into(),
                    ));
                }
                let scope_kw = self.fence_syncscope(*scope);
                let ord_kw = match ordering {
                    MemoryOrdering::Acquire => "acquire",
                    MemoryOrdering::Release => "release",
                    MemoryOrdering::AcqRel => "acq_rel",
                };
                let _ = writeln!(w, "  fence syncscope(\"{scope_kw}\") {ord_kw}");
                Ok(())
            }
        }
    }

    pub(super) fn emit_terminator(
        &mut self,
        t: &Terminator,
        w: &mut String,
    ) -> Result<(), EmitError> {
        match t {
            Terminator::Goto { target } => {
                let _ = writeln!(w, "  br label %bb{}", target.index);
                Ok(())
            }
            Terminator::Return => {
                // SpirvVulkan, when `debranch` gave this body a fault accumulator (card 531c): flush its
                // current value into the reserved error word before returning. Every `Return` needs this,
                // not just ones `debranch` itself introduced for an `Unreachable` site - an `Assert`'s
                // failure bit must reach the word even on the kernel's own, pre-existing early exits.
                if let Some(accum) = self.error_accum {
                    let (val, _) = self.operand(&Operand::Copy(Place::local(accum)), w)?;
                    let (bufty, mangle) = self.spirv_buf(&Ty::U32, true)?;
                    let p = self.fresh();
                    let _ = writeln!(
                        w,
                        "  {p} = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty} %herr, i32 0)"
                    );
                    self.declares.insert(format!(
                        "declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.{mangle}({bufty}, i32)"
                    ));
                    let old = self.fresh();
                    let scope = self.atomic_syncscope();
                    let _ = writeln!(
                        w,
                        "  {old} = atomicrmw or ptr addrspace(11) {p}, i32 {val} {scope}monotonic, align 4"
                    );
                }
                w.push_str("  ret void\n");
                Ok(())
            }
            Terminator::SwitchInt { discr, targets } => {
                let (val, ty) = self.operand(discr, w)?;
                // 2-way bool guard shape (an i1 discr, value 0 -> branch, otherwise -> the other): a conditional branch, as
                // `guard()` emits and the corpus relies on.
                if ty == "i1" && targets.branches.len() == 1 && targets.branches[0].0 == 0 {
                    let els = targets.branches[0].1.index;
                    let then = targets.otherwise.index;
                    let _ = writeln!(w, "  br i1 {val}, label %bb{then}, label %bb{els}");
                    Ok(())
                } else {
                    // General multi-way switch on an integer discriminant (e.g. the M2 scheduler's per-task op dispatch): one
                    // `switch` instruction, `otherwise` is the default. NVPTX handles this directly; SPIR-V would need
                    // structurization, so multi-way switch kernels are NVPTX-only for now.
                    let _ = write!(
                        w,
                        "  switch {ty} {val}, label %bb{} [",
                        targets.otherwise.index
                    );
                    for (v, target) in &targets.branches {
                        let _ = write!(w, " {ty} {v}, label %bb{}", target.index);
                    }
                    let _ = writeln!(w, " ]");
                    Ok(())
                }
            }
            Terminator::ThreadIndexCall {
                destination,
                dim,
                target,
            } => {
                let v = self.emit_thread_index(*dim, w)?;
                // store into the destination local (usize-width).
                let llty = self.usize_llty().to_string();
                self.store_place(destination, &v, &llty, w)?;
                let _ = writeln!(w, "  br label %bb{}", target.index);
                Ok(())
            }
            Terminator::Barrier { target } => {
                // Workgroup-scope control + memory barrier. SPIR-V: OpControlBarrier via the spv intrinsic; NVPTX: bar.sync 0
                // via the nvvm intrinsic.
                match self.target {
                    Target::SpirvVulkan => {
                        // The barrier must be `convergent` (attr group #1): otherwise LLVM's SPIR-V backend mis-places it relative to
                        // loop merge blocks, so a barrier inside a loop (e.g. the per-k-chunk barrier of a tiled GEMM) hangs or faults
                        // the GPU. Not a `tail` call: a barrier is never in tail position (a branch follows) and `tail` confuses the
                        // backend.
                        self.declares.insert(
                            "declare void @llvm.spv.group.memory.barrier.with.group.sync() #1"
                                .into(),
                        );
                        let _ = writeln!(
                            w,
                            "  call void @llvm.spv.group.memory.barrier.with.group.sync() #1"
                        );
                    }
                    Target::Nvptx => {
                        self.declares
                            .insert("declare void @llvm.nvvm.barrier0()".into());
                        let _ = writeln!(w, "  call void @llvm.nvvm.barrier0()");
                    }
                    // A single AIE core has no workgroup to synchronize; a barrier is a GPU/SPMD construct. Reject by name:
                    // an AIE kernel must be a barrier-free sequential loop.
                    Target::AieCore => {
                        return Err(EmitError::Unsupported(
                            "workgroup Barrier on AIE-core: a single AIE core has no workgroup to \
                             synchronize (the kernel must be a sequential loop; data movement is the \
                             IRON harness's job)".into(),
                        ));
                    }
                    // AMDGPU: `@llvm.amdgcn.s.barrier` lowers to `s_barrier`, which synchronizes wave execution but does not by
                    // itself guarantee cross-wave LDS memory visibility. Without an `s_waitcnt lgkmcnt(0)` before the barrier, one
                    // wave's LDS stores may not be visible to another wave after it: a data race when more than one wave does
                    // cross-wave LDS communication (e.g. GEMV_WIDTH=128 = 4 wave32s on gfx1151; the Q6_K dequant_gemv_lds LDS-tree
                    // combine was nondeterministic, cards 171/174). A `fence syncscope("workgroup") release` before the barrier
                    // makes the AMDGPU backend insert that waitcnt, so every SPMD wave has completed its LDS writes once any wave
                    // passes the shared barrier. A matching `acquire` fence after the barrier is not needed on RDNA: LDS is direct
                    // SRAM (not per-wave cached), so a post-barrier `ds_read` sees the fresh writes, and the barrier is a
                    // scheduling fence so reads are not hoisted above it. The `acquire` half was also where nearly all the cost
                    // lived (card 176: release+acquire ~2x ROCm decode slowdown; release-only is full speed and still
                    // deterministic on the q6k race repro and the capture/tiled-GEMM LDS fuzzers).
                    //
                    // Scoped (card 176): the release waitcnt is pointless with no cross-wave LDS to protect, so emit it only when
                    // the kernel uses LDS (`workgroup_locals`) and the workgroup spans more than one wave (>32 threads on
                    // wave32). Otherwise a bare barrier is correct and fastest.
                    Target::AmdGcn(_) => {
                        self.declares
                            .insert("declare void @llvm.amdgcn.s.barrier()".into());
                        let threads: u32 = self.body.workgroup_size.iter().product();
                        let needs_lds_fence =
                            !self.body.workgroup_locals.is_empty() && threads > 32;
                        if needs_lds_fence {
                            let _ = writeln!(w, "  fence syncscope(\"workgroup\") release");
                            let _ = writeln!(w, "  call void @llvm.amdgcn.s.barrier()");
                        } else {
                            let _ = writeln!(w, "  call void @llvm.amdgcn.s.barrier()");
                        }
                    }
                }
                let _ = writeln!(w, "  br label %bb{}", target.index);
                Ok(())
            }
            // A kernel-subset fault (a failed Assert or a reached Unreachable): abort instead of letting
            // the thread continue with whatever it had computed so far (card 531c, R468-007). ROCm and PTX
            // have a real device trap instruction (`llvm.trap`, generic across both backends). SpirvVulkan
            // never reaches this arm at all: `debranch::debranch_traps_for_spirv` runs before
            // structurization and rewrites every `Trap` into branchless fault accumulation (Vulkan compute
            // has no trap instruction), so a `Trap` surviving to emission on that target is an internal
            // error, not a legitimate shape to lower.
            Terminator::Trap { code: _ } => match self.target {
                Target::AmdGcn(_) | Target::Nvptx => {
                    self.declares.insert("declare void @llvm.trap()".into());
                    let _ = writeln!(w, "  call void @llvm.trap()");
                    w.push_str("  unreachable\n");
                    Ok(())
                }
                Target::SpirvVulkan => Err(EmitError::Unsupported(
                    "internal error: a Trap terminator reached SpirvVulkan emission; debranch_traps_for_spirv \
                     should have removed every one (card 531c)"
                        .into(),
                )),
                Target::AieCore => Err(EmitError::Unsupported(
                    "Assert/Unreachable trap on AIE-core: not in the AIE-core PoC subset".into(),
                )),
            },
        }
    }

    pub(super) fn finish(self) -> String {
        self.out
    }
}
