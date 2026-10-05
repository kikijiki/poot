use std::fmt::Write as _;

use super::{
    AtomicOp, BinOp, EmitError, Emitter, Fp8Format, NO_CONTRACT_STRICTFP_ATTR, Operand, ResultTy,
    Rvalue, Target, Ty, align_of, binop_inst, float_bits, int_bits, positive_e4m3fn_value,
};

impl<'a> Emitter<'a> {
    /// Emit an rvalue, returning (value register/literal, its llvm type). `dest_ty` is the declared type of the
    /// `Assign`'s LHS local; most rvalues ignore it, but `VectorLoad`/`VectorSplat` (spec 134 P1) read `elem`/`lanes`
    /// from it, since a bare scalar splat source or a scalar-slice load source carries no lane count.
    pub(super) fn emit_rvalue(
        &mut self,
        rv: &Rvalue,
        dest_ty: &Ty,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        match rv {
            Rvalue::Use(op) => self.operand(op, w),
            Rvalue::BinaryOp(op, a, b) => self.emit_binop(*op, a, b, w),
            Rvalue::BinaryOpNoContract(op, a, b) => self.emit_binop_no_contract(*op, a, b, w),
            Rvalue::UnaryOp(op, a) => self.emit_unary(*op, a, w),
            Rvalue::MathUnary(op, a) => self.emit_mathunary(*op, a, w),
            Rvalue::IntScalarUnary(op, a) => self.emit_int_scalar_unary(*op, a, w),
            Rvalue::Len(place) => self.emit_len(place.local, w),
            Rvalue::Cast { to, operand } => self.emit_cast(to, operand, w),
            Rvalue::Bitcast { to, operand } => self.emit_bitcast(to, operand, w),
            Rvalue::Fp8Decode { format, operand } => {
                self.emit_fp8_decode(*format, operand, dest_ty, w)
            }
            Rvalue::Fp8Encode { format, operand } => {
                self.emit_fp8_encode(*format, operand, dest_ty, w)
            }
            Rvalue::WorkgroupLocalRead { idx, array } => {
                let (i, _) = self.operand(idx, w)?;
                let decl = &self.body.workgroup_locals[*array as usize];
                let (n, elem) = (decl.len, self.scalar_llty(&decl.elem_ty)?);
                let us = self.usize_llty();
                let name = self.body.name.clone();
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr [{n} x {elem}], ptr addrspace(3) @{name}_lds{array}, {us} 0, {us} {i}"
                );
                let t = self.fresh();
                let align = align_of(&elem);
                let _ = writeln!(
                    w,
                    "  {t} = load {elem}, ptr addrspace(3) {p}, align {align}"
                );
                Ok((t, elem))
            }
            // Atomic RMW on a workgroup-local (LDS, addrspace 3) array element, returning the old value: the
            // intra-workgroup counterpart of GlobalAtomic, for in-tile reductions and histograms. One workgroup is
            // co-resident, so there is no forward-progress concern and it verifies on RADV directly. Same atomicrmw
            // lowering as GlobalAtomic (including the `syncscope` fix; see `atomic_syncscope`).
            Rvalue::WorkgroupLocalAtomic {
                idx,
                value,
                op,
                array,
            } => {
                let (i, _) = self.operand(idx, w)?;
                let (v, _) = self.operand(value, w)?;
                let decl = &self.body.workgroup_locals[*array as usize];
                let (n, elem) = (decl.len, self.scalar_llty(&decl.elem_ty)?);
                // Matching only "float" (f32) would emit an integer atomicrmw mnemonic (`add`, not `fadd`) on a float-typed
                // pointer for f16/f64/bf16, which the LLVM verifier rejects with an opaque error instead of this crate's named
                // Unsupported diagnostic (no producer emits these today).
                let is_float = matches!(elem.as_str(), "float" | "double" | "half" | "bfloat");
                let inst = match (op, is_float) {
                    (AtomicOp::Add, true) => "fadd",
                    (AtomicOp::Add, false) => "add",
                    (AtomicOp::Min, true) => "fmin",
                    (AtomicOp::Min, false) => "min",
                    (AtomicOp::Max, true) => "fmax",
                    (AtomicOp::Max, false) => "max",
                    (AtomicOp::And, false) => "and",
                    (AtomicOp::Or, false) => "or",
                    (AtomicOp::Xor, false) => "xor",
                    (AtomicOp::Exchange, _) => "xchg",
                    (other_op, _) => {
                        return Err(EmitError::Unsupported(format!(
                            "workgroup atomic {other_op:?} on {elem}"
                        )));
                    }
                };
                let us = self.usize_llty();
                let name = self.body.name.clone();
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr [{n} x {elem}], ptr addrspace(3) @{name}_lds{array}, {us} 0, {us} {i}"
                );
                let t = self.fresh();
                let align = align_of(&elem);
                let scope = self.atomic_syncscope();
                let _ = writeln!(
                    w,
                    "  {t} = atomicrmw {inst} ptr addrspace(3) {p}, {elem} {v} {scope}monotonic, align {align}"
                );
                Ok((t, elem))
            }
            // Atomic read-modify-write on a global slice element (`slice[idx] op= value`), returning the old value. It is
            // an event-counter primitive: a producer atomically bumps a dependency counter and a consumer spin-waits on it. Ordering is monotonic (relaxed):
            // correct for a counter read after a barrier; the cross-CTA acquire/release spin-wait is the NVPTX
            // persistent-grid concern (M2). atomicrmw lowers to OpAtomic* on SPIR-V (float add via
            // SPV_EXT_shader_atomic_float_add) and atom.global.* on NVPTX.
            //
            // Card 147 probe: without an explicit `syncscope`, LLVM's SPIR-V backend emits `OpAtomic*`'s Memory Scope as
            // `CrossDevice` (0), which `spirv-val --target-env vulkan1.3` rejects (VUID-StandaloneSpirv-None-04638; Vulkan
            // shaders lack the capability). RADV tolerated it, and the CAS work (`GlobalCompareExchange`) surfaced it. So on
            // SpirvVulkan emit `syncscope("device")`, which selects `Device` (1) scope and validates; it is also the right
            // scope, since contending threads span workgroups. Probed with a hand-written `.ll` through
            // `llc --spirv-ext=+SPV_EXT_shader_atomic_float_add` and `spirv-dis`: both integer `atomicrmw add` and float
            // `atomicrmw fadd` lower to `OpAtomicIAdd`/`OpAtomicFAddEXT` with Scope `Device`. NVPTX/AMDGCN are unchanged
            // (no explicit syncscope; `atom.global.*`/`global_atomic_*` are the correct SYSTEM-scope lowering). See
            // `atomic_syncscope`.
            Rvalue::GlobalAtomic { place, value, op } => {
                let (ptr, elem_llty, addrspace) = self.element_ptr(place, w)?;
                let (v, _) = self.operand(value, w)?;
                let is_float = matches!(elem_llty.as_str(), "float" | "double" | "half" | "bfloat");
                let inst = match (op, is_float) {
                    (AtomicOp::Add, true) => "fadd",
                    (AtomicOp::Add, false) => "add",
                    (AtomicOp::Min, true) => "fmin",
                    (AtomicOp::Min, false) => "min",
                    (AtomicOp::Max, true) => "fmax",
                    (AtomicOp::Max, false) => "max",
                    (AtomicOp::And, false) => "and",
                    (AtomicOp::Or, false) => "or",
                    (AtomicOp::Xor, false) => "xor",
                    (AtomicOp::Exchange, _) => "xchg",
                    (other_op, _) => {
                        return Err(EmitError::Unsupported(format!(
                            "atomic {other_op:?} on {elem_llty}"
                        )));
                    }
                };
                let t = self.fresh();
                let align = align_of(&elem_llty);
                let scope = self.atomic_syncscope();
                let _ = writeln!(
                    w,
                    "  {t} = atomicrmw {inst} ptr addrspace({addrspace}) {ptr}, {elem_llty} {v} {scope}monotonic, align {align}"
                );
                Ok((t, elem_llty))
            }
            // Compare-and-swap on a workgroup-local (LDS) array element, returning the old value (spec 138).
            // Integer-only: LLVM `cmpxchg` rejects float operands and Vulkan has no float `OpAtomicCompareExchange`, so
            // this type domain is narrower than the RMW atomics (which support float add/min/max); a float element gets a
            // named diagnostic. AieCore is rejected here (in `emit_rvalue`, not `emit_terminator`): a single AIE core has
            // no other CTA to CAS against, as for Barrier.
            //
            // SpirvVulkan is also rejected: LLVM 22's SPIR-V backend crashes lowering generic `cmpxchg`
            // (`SPIRVLegalizePointerCast::legalizePointerCast` hits `llvm_unreachable` on the auto-generated
            // `llvm.spv.cmpxchg` call, which is not in its handled-user list). A directed llc probe (spec 138 Phase 0)
            // confirmed this on three address spaces (a Vulkan-buffer resource pointer, an LDS global, a plain `alloca`),
            // with and without an explicit `syncscope`; a hand-emitted `llvm.spv.cmpxchg` call yields spirv-val-invalid
            // output (the backend inserts an `OpCompositeInsert` into a scalar `OpUndef` instead of an aggregate one).
            // Unlike the RMW atomics, whose CrossDevice scope is invalid but still produces a module, there is no module
            // to postprocess. NVPTX/AMDGCN lower `cmpxchg` cleanly with the default scope
            // (`atom.relaxed.sys.global.cas.b32` / `global_atomic_cmpswap_b32`), so no per-target syncscope is needed.
            Rvalue::WorkgroupLocalCompareExchange {
                idx,
                expected,
                desired,
                array,
            } => {
                if self.target == Target::AieCore {
                    return Err(EmitError::Unsupported(
                        "workgroup compare-exchange on AIE-core: a single AIE core has no other CTA to \
                         CAS against (the kernel must be a sequential loop; data movement is the IRON \
                         harness's job)"
                            .into(),
                    ));
                }
                if self.target == Target::SpirvVulkan {
                    return Err(EmitError::Unsupported(
                        "compare-exchange on SpirvVulkan: LLVM 22's SPIR-V backend crashes lowering \
                         `cmpxchg` (SPIRVLegalizePointerCast has no handler for the auto-generated \
                         `llvm.spv.cmpxchg` intrinsic's ptrcast use) - no valid lowering exists on this \
                         toolchain (spec 138 Phase 0 probe); CAS is NVPTX/AMDGCN-only for now"
                            .into(),
                    ));
                }
                let decl = &self.body.workgroup_locals[*array as usize];
                let (n, elem) = (decl.len, decl.elem_ty.clone());
                if elem.is_float() {
                    return Err(EmitError::Unsupported(format!(
                        "workgroup compare-exchange on {elem:?}: CAS is integer-only (LLVM `cmpxchg` \
                         rejects float; Vulkan has no float OpAtomicCompareExchange)"
                    )));
                }
                let elem_llty = self.scalar_llty(&elem)?;
                let (i, _) = self.operand(idx, w)?;
                let (e, _) = self.operand(expected, w)?;
                let (d, _) = self.operand(desired, w)?;
                let us = self.usize_llty();
                let name = self.body.name.clone();
                let p = self.fresh();
                let _ = writeln!(
                    w,
                    "  {p} = getelementptr [{n} x {elem_llty}], ptr addrspace(3) @{name}_lds{array}, {us} 0, {us} {i}"
                );
                let agg = self.fresh();
                let align = align_of(&elem_llty);
                let _ = writeln!(
                    w,
                    "  {agg} = cmpxchg ptr addrspace(3) {p}, {elem_llty} {e}, {elem_llty} {d} monotonic monotonic, align {align}"
                );
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = extractvalue {{ {elem_llty}, i1 }} {agg}, 0");
                Ok((t, elem_llty))
            }
            // Compare-and-swap on a global slice element, returning the old value (spec 138): the
            // worker-queue / task-tail counter primitive, where `Exchange` cannot express
            // read-compare-conditional-write. Same integer-only, per-target gating as the workgroup-local sibling above
            // (see its comment for the SpirvVulkan crash).
            Rvalue::GlobalCompareExchange {
                place,
                expected,
                desired,
            } => {
                if self.target == Target::AieCore {
                    return Err(EmitError::Unsupported(
                        "compare-exchange on AIE-core: a single AIE core has no other CTA to CAS against \
                         (the kernel must be a sequential loop; data movement is the IRON harness's job)"
                            .into(),
                    ));
                }
                if self.target == Target::SpirvVulkan {
                    return Err(EmitError::Unsupported(
                        "compare-exchange on SpirvVulkan: LLVM 22's SPIR-V backend crashes lowering \
                         `cmpxchg` (SPIRVLegalizePointerCast has no handler for the auto-generated \
                         `llvm.spv.cmpxchg` intrinsic's ptrcast use) - no valid lowering exists on this \
                         toolchain (spec 138 Phase 0 probe); CAS is NVPTX/AMDGCN-only for now"
                            .into(),
                    ));
                }
                let elem = self.slice_elem(place.local)?;
                if elem.is_float() {
                    return Err(EmitError::Unsupported(format!(
                        "compare-exchange on {elem:?}: CAS is integer-only (LLVM `cmpxchg` rejects float; \
                         Vulkan has no float OpAtomicCompareExchange)"
                    )));
                }
                let (ptr, elem_llty, addrspace) = self.element_ptr(place, w)?;
                let (e, _) = self.operand(expected, w)?;
                let (d, _) = self.operand(desired, w)?;
                let agg = self.fresh();
                let align = align_of(&elem_llty);
                let _ = writeln!(
                    w,
                    "  {agg} = cmpxchg ptr addrspace({addrspace}) {ptr}, {elem_llty} {e}, {elem_llty} {d} monotonic monotonic, align {align}"
                );
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = extractvalue {{ {elem_llty}, i1 }} {agg}, 0");
                Ok((t, elem_llty))
            }
            // spec 134 P1 (FR-001, read side): load `lanes` contiguous elements of a scalar-element slice as one
            // `Ty::Vec` at `place`'s vector-group index. `dest_ty` (the Assign's LHS local) supplies `elem`/`lanes`;
            // `vector_element_ptr` declares the vec4-typed resource (FR-004) instead of bitcasting a scalar one.
            Rvalue::VectorLoad { place } => {
                if !matches!(dest_ty, Ty::Vec { .. }) {
                    return Err(EmitError::Unsupported(
                        "VectorLoad requires a Ty::Vec destination local".into(),
                    ));
                }
                let (ptr, elem_llty, addrspace) = self.vector_element_ptr(place, dest_ty, w)?;
                let v = self.fresh();
                let align = align_of(&elem_llty);
                let _ = writeln!(
                    w,
                    "  {v} = load {elem_llty}, ptr addrspace({addrspace}) {ptr}, align {align}"
                );
                Ok((v, elem_llty))
            }
            // Broadcast: insert the scalar into lane 0 of a poison vector, then `shufflevector` with an all-zero mask to
            // splat it (the same insertelement/extractelement style the WMMA fragment-build code uses).
            Rvalue::VectorSplat(op) => {
                let Ty::Vec { lanes, .. } = dest_ty else {
                    return Err(EmitError::Unsupported(
                        "VectorSplat requires a Ty::Vec destination local".into(),
                    ));
                };
                if self.target != Target::SpirvVulkan {
                    return Err(EmitError::Unsupported(format!(
                        "VectorSplat on {:?}: vector codegen is SpirvVulkan-only in this increment \
                         (spec 134 P1)",
                        self.target
                    )));
                }
                // The operand's llty (`sllty`) already reflects `dest_ty`'s scalar elem type.
                let (v, sllty) = self.operand(op, w)?;
                let vecty = format!("<{lanes} x {sllty}>");
                let tmp = self.fresh();
                let _ = writeln!(
                    w,
                    "  {tmp} = insertelement {vecty} poison, {sllty} {v}, i32 0"
                );
                let mask = format!("<{lanes} x i32> zeroinitializer");
                let out = self.fresh();
                let _ = writeln!(
                    w,
                    "  {out} = shufflevector {vecty} {tmp}, {vecty} poison, {mask}"
                );
                Ok((out, vecty))
            }
        }
    }

    /// Value-converting numeric cast: int<->float and int width changes (the LLVM conv ops).
    pub(super) fn emit_cast(
        &mut self,
        to: &Ty,
        operand: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let (v, from_llty) = self.operand(operand, w)?;
        let from = self.operand_ty(operand)?;
        let to_llty = self.scalar_llty(to)?;
        if from_llty == to_llty {
            return Ok((v, to_llty)); // no-op (e.g. usize->u32 on a backend where both are i32)
        }
        let from_float = from.is_float();
        let to_float = to.is_float();
        let from_signed = from.is_signed_int();
        let to_signed = to.is_signed_int();
        let op = match (from_float, to_float) {
            (true, false) => {
                if to_signed {
                    "fptosi"
                } else {
                    "fptoui"
                }
            }
            (false, true) => {
                if from_signed {
                    "sitofp"
                } else {
                    "uitofp"
                }
            }
            (true, true) => {
                // float<->float: widen (fpext) or narrow (fptrunc) by bit width; covers bf16/f16 (16) <-> f32 (32) <-> f64
                // (64), i.e. the bf16 load (widen) and store (narrow).
                if float_bits(&to_llty) > float_bits(&from_llty) {
                    "fpext"
                } else {
                    "fptrunc"
                }
            }
            (false, false) => {
                // int width change: pick by target width vs source width.
                let w_to = int_bits(&to_llty);
                let w_from = int_bits(&from_llty);
                if w_to > w_from {
                    if from_signed { "sext" } else { "zext" }
                } else {
                    "trunc"
                }
            }
        };
        // NVPTX: a direct float->i64 conversion (`fptoui/fptosi ... to i64`) is miscompiled by llc and returns garbage
        // (the prefill's embedding gather casts the token id f32->usize=i64 and read wild OOB rows). Go via i32, then
        // widen; poot's float->int casts are small indices (< 2^31), so the i32 step is lossless. (The SPIR-V backend
        // also does f32->i32, where it is correct.)
        if from_float && !to_float && self.target == Target::Nvptx && to_llty == "i64" {
            let i32op = if to_signed { "fptosi" } else { "fptoui" };
            let widen = if to_signed { "sext" } else { "zext" };
            let mid = self.fresh();
            let _ = writeln!(w, "  {mid} = {i32op} {from_llty} {v} to i32");
            let t = self.fresh();
            let _ = writeln!(w, "  {t} = {widen} i32 {mid} to i64");
            return Ok((t, to_llty));
        }
        // AMDGPU: scalar bf16<->f32 via fptrunc/fpext lowers to ocml device-lib calls that are absent in this
        // toolchain, so the conversion silently yields 0 (card 045). bf16 is the top 16 bits of an f32, so emit the
        // conversion with integer bit-ops.
        if matches!(self.target, Target::AmdGcn(_)) && from_float && to_float {
            if from_llty == "float" && to_llty == "bfloat" {
                // f32 -> bf16, round-to-nearest-even: bias = 0x7FFF + ((bits>>16)&1); (bits+bias)>>16. The bias
                // add is wrong for a NaN (a payload in the low bits carries into the exponent and yields Inf, an
                // all-ones payload wraps to -0.0), so a NaN takes its top 16 bits with the quiet bit forced, the
                // same value `poot_runtime_common::f32_to_bf16` produces on the host.
                let iv = self.fresh();
                let _ = writeln!(w, "  {iv} = bitcast float {v} to i32");
                let sh = self.fresh();
                let _ = writeln!(w, "  {sh} = lshr i32 {iv}, 16");
                let is_nan = self.fresh();
                let _ = writeln!(w, "  {is_nan} = fcmp uno float {v}, {v}");
                let quiet = self.fresh();
                let _ = writeln!(w, "  {quiet} = or i32 {sh}, 64");
                let lsb = self.fresh();
                let _ = writeln!(w, "  {lsb} = and i32 {sh}, 1");
                let bias = self.fresh();
                let _ = writeln!(w, "  {bias} = add i32 {lsb}, 32767");
                let rnd = self.fresh();
                let _ = writeln!(w, "  {rnd} = add i32 {iv}, {bias}");
                let rounded = self.fresh();
                let _ = writeln!(w, "  {rounded} = lshr i32 {rnd}, 16");
                let top = self.fresh();
                let _ = writeln!(
                    w,
                    "  {top} = select i1 {is_nan}, i32 {quiet}, i32 {rounded}"
                );
                let tr = self.fresh();
                let _ = writeln!(w, "  {tr} = trunc i32 {top} to i16");
                let res = self.fresh();
                let _ = writeln!(w, "  {res} = bitcast i16 {tr} to bfloat");
                return Ok((res, to_llty));
            }
            if from_llty == "bfloat" && to_llty == "float" {
                // bf16 -> f32: the 16 bits become the HIGH half of the f32 (lossless).
                let iv = self.fresh();
                let _ = writeln!(w, "  {iv} = bitcast bfloat {v} to i16");
                let z = self.fresh();
                let _ = writeln!(w, "  {z} = zext i16 {iv} to i32");
                let sh = self.fresh();
                let _ = writeln!(w, "  {sh} = shl i32 {z}, 16");
                let res = self.fresh();
                let _ = writeln!(w, "  {res} = bitcast i32 {sh} to float");
                return Ok((res, to_llty));
            }
        }
        let t = self.fresh();
        let _ = writeln!(w, "  {t} = {op} {from_llty} {v} to {to_llty}");
        Ok((t, to_llty))
    }

    pub(super) fn emit_fp8_decode(
        &mut self,
        format: Fp8Format,
        operand: &Operand,
        dest_ty: &Ty,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        if self.operand_ty(operand)? != Ty::U32 || *dest_ty != Ty::F32 {
            return Err(EmitError::Unsupported(format!(
                "Fp8Decode requires U32 byte carrier -> F32, got {:?} -> {dest_ty:?}",
                self.operand_ty(operand)?
            )));
        }
        match format {
            Fp8Format::E4M3Fn => {}
        }
        let (value, _) = self.operand(operand, w)?;
        let code = self.fresh();
        let _ = writeln!(w, "  {code} = and i32 {value}, 255");
        let sign_lane = self.fresh();
        let _ = writeln!(w, "  {sign_lane} = and i32 {code}, 128");
        let sign = self.fresh();
        let _ = writeln!(w, "  {sign} = shl i32 {sign_lane}, 24");
        let shifted = self.fresh();
        let _ = writeln!(w, "  {shifted} = lshr i32 {code}, 3");
        let exponent = self.fresh();
        let _ = writeln!(w, "  {exponent} = and i32 {shifted}, 15");
        let mantissa = self.fresh();
        let _ = writeln!(w, "  {mantissa} = and i32 {code}, 7");

        let exponent_is_zero = self.fresh();
        let _ = writeln!(w, "  {exponent_is_zero} = icmp eq i32 {exponent}, 0");
        let exponent_is_special = self.fresh();
        let _ = writeln!(w, "  {exponent_is_special} = icmp eq i32 {exponent}, 15");
        let mantissa_is_nan = self.fresh();
        let _ = writeln!(w, "  {mantissa_is_nan} = icmp eq i32 {mantissa}, 7");
        let is_nan = self.fresh();
        let _ = writeln!(
            w,
            "  {is_nan} = and i1 {exponent_is_special}, {mantissa_is_nan}"
        );

        let subnormal_integer = self.fresh();
        let _ = writeln!(w, "  {subnormal_integer} = uitofp i32 {mantissa} to float");
        let subnormal = self.fresh();
        let _ = writeln!(
            w,
            "  {subnormal} = fmul float {subnormal_integer}, 0x3F60000000000000"
        );
        let subnormal_bits = self.fresh();
        let _ = writeln!(w, "  {subnormal_bits} = bitcast float {subnormal} to i32");
        let signed_subnormal_bits = self.fresh();
        let _ = writeln!(
            w,
            "  {signed_subnormal_bits} = or i32 {subnormal_bits}, {sign}"
        );

        let normal_exponent = self.fresh();
        let _ = writeln!(w, "  {normal_exponent} = add i32 {exponent}, 120");
        let normal_exponent_bits = self.fresh();
        let _ = writeln!(
            w,
            "  {normal_exponent_bits} = shl i32 {normal_exponent}, 23"
        );
        let normal_mantissa_bits = self.fresh();
        let _ = writeln!(w, "  {normal_mantissa_bits} = shl i32 {mantissa}, 20");
        let normal_magnitude_bits = self.fresh();
        let _ = writeln!(
            w,
            "  {normal_magnitude_bits} = or i32 {normal_exponent_bits}, {normal_mantissa_bits}"
        );
        let signed_normal_bits = self.fresh();
        let _ = writeln!(
            w,
            "  {signed_normal_bits} = or i32 {normal_magnitude_bits}, {sign}"
        );
        let finite_bits = self.fresh();
        let _ = writeln!(
            w,
            "  {finite_bits} = select i1 {exponent_is_zero}, i32 {signed_subnormal_bits}, i32 {signed_normal_bits}"
        );
        let result_bits = self.fresh();
        let _ = writeln!(
            w,
            "  {result_bits} = select i1 {is_nan}, i32 2143289344, i32 {finite_bits}"
        );
        let result = self.fresh();
        let _ = writeln!(w, "  {result} = bitcast i32 {result_bits} to float");
        Ok((result, "float".into()))
    }

    pub(super) fn emit_fp8_encode(
        &mut self,
        format: Fp8Format,
        operand: &Operand,
        dest_ty: &Ty,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        if self.operand_ty(operand)? != Ty::F32 || *dest_ty != Ty::U32 {
            return Err(EmitError::Unsupported(format!(
                "Fp8Encode requires F32 -> U32 byte carrier, got {:?} -> {dest_ty:?}",
                self.operand_ty(operand)?
            )));
        }
        match format {
            Fp8Format::E4M3Fn => {}
        }
        let (value, _) = self.operand(operand, w)?;
        let bits = self.fresh();
        let _ = writeln!(w, "  {bits} = bitcast float {value} to i32");
        let sign_bits = self.fresh();
        let _ = writeln!(w, "  {sign_bits} = and i32 {bits}, -2147483648");
        let sign = self.fresh();
        let _ = writeln!(w, "  {sign} = lshr i32 {sign_bits}, 24");
        let magnitude_bits = self.fresh();
        let _ = writeln!(w, "  {magnitude_bits} = and i32 {bits}, 2147483647");
        let magnitude = self.fresh();
        let _ = writeln!(w, "  {magnitude} = bitcast i32 {magnitude_bits} to float");
        let is_nan = self.fresh();
        let _ = writeln!(w, "  {is_nan} = fcmp uno float {value}, {value}");

        let mut selected = "0".to_string();
        for upper in 1u8..=0x7e {
            let lower_value = positive_e4m3fn_value(upper - 1);
            let upper_value = positive_e4m3fn_value(upper);
            let midpoint = (lower_value + upper_value) * 0.5;
            let midpoint_literal = format!("0x{:016X}", (midpoint as f64).to_bits());
            let choose_upper = self.fresh();
            let comparison = if upper & 1 == 0 { "oge" } else { "ogt" };
            let _ = writeln!(
                w,
                "  {choose_upper} = fcmp {comparison} float {magnitude}, {midpoint_literal}"
            );
            let next = self.fresh();
            let _ = writeln!(
                w,
                "  {next} = select i1 {choose_upper}, i32 {upper}, i32 {selected}"
            );
            selected = next;
        }
        let signed = self.fresh();
        let _ = writeln!(w, "  {signed} = or i32 {selected}, {sign}");
        let result = self.fresh();
        let _ = writeln!(w, "  {result} = select i1 {is_nan}, i32 127, i32 {signed}");
        Ok((result, "i32".into()))
    }

    /// Reinterpret an operand's bits as `to` (e.g. `f32::from_bits`: read an f32 packed into a u32 buffer). A no-op
    /// when the LLVM types already match; otherwise an LLVM `bitcast` (same bit width, both backends).
    pub(super) fn emit_bitcast(
        &mut self,
        to: &Ty,
        operand: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let (v, from_llty) = self.operand(operand, w)?;
        let to_llty = self.scalar_llty(to)?;
        if from_llty == to_llty {
            return Ok((v, to_llty));
        }
        let t = self.fresh();
        let _ = writeln!(w, "  {t} = bitcast {from_llty} {v} to {to_llty}");
        Ok((t, to_llty))
    }

    pub(super) fn emit_unary(
        &mut self,
        op: poot_kernel_ir::UnOp,
        a: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        use poot_kernel_ir::UnOp;
        let (va, ty) = self.operand(a, w)?;
        let opnd_ty = self.operand_ty(a)?;
        let t = self.fresh();
        match op {
            UnOp::Neg => {
                if opnd_ty.is_float() {
                    let _ = writeln!(w, "  {t} = fneg {ty} {va}");
                } else {
                    let _ = writeln!(w, "  {t} = sub {ty} 0, {va}");
                }
            }
            UnOp::Not => {
                let mask = if opnd_ty == Ty::Bool { "true" } else { "-1" };
                let _ = writeln!(w, "  {t} = xor {ty} {va}, {mask}");
            }
        }
        Ok((t, ty))
    }

    pub(super) fn emit_int_scalar_unary(
        &mut self,
        op: poot_kernel_ir::IntScalarOp,
        a: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let (va, ty) = self.operand(a, w)?;
        match op {
            poot_kernel_ir::IntScalarOp::LeadingZeros => match self.target {
                Target::SpirvVulkan | Target::AieCore => self.emit_clz_i32_software(&va, w),
                Target::Nvptx | Target::AmdGcn(_) => {
                    self.declares
                        .insert("declare i32 @llvm.ctlz.i32(i32, i1)".into());
                    let t = self.fresh();
                    let _ = writeln!(w, "  {t} = call i32 @llvm.ctlz.i32({ty} {va}, i1 false)");
                    Ok((t, "i32".into()))
                }
            },
            other => Err(EmitError::Unsupported(format!(
                "integer scalar unary {other:?}"
            ))),
        }
    }

    /// Portable leading-zero count of an i32 word as unsigned bits. SPIR-V GlobalISel cannot legalize
    /// `llvm.ctlz.i32`, so this uses only logical shifts, `icmp`, and `select`.
    pub(super) fn emit_clz_i32_software(
        &mut self,
        value: &str,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let mut x = value.to_string();
        let mut n = self.fresh();
        let _ = writeln!(w, "  {n} = add i32 0, 32");
        for shift in [16, 8, 4, 2, 1] {
            let y = self.fresh();
            let nz = self.fresh();
            let n_minus = self.fresh();
            let next_n = self.fresh();
            let next_x = self.fresh();
            let _ = writeln!(w, "  {y} = lshr i32 {x}, {shift}");
            let _ = writeln!(w, "  {nz} = icmp ne i32 {y}, 0");
            let _ = writeln!(w, "  {n_minus} = sub i32 {n}, {shift}");
            let _ = writeln!(w, "  {next_n} = select i1 {nz}, i32 {n_minus}, i32 {n}");
            let _ = writeln!(w, "  {next_x} = select i1 {nz}, i32 {y}, i32 {x}");
            n = next_n;
            x = next_x;
        }
        let result = self.fresh();
        let _ = writeln!(w, "  {result} = sub i32 {n}, {x}");
        Ok((result, "i32".into()))
    }

    pub(super) fn emit_mathunary(
        &mut self,
        op: poot_kernel_ir::MathOp,
        a: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        use poot_kernel_ir::MathOp;
        let (va, ty) = self.operand(a, w)?;
        // exp needs a per-backend split (NVPTX has no llvm.exp lowering); the rest use a generic llvm
        // intrinsic that lowers on both. (sin/cos unneeded: RoPE uses precomputed cos/sin tables.)
        if op == MathOp::Exp {
            return self.emit_exp(&va, &ty, w);
        }
        if op == MathOp::Log {
            return self.emit_log(&va, &ty, w);
        }
        let intrin = match op {
            MathOp::Sqrt => "llvm.sqrt.f32",
            MathOp::Abs => "llvm.fabs.f32",
            MathOp::Floor => "llvm.floor.f32",
            MathOp::Ceil => "llvm.ceil.f32",
            MathOp::Trunc => "llvm.trunc.f32",
            MathOp::Round => "llvm.round.f32",
            MathOp::Exp | MathOp::Log => unreachable!(),
            MathOp::Sin | MathOp::Cos => {
                return Err(EmitError::Unsupported(format!(
                    "transcendental {op:?} (not needed yet)"
                )));
            }
        };
        self.declares
            .insert(format!("declare {ty} @{intrin}({ty})"));
        let t = self.fresh();
        let _ = writeln!(w, "  {t} = tail call {ty} @{intrin}({ty} {va})");
        Ok((t, ty))
    }

    /// `exp(x)`, per backend. SPIR-V: `llvm.exp.f32` -> GLSL Exp. NVPTX: `ex2.approx(x * log2 e)`.
    /// AIE-core: software sequence (no native exp; Peano libm only has fabs).
    pub(super) fn emit_exp(
        &mut self,
        va: &str,
        ty: &str,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        match self.target {
            Target::SpirvVulkan => {
                self.declares
                    .insert(format!("declare {ty} @llvm.exp.f32({ty})"));
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = tail call {ty} @llvm.exp.f32({ty} {va})");
                Ok((t, ty.to_string()))
            }
            Target::Nvptx => {
                let log2e = format!("0x{:016X}", (std::f32::consts::LOG2_E as f64).to_bits());
                let scaled = self.fresh();
                let _ = writeln!(w, "  {scaled} = fmul {ty} {va}, {log2e}");
                self.declares
                    .insert("declare float @llvm.nvvm.ex2.approx.f32(float)".to_string());
                let t = self.fresh();
                let _ = writeln!(
                    w,
                    "  {t} = tail call float @llvm.nvvm.ex2.approx.f32(float {scaled})"
                );
                Ok((t, ty.to_string()))
            }
            // AIE2p has no native float exp instruction, and Peano's libm only provides fabs variants, so `expf` would be
            // an unresolved symbol at aiecc link time. Lower to a software inline sequence using only ops Peano can lower
            // through the compiler-rt builtins (libclang_rt.builtins.a: __mulsf3, __addsf3, __fixsfsi, __gtsf2, __ltsf2
            // etc.), which the IRON aiecc build always links.
            //
            // Algorithm:
            //   1. Clamp x to [-88, 88] to keep 2^n in normal f32 range.
            //   2. n = round(x * log2e) via the "3*2^22 magic" trick: adding magic = 12582912 = 3*2^22 moves
            //      y = x*log2e into [2^23, 2^24), where f32 ULP = 1, so add-then-subtract rounds to the nearest integer
            //      even for negative y. (The simpler 2^23 magic fails for y < 0: y + 2^23 can land in [2^22, 2^23) where
            //      ULP = 0.5.)
            //   3. r = x - n * ln2 (range-reduced, |r| <= ln2/2 ~= 0.347).
            //   4. Horner poly for exp(r): 6-term Taylor 1 + r*(1 + r*(1/2 + ...)).
            //   5. Scale by 2^n: set the float exponent bits via i32 add + shl + bitcast.
            //
            // Accuracy: max_rel_err < 1e-5 for x in [-88, 88]; max_abs_err < 1e-6 for silu inputs in [-5, 5].
            Target::AieCore => {
                // Encode an f32 constant as the double-hex literal LLVM IR requires.
                let c = |v: f32| format!("0x{:016X}", (v as f64).to_bits());
                // Clamp x to [-88, 88].
                let clamp_hi = self.fresh();
                let _ = writeln!(w, "  {clamp_hi} = fcmp ogt {ty} {va}, {}", c(88.0_f32));
                let x1 = self.fresh();
                let _ = writeln!(
                    w,
                    "  {x1} = select i1 {clamp_hi}, {ty} {}, {ty} {va}",
                    c(88.0_f32)
                );
                let clamp_lo = self.fresh();
                let _ = writeln!(w, "  {clamp_lo} = fcmp olt {ty} {x1}, {}", c(-88.0_f32));
                let x2 = self.fresh();
                let _ = writeln!(
                    w,
                    "  {x2} = select i1 {clamp_lo}, {ty} {}, {ty} {x1}",
                    c(-88.0_f32)
                );
                // n = round(x * log2e) via 3*2^22 magic.
                let y = self.fresh();
                let _ = writeln!(w, "  {y} = fmul {ty} {x2}, {}", c(std::f32::consts::LOG2_E));
                let yp = self.fresh();
                let _ = writeln!(w, "  {yp} = fadd {ty} {y}, {}", c(12582912.0_f32)); // 3*2^22
                let n_f = self.fresh();
                let _ = writeln!(w, "  {n_f} = fsub {ty} {yp}, {}", c(12582912.0_f32));
                let n_i = self.fresh();
                let _ = writeln!(w, "  {n_i} = fptosi {ty} {n_f} to i32");
                // r = x - n * ln2.
                let rh = self.fresh();
                let _ = writeln!(w, "  {rh} = fmul {ty} {n_f}, {}", c(std::f32::consts::LN_2));
                let r = self.fresh();
                let _ = writeln!(w, "  {r} = fsub {ty} {x2}, {rh}");
                // Horner polynomial: 1 + r*(1 + r*(0.5 + r*(1/6 + r*(1/24 + r/120)))).
                let s5 = self.fresh();
                let _ = writeln!(w, "  {s5} = fmul {ty} {r}, {}", c(1.0_f32 / 120.0_f32));
                let s4 = self.fresh();
                let _ = writeln!(w, "  {s4} = fadd {ty} {s5}, {}", c(1.0_f32 / 24.0_f32));
                let s3 = self.fresh();
                let _ = writeln!(w, "  {s3} = fmul {ty} {r}, {s4}");
                let s3a = self.fresh();
                let _ = writeln!(w, "  {s3a} = fadd {ty} {s3}, {}", c(1.0_f32 / 6.0_f32));
                let s2 = self.fresh();
                let _ = writeln!(w, "  {s2} = fmul {ty} {r}, {s3a}");
                let s2a = self.fresh();
                let _ = writeln!(w, "  {s2a} = fadd {ty} {s2}, {}", c(0.5_f32));
                let s1 = self.fresh();
                let _ = writeln!(w, "  {s1} = fmul {ty} {r}, {s2a}");
                let s1a = self.fresh();
                let _ = writeln!(w, "  {s1a} = fadd {ty} {s1}, {}", c(1.0_f32));
                let s0 = self.fresh();
                let _ = writeln!(w, "  {s0} = fmul {ty} {r}, {s1a}");
                let poly = self.fresh();
                let _ = writeln!(w, "  {poly} = fadd {ty} {s0}, {}", c(1.0_f32));
                // Scale by 2^n: adjust the float exponent bits.
                let exp_adj = self.fresh();
                let _ = writeln!(w, "  {exp_adj} = add i32 {n_i}, 127");
                let scale_bits = self.fresh();
                let _ = writeln!(w, "  {scale_bits} = shl i32 {exp_adj}, 23");
                let scale = self.fresh();
                let _ = writeln!(w, "  {scale} = bitcast i32 {scale_bits} to {ty}");
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = fmul {ty} {poly}, {scale}");
                Ok((t, ty.to_string()))
            }
            // AMDGPU: clang + --rocm-device-lib-path links ocml, which resolves llvm.exp.f32 to __ocml_exp_f32 (same
            // intrinsic as SPIR-V).
            Target::AmdGcn(_) => {
                self.declares
                    .insert(format!("declare {ty} @llvm.exp.f32({ty})"));
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = tail call {ty} @llvm.exp.f32({ty} {va})");
                Ok((t, ty.to_string()))
            }
        }
    }

    /// `ln(x)`. SPIR-V uses the generic `llvm.log.f32`; NVPTX has no llvm.log lowering, so it uses the hardware
    /// `lg2.approx` (log base 2) scaled by `ln(2)`: `ln(x) = log2(x) * ln(2)`. AIE-core uses a software sequence
    /// (see `emit_exp`).
    pub(super) fn emit_log(
        &mut self,
        va: &str,
        ty: &str,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        match self.target {
            Target::SpirvVulkan => {
                self.declares
                    .insert(format!("declare {ty} @llvm.log.f32({ty})"));
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = tail call {ty} @llvm.log.f32({ty} {va})");
                Ok((t, ty.to_string()))
            }
            Target::Nvptx => {
                // NOTE: the single-precision lg2 intrinsic is `.f`, not `.f32` (unlike ex2.approx.f32); LLVM's NVVM naming is
                // inconsistent. With `.f32` llc emits an extern call instead of the hardware `lg2.approx`, and the driver
                // rejects the PTX (CUDA_ERROR_INVALID_PTX).
                self.declares
                    .insert("declare float @llvm.nvvm.lg2.approx.f(float)".to_string());
                let lg2 = self.fresh();
                let _ = writeln!(
                    w,
                    "  {lg2} = tail call float @llvm.nvvm.lg2.approx.f(float {va})"
                );
                let ln2 = format!("0x{:016X}", (std::f32::consts::LN_2 as f64).to_bits());
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = fmul {ty} {lg2}, {ln2}");
                Ok((t, ty.to_string()))
            }
            // AIE2p: software ln(x) via IEEE bit decomposition + atanh-series poly.
            //
            // Algorithm:
            //   1. Extract the integer exponent n from the float bits: n = (bits >> 23) & 0xFF - 127.
            //   2. Set the exponent to 127: m = x * 2^(-n), m in [1, 2).
            //   3. ln(m) via the atanh series: u = (m-1)/(m+1), ln(m) = 2*atanh(u)
            //      = 2*u*(1 + u^2/3 + u^4/5 + u^6/7 + u^8/9). u is in [0, 1/3), so 5 terms give f32-accurate results
            //      (max_rel_err < 1e-4; abs_err < 1e-6 near x=1).
            //   4. ln(x) = n * ln(2) + ln(m).
            //
            // Accuracy: max_rel_err < 1e-4 for x in (0, inf); better away from x~=1, where |ln(x)| is tiny and relative
            // error magnifies.
            Target::AieCore => {
                let c = |v: f32| format!("0x{:016X}", (v as f64).to_bits());
                // Extract exponent bits: e_field = (bits >> 23) & 0xFF; n = e_field - 127.
                let xi = self.fresh();
                let _ = writeln!(w, "  {xi} = bitcast {ty} {va} to i32");
                let xi_shr = self.fresh();
                let _ = writeln!(w, "  {xi_shr} = lshr i32 {xi}, 23");
                let e_field = self.fresh();
                let _ = writeln!(w, "  {e_field} = and i32 {xi_shr}, 255");
                let n_i = self.fresh();
                let _ = writeln!(w, "  {n_i} = sub i32 {e_field}, 127");
                // m = x with exponent set to 127: bits = (bits & 0x7FFFFF) | 0x3F800000.
                let mantissa = self.fresh();
                let _ = writeln!(w, "  {mantissa} = and i32 {xi}, 8388607"); // 0x7FFFFF
                let m_bits = self.fresh();
                let _ = writeln!(w, "  {m_bits} = or i32 {mantissa}, 1065353216"); // 0x3F800000
                let m = self.fresh();
                let _ = writeln!(w, "  {m} = bitcast i32 {m_bits} to {ty}");
                // t = m - 1; u = t / (t + 2); ln(m) = 2*atanh(u).
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = fsub {ty} {m}, {}", c(1.0_f32));
                let t2 = self.fresh();
                let _ = writeln!(w, "  {t2} = fadd {ty} {t}, {}", c(2.0_f32));
                let u = self.fresh();
                let _ = writeln!(w, "  {u} = fdiv {ty} {t}, {t2}");
                let u2 = self.fresh();
                let _ = writeln!(w, "  {u2} = fmul {ty} {u}, {u}");
                // Horner: 1 + u^2*(1/3 + u^2*(1/5 + u^2*(1/7 + u^2/9))).
                let p9 = self.fresh();
                let _ = writeln!(w, "  {p9} = fmul {ty} {u2}, {}", c(1.0_f32 / 9.0_f32));
                let p7 = self.fresh();
                let _ = writeln!(w, "  {p7} = fadd {ty} {p9}, {}", c(1.0_f32 / 7.0_f32));
                let p6 = self.fresh();
                let _ = writeln!(w, "  {p6} = fmul {ty} {u2}, {p7}");
                let p5 = self.fresh();
                let _ = writeln!(w, "  {p5} = fadd {ty} {p6}, {}", c(1.0_f32 / 5.0_f32));
                let p4 = self.fresh();
                let _ = writeln!(w, "  {p4} = fmul {ty} {u2}, {p5}");
                let p3 = self.fresh();
                let _ = writeln!(w, "  {p3} = fadd {ty} {p4}, {}", c(1.0_f32 / 3.0_f32));
                let p2 = self.fresh();
                let _ = writeln!(w, "  {p2} = fmul {ty} {u2}, {p3}");
                let p1 = self.fresh();
                let _ = writeln!(w, "  {p1} = fadd {ty} {p2}, {}", c(1.0_f32));
                let atanh_half = self.fresh();
                let _ = writeln!(w, "  {atanh_half} = fmul {ty} {u}, {p1}");
                let ln_m = self.fresh();
                let _ = writeln!(w, "  {ln_m} = fadd {ty} {atanh_half}, {atanh_half}"); // * 2
                // ln(x) = n * ln(2) + ln(m).
                let n_f = self.fresh();
                let _ = writeln!(w, "  {n_f} = sitofp i32 {n_i} to {ty}");
                let n_ln2 = self.fresh();
                let _ = writeln!(
                    w,
                    "  {n_ln2} = fmul {ty} {n_f}, {}",
                    c(std::f32::consts::LN_2)
                );
                let t_final = self.fresh();
                let _ = writeln!(w, "  {t_final} = fadd {ty} {n_ln2}, {ln_m}");
                Ok((t_final, ty.to_string()))
            }
            // AMDGPU: clang + ocml resolves llvm.log.f32 to __ocml_log_f32. Same as SPIR-V.
            Target::AmdGcn(_) => {
                self.declares
                    .insert(format!("declare {ty} @llvm.log.f32({ty})"));
                let t = self.fresh();
                let _ = writeln!(w, "  {t} = tail call {ty} @llvm.log.f32({ty} {va})");
                Ok((t, ty.to_string()))
            }
        }
    }

    pub(super) fn emit_binop(
        &mut self,
        op: BinOp,
        a: &Operand,
        b: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let (va, tya) = self.operand(a, w)?;
        let (vb, _tyb) = self.operand(b, w)?;
        let opnd_ty = self.operand_ty(a)?;
        let is_float = opnd_ty.is_float();
        let signed = opnd_ty.is_signed_int();
        // Min/Max lower to a scalar-named intrinsic call (`llvm.minnum.f32`, not a `.v4f32` variant), so reject them
        // on a `Ty::Vec` operand instead of emitting a bogus declare/call (FR-007). Add/Sub/Mul/etc reuse the plain
        // mnemonic path below, which LLVM applies lanewise on `<N x T>` (spec 134 P1: no vector-specific opcodes).
        if matches!(op, BinOp::Min | BinOp::Max) && tya.starts_with('<') {
            return Err(EmitError::Unsupported(format!(
                "BinOp::{op:?} on a Ty::Vec operand ({tya}): only elementwise Add/Sub/Mul/etc are wired \
                 for vectors this increment, not the Min/Max intrinsic path"
            )));
        }
        // Min/Max lower to an intrinsic call, not a plain binary instruction.
        if matches!(op, BinOp::Min | BinOp::Max) {
            let intrin = match (op, is_float) {
                (BinOp::Min, true) => "llvm.minnum.f32",
                (BinOp::Max, true) => "llvm.maxnum.f32",
                (BinOp::Min, false) if signed => "llvm.smin.i32",
                (BinOp::Min, false) => "llvm.umin.i32",
                (BinOp::Max, false) if signed => "llvm.smax.i32",
                (BinOp::Max, false) => "llvm.umax.i32",
                _ => unreachable!(),
            };
            self.declares
                .insert(format!("declare {tya} @{intrin}({tya}, {tya})"));
            let t = self.fresh();
            let _ = writeln!(
                w,
                "  {t} = tail call {tya} @{intrin}({tya} {va}, {tya} {vb})"
            );
            return Ok((t, tya));
        }
        // NVPTX has no native integer remainder; llc software-emulates i64 `urem`/`srem` and miscompiles it for large
        // dividends, returning wrong values. That corrupted index math (unraveling a linear thread id into tensor
        // coords via `i % shape`) at large shapes (prefill garbled at n=864, fine at n<=421; a power-of-2 modulus was
        // safe because llc lowers it to AND). `udiv`/`sdiv` are correct, so lower integer remainder as
        // `a - (a/b)*b`.
        if matches!(op, BinOp::Rem) && !is_float && self.target == Target::Nvptx {
            let divi = if signed { "sdiv" } else { "udiv" };
            let q = self.fresh();
            let _ = writeln!(w, "  {q} = {divi} {tya} {va}, {vb}");
            let m = self.fresh();
            let _ = writeln!(w, "  {m} = mul {tya} {q}, {vb}");
            let t = self.fresh();
            let _ = writeln!(w, "  {t} = sub {tya} {va}, {m}");
            return Ok((t, tya));
        }
        let t = self.fresh();
        let (inst, result_ty) = binop_inst(op, is_float, signed);
        // comparison instructions take the operand type but yield i1.
        let _ = writeln!(w, "  {t} = {inst} {tya} {va}, {vb}", tya = tya);
        // card 628: keep `float_binop_ordinal` counting every float Add/Sub/Mul this body emits (marked or
        // not), so `emit_binop_no_contract`'s ordinals stay correct positions in the SpirvVulkan stream.
        if self.target == Target::SpirvVulkan
            && is_float
            && matches!(op, BinOp::Add | BinOp::Sub | BinOp::Mul)
        {
            self.float_binop_ordinal += 1;
        }
        let res_llty = match result_ty {
            ResultTy::Bool => "i1".to_string(),
            ResultTy::Same => tya,
        };
        Ok((t, res_llty))
    }

    /// Emit a marked no-contraction float `Add`/`Sub`/`Mul` (card 628): `op` and the operand
    /// type are already float/Add-Sub-Mul by construction (`Body::verify`'s `BinaryOpNoContract` arm), so
    /// this only needs the per-target lowering, not a redundant re-check.
    ///
    /// - `SpirvVulkan`: emitted as the ordinary plain instruction (`fadd`/`fsub`/`fmul`, byte-identical to
    ///   an unmarked `BinaryOp`), but its [`Emitter::float_binop_ordinal`] is recorded in
    ///   `no_contract_ordinals` first. llc's SPIR-V backend does not translate any LLVM-level fast-math flag
    ///   or intrinsic into a `NoContraction` decoration (checked directly: a `contract`-flagged and a
    ///   plain `fmul`/`fsub` compile to byte-identical SPIR-V, neither decorated), so the decoration has to
    ///   be added to the compiled binary; `compile_with` does that positionally via `spirv_postprocess`,
    ///   using these ordinals (checked positionally since -O0 never reorders/merges/drops a float binop).
    /// - `Nvptx`/`AmdGcn`: an `llvm.experimental.constrained.{fadd,fsub,fmul}.f32` call, which needs the
    ///   `strictfp` function attribute (`strictfp_needed`, consumed by `emit_signature`/`emit_footer`).
    ///   LLVM's DAG combiner never forms an FMA from a constrained node, independent of the backend's own
    ///   contraction default - checked directly: a plain, unflagged `fmul`+`fsub` pair stays two
    ///   instructions on both backends' default (`-mcpu=sm_80`/`gfx1151`) already, but a `contract`-flagged
    ///   pair fuses into `fma.rn.f32` (NVPTX) / `s_fmac_f32` (AMDGCN); the constrained form stays unfused
    ///   the same way the plain form does today, immune to a future default change since it does not rely
    ///   on flag absence. The compiled AMDGCN HSACO is the device's final ISA (ahead-of-time, no further
    ///   JIT), so this is a complete fix there; NVPTX's compiled `.ptx` is JIT-recompiled by the CUDA
    ///   driver's `ptxas` at kernel launch (outside poot-codegen's own toolchain and this box's hardware -
    ///   SC-001 runs on the PTX pod), so this is poot-codegen's best available per-instruction
    ///   signal there; its `.ptx` text is confirmed unfused today, matching the plain path.
    /// - `AieCore`: not in this card's scope (Owns: SpirvVulkan/AmdGcn/Nvptx only) - a named `Unsupported`
    ///   rather than a silently-wrong lowering, matching this crate's other per-target rejections.
    pub(super) fn emit_binop_no_contract(
        &mut self,
        op: BinOp,
        a: &Operand,
        b: &Operand,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        if self.target == Target::AieCore {
            return Err(EmitError::Unsupported(
                "BinaryOpNoContract on AieCore: card 628 wires SpirvVulkan/AmdGcn/Nvptx only"
                    .into(),
            ));
        }
        let (va, tya) = self.operand(a, w)?;
        let (vb, _tyb) = self.operand(b, w)?;
        let (inst, _) = binop_inst(op, true, false);
        let t = self.fresh();
        if self.target == Target::SpirvVulkan {
            let _ = writeln!(w, "  {t} = {inst} {tya} {va}, {vb}");
            self.no_contract_ordinals.push(self.float_binop_ordinal);
            self.float_binop_ordinal += 1;
        } else {
            let intrin = format!("llvm.experimental.constrained.{inst}.f32");
            self.declares.insert(format!(
                "declare {tya} @{intrin}({tya}, {tya}, metadata, metadata)"
            ));
            self.strictfp_needed = true;
            let _ = writeln!(
                w,
                "  {t} = call {tya} @{intrin}({tya} {va}, {tya} {vb}, metadata !\"round.tonearest\", \
                 metadata !\"fpexcept.ignore\") #{NO_CONTRACT_STRICTFP_ATTR}"
            );
        }
        Ok((t, tya))
    }
}
