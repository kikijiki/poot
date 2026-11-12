use std::fmt::Write as _;

use super::{EmitError, Emitter, Operand, Target, Ty};

impl<'a> Emitter<'a> {
    /// Lower `Rvalue::SubgroupShuffleXor`: the value `operand` holds in the lane whose `LocalX` is
    /// `LocalX ^ mask`. The verifier proved `mask < COLLECTIVE_LANES` and that every lane of the aligned 32-lane
    /// run reaches the node together, so each lowering reads a lane of the caller's own run on every device whose
    /// subgroup holds at least 32 lanes. The payload travels as an `i32` (an `F32` is bitcast around the exchange).
    pub(super) fn emit_subgroup_shuffle_xor(
        &mut self,
        operand: &Operand,
        mask: u32,
        w: &mut String,
    ) -> Result<(String, String), EmitError> {
        let ty = self.operand_ty(operand)?;
        if !matches!(ty, Ty::F32 | Ty::I32 | Ty::U32) {
            return Err(EmitError::Unsupported(format!(
                "SubgroupShuffleXor of a {ty:?}: the exchange carries F32, I32 and U32"
            )));
        }
        let (value, llty) = self.operand(operand, w)?;
        let bits = if ty == Ty::F32 {
            let t = self.fresh();
            let _ = writeln!(w, "  {t} = bitcast float {value} to i32");
            t
        } else {
            value
        };
        let received = match self.target {
            Target::AieCore => {
                return Err(EmitError::Unsupported(
                    "SubgroupShuffleXor on the AIE core: one core has no subgroup to exchange within".into(),
                ));
            }
            // `shfl.sync.bfly` with a full-warp member mask and `c = 0x1f` (segment of 32): lane `i` reads lane
            // `i ^ mask`.
            Target::Nvptx => {
                self.declares
                    .insert("declare i32 @llvm.nvvm.shfl.sync.bfly.i32(i32, i32, i32, i32)".into());
                let t = self.fresh();
                let _ = writeln!(
                    w,
                    "  {t} = tail call i32 @llvm.nvvm.shfl.sync.bfly.i32(i32 -1, i32 {bits}, i32 {mask}, i32 31)"
                );
                t
            }
            // `ds_bpermute` reads the lane whose byte address (lane index << 2) it is given. The lane index is
            // `mbcnt_hi(~0, mbcnt_lo(~0, 0))`: the lane's position in its wavefront, 0..31 on wave32 (where only
            // `mbcnt_lo` counts) and 0..63 on wave64. `mask < 32` keeps the source in the same half-wave.
            Target::AmdGcn(arch) => {
                self.declares
                    .insert("declare i32 @llvm.amdgcn.mbcnt.lo(i32, i32)".into());
                self.declares
                    .insert("declare i32 @llvm.amdgcn.ds.bpermute(i32, i32)".into());
                let lo = self.fresh();
                let _ = writeln!(
                    w,
                    "  {lo} = tail call i32 @llvm.amdgcn.mbcnt.lo(i32 -1, i32 0)"
                );
                let lane = if arch.wave == 64 {
                    self.declares
                        .insert("declare i32 @llvm.amdgcn.mbcnt.hi(i32, i32)".into());
                    let hi = self.fresh();
                    let _ = writeln!(
                        w,
                        "  {hi} = tail call i32 @llvm.amdgcn.mbcnt.hi(i32 -1, i32 {lo})"
                    );
                    hi
                } else {
                    lo
                };
                let source = self.fresh();
                let _ = writeln!(w, "  {source} = xor i32 {lane}, {mask}");
                let address = self.fresh();
                let _ = writeln!(w, "  {address} = shl i32 {source}, 2");
                let t = self.fresh();
                let _ = writeln!(
                    w,
                    "  {t} = tail call i32 @llvm.amdgcn.ds.bpermute(i32 {address}, i32 {bits})"
                );
                t
            }
            // LLVM's SPIR-V backend has no xor-shuffle intrinsic. `wave.readlane` at `SubgroupLocalInvocationId ^
            // mask` lowers to `OpGroupNonUniformShuffle` (capability `GroupNonUniformShuffle`) and is the same
            // exchange: a subgroup holds at least 32 lanes and is aligned, so the xor stays inside it.
            Target::SpirvVulkan => {
                self.declares
                    .insert("declare i32 @llvm.spv.subgroup.local.invocation.id()".into());
                self.declares
                    .insert("declare i32 @llvm.spv.wave.readlane.i32(i32, i32)".into());
                let lane = self.fresh();
                let _ = writeln!(
                    w,
                    "  {lane} = tail call i32 @llvm.spv.subgroup.local.invocation.id()"
                );
                let source = self.fresh();
                let _ = writeln!(w, "  {source} = xor i32 {lane}, {mask}");
                let t = self.fresh();
                let _ = writeln!(
                    w,
                    "  {t} = tail call i32 @llvm.spv.wave.readlane.i32(i32 {bits}, i32 {source})"
                );
                t
            }
        };
        if ty == Ty::F32 {
            let t = self.fresh();
            let _ = writeln!(w, "  {t} = bitcast i32 {received} to float");
            Ok((t, llty))
        } else {
            Ok((received, llty))
        }
    }
}
