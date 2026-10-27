//! Matrix-fragment codegen probes. No generated contraction runs a fragment body (POOT-1403: every M > 1
//! contraction is the register-tiled GEMM); the fragment IR and its lowering stay, checked here and by the
//! `wmma_tile` fixtures, for a fragment step of that GEMM.

use crate::helpers::{Alloc, elem, ld, local};
use poot_kernel_ir::{
    BasicBlock, Body, Constant, FragmentTy, Operand, Place, Rvalue, Statement, Terminator, Ty,
    WmmaDtype, WmmaMat, WmmaShape,
};

mod tests {
    use poot_codegen::{Target, compile};

    use super::test_only::wmma_loadlds_probe;

    /// WmmaLoadLds (WMMA-load a fragment from a bf16 LDS tile) lowers to NVPTX with the tensor-core
    /// load reading `.shared`. Split out of `tests/bf16_probe.rs` (card 546c: its only other caller,
    /// poot-rocm-device's schedule-batch atomicity test, was deleted with the resident E4M3 region's
    /// `compile_schedule` stack, which made this an external-only consumer that `pub(crate)` cannot
    /// serve). The enabling capability for the fused tensor-core dequant gemm.
    #[test]
    fn wmma_loadlds_nvptx() {
        let dir = std::env::temp_dir().join("poot-wmma-lds");
        std::fs::create_dir_all(&dir).unwrap();
        let body = wmma_loadlds_probe("k");
        let out = dir.join("wmma_loadlds.ptx");
        let _ = compile(&body, Target::Nvptx, &out)
            .unwrap_or_else(|e| panic!("wmma_loadlds compile failed: {e}"));
        let ptx = std::fs::read_to_string(&out).unwrap();
        let n_load = ptx.matches("wmma.load").count();
        let shared = ptx.contains(".shared") || ptx.contains("shared");
        eprintln!("wmma_loadlds PTX: load={n_load} shared={shared}");
        assert!(n_load >= 2, "expected 2 wmma.load (A+B from LDS):\n{ptx}");
        assert!(shared, "expected the fragment load to read .shared (LDS)");
        assert!(compile(&body, Target::SpirvVulkan, &dir.join("x.spv")).is_err());
    }
}

pub(crate) mod test_only {
    use super::*;

    /// Probe: WMMA-load both A and B fragments from a bf16 LDS tile (`WmmaLoadLds`), mma, store to a global f32 C.
    /// Exercises the `WmmaLoadLds` capability the fused tensor-core dequant gemm needs (it dequants a packed int4
    /// B-tile into a bf16 LDS tile, then WMMA-loads it). Compile-only: the LDS is zero-initialized here, so it is a
    /// codegen check (does `wmma.load` from `.shared` lower), not a value test. NVPTX-only (see `WmmaLoadLds`'s doc).
    pub(crate) fn wmma_loadlds_probe(name: &str) -> Body {
        use crate::helpers::slice_f32;
        use poot_kernel_ir::WorkgroupLocalDecl;
        let shape = WmmaShape::M16N16K16;
        let mut al = Alloc::new(vec![
            ld(Ty::Unit, false),
            ld(slice_f32(true), true), // 1 c (f32 out [16x16])
        ]);
        let c = local(1);
        let idx0 = al.add(Ty::Usize, false);
        let af = al.add(
            Ty::Fragment(FragmentTy::matrix(WmmaMat::A, WmmaDtype::Bf16, shape)),
            false,
        );
        let bf = al.add(
            Ty::Fragment(FragmentTy::matrix(WmmaMat::B, WmmaDtype::Bf16, shape)),
            false,
        );
        let acc = al.add(
            Ty::Fragment(FragmentTy::accumulator(WmmaDtype::Bf16, shape)),
            false,
        );

        let stmts = vec![
            Statement::Assign(
                Place::local(idx0),
                Rvalue::Use(Operand::Const(Constant::Usize(0))),
            ),
            Statement::WmmaZero {
                dtype: WmmaDtype::Bf16,
                shape: WmmaShape::M16N16K16,
                dst: acc,
            },
            Statement::WmmaLoadLds {
                which: WmmaMat::A,
                dtype: WmmaDtype::Bf16,
                shape: WmmaShape::M16N16K16,
                array: 0,
                stride: 16,
                dst: af,
            },
            Statement::WmmaLoadLds {
                which: WmmaMat::B,
                dtype: WmmaDtype::Bf16,
                shape: WmmaShape::M16N16K16,
                array: 0,
                stride: 16,
                dst: bf,
            },
            Statement::WmmaMma {
                dtype: WmmaDtype::Bf16,
                shape: WmmaShape::M16N16K16,
                a: af,
                b: bf,
                c: acc,
                dst: acc,
            },
            Statement::WmmaStore {
                dtype: WmmaDtype::Bf16,
                shape: WmmaShape::M16N16K16,
                tile: elem(c, idx0),
                stride: 16,
                src: acc,
            },
        ];
        let bb0 = BasicBlock {
            statements: stmts,
            terminator: Terminator::Return,
        };
        let mut body = Body::new(name, 1, al.locals, vec![bb0]);
        body.workgroup_size = [32, 1, 1];
        body.workgroup_locals = vec![WorkgroupLocalDecl {
            elem_ty: Ty::BF16,
            len: 256,
        }];
        body
    }
}
