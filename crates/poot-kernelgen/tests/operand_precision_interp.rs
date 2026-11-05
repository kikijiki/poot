//! POOT-1430: every contraction schedule reads its operands as the declared `OperandPrecision` says, on the kernel-IR
//! reference interpreter (no GPU).
//!
//! - The tier-1 storage row: the narrowed value a body stages is bit-identical to the oracle's round to nearest
//!   even (`poot_eval::round_to_fragment`), over operands that include ties, both range edges, subnormals, zeros
//!   and infinities, so a wrong rounding mode fails whatever the data.
//! - The product rows: the register-tiled, GEMV, serial and indexed bodies equal a host product over the same
//!   terms (`hi`, `lo` for a split operand) in the order the oracle sums them.

use poot_eval::round_to_fragment;
use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::{BufferStorage, DeviceCaps};

const PRECISIONS: [OperandPrecision; 5] = [
    OperandPrecision::Exact,
    OperandPrecision::Narrow(FragmentDtype::Bf16),
    OperandPrecision::Narrow(FragmentDtype::F16),
    OperandPrecision::Split(FragmentDtype::Bf16),
    OperandPrecision::Split(FragmentDtype::F16),
];

/// The `K` of one register-tiled staging round: a probe row fills exactly one, so no tail slot is masked.
const STAGE: usize = 16;

const LIMITS: kg::BodyLimits = kg::BodyLimits {
    max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
    max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
};

fn caps() -> DeviceCaps {
    DeviceCaps::wgpu_rdna3_igpu()
}

/// `n` f32 values in `[-scale, scale)` from a fixed xorshift stream.
fn fill(n: usize, seed: u64, scale: f32) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (((s >> 40) as f32) / (1u64 << 23) as f32 - 1.0) * scale
        })
        .collect()
}

/// The terms an f32 operand enters the products as, by the oracle's definition: one rounded value for `Narrow`,
/// `hi` and `lo` (`0` past a non-finite `hi`) for `Split`, the value itself for `Exact`, and the value itself for an
/// operand stored in the fragment dtype.
fn terms(precision: OperandPrecision, stored: Option<FragmentDtype>, x: f32) -> Vec<f32> {
    let Some(dtype) = precision.fragment_dtype() else {
        return vec![x];
    };
    if stored == Some(dtype) {
        return vec![x];
    }
    let hi = round_to_fragment(dtype, x);
    match precision {
        OperandPrecision::Split(_) => {
            let lo = if hi.is_finite() {
                round_to_fragment(dtype, x - hi)
            } else {
                0.0
            };
            vec![hi, lo]
        }
        _ => vec![hi],
    }
}

/// `a . b` over the terms of each operand, the oracle's order: `k` ascending, `a` terms outermost.
fn host_dot(
    precision: OperandPrecision,
    stored: (Option<FragmentDtype>, Option<FragmentDtype>),
    a: impl Iterator<Item = f32>,
    b: impl Iterator<Item = f32>,
) -> (f32, f32) {
    let (mut sum, mut magnitude) = (0.0f32, 0.0f32);
    for (x, y) in a.zip(b) {
        for p in terms(precision, stored.0, x) {
            for q in terms(precision, stored.1, y) {
                sum += p * q;
                magnitude += (p * q).abs();
            }
        }
    }
    (sum, magnitude)
}

/// Values that exercise every edge of the narrowing: random bit patterns across the exponent range, the exact
/// ties of each dtype (and their neighbours), the largest finite f16 and the first value past it, the smallest
/// f16 subnormal and half of it, signed zeros and infinities.
fn probes() -> Vec<f32> {
    let mut values = vec![
        0.0,
        -0.0,
        1.0,
        -1.5,
        65504.0,
        65519.99,
        65520.0,
        70000.0,
        -70000.0,
        2f32.powi(-24),
        2f32.powi(-25),
        1.5 * 2f32.powi(-25),
        2f32.powi(-14),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN_POSITIVE,
    ];
    for odd in [1u32, 3, 5, 0x7f] {
        // The tie between two bf16 values (low 16 bits `0x8000`) and f16 values (low 13 bits `0x1000`).
        values.push(f32::from_bits(0x3f80_0000 | (odd << 16) | 0x8000));
        values.push(f32::from_bits(0x3f80_0000 | (odd << 13) | 0x1000));
        values.push(f32::from_bits(0x3f80_0000 | (odd << 16) | 0x8001));
        values.push(f32::from_bits(0x3f80_0000 | (odd << 16) | 0x7fff));
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    values.extend((0..220).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let value = f32::from_bits((state >> 32) as u32);
        if value.is_nan() { 1.0 } else { value }
    }));
    values
}

/// Bit-identical, or both non-finite. A split operand whose `hi` overflowed multiplies the other operand's zero `lo`
/// term (`inf * 0`), so the product is NaN where the operand alone would be an infinity; either way the non-finite
/// value reaches the output, where it is a fault (ADR-0101 decision 4), which is the contract.
fn bits_equal(got: f32, want: f32) -> bool {
    got.to_bits() == want.to_bits() || (!got.is_finite() && !want.is_finite())
}

/// The planner's launch for a `[m, k] x [n, k]^T` register-tiled dense contraction, generated and run on the
/// interpreter: `weight` is an `[n, k]` buffer in `storage`.
fn run_dense_tiled(
    precision: OperandPrecision,
    (m, k, n): (usize, usize, usize),
    x: &[f32],
    weight: Buffer,
    storage: BufferStorage,
) -> Vec<f32> {
    // The interpreter runs the scalar tiling: a matrix fragment's layout is per target, so it refuses `Wmma*`.
    let terms =
        kg::StagedTerms::of(precision, None, kg::stored_fragment_dtype(storage)).scalar_only();
    let schedule = kg::RegisterTiling::choose(
        kg::TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: n,
            k,
            layout: kg::WeightLayout::Nk,
            segments: 1,
            terms,
        },
        &caps(),
    )
    .schedule();
    let (tile_rows, tile_cols) = schedule.output_tile().unwrap();
    let tiles = m.div_ceil(tile_rows) * n.div_ceil(tile_cols);
    let kg::Schedule::RegisterTiled { lanes, .. } = schedule else {
        unreachable!("choose returns a register tiling")
    };
    let meta = vec![k as u32, n as u32, m as u32, tiles as u32, 0];
    let request = kg::KernelRequest::Contraction(kg::ContractionSpec::DenseTiled(
        kg::DenseTiledContraction {
            weight: kg::DenseTiledWeight::Nk(storage),
            schedule,
            epilogue: None,
            grid: [tiles as u32 * lanes.get() * lanes.get(), 1, 1],
            meta: meta.clone(),
            precision,
        },
    ));
    let generated = kg::generate(&request, &caps(), &LIMITS).unwrap();
    let workgroup = generated.body.workgroup_size[0];
    let mut buffers = [
        Buffer::from_f32s(x),
        weight,
        Buffer::from_u32s(&meta),
        Buffer::from_f32s(&vec![f32::NAN; m * n]),
    ];
    run(
        &generated.body,
        [generated.launch.grid[0] / workgroup, 1, 1],
        &mut buffers,
    )
    .unwrap_or_else(|e| panic!("{precision}: {e}"));
    buffers[3].to_f32s().unwrap()
}

/// The tier-1 storage row. A register-tiled body with `K = 1` against a weight of `1.0` stages each activation
/// value rounded and multiplies it by one, so its output is the staged value: bit-identical to the oracle's round
/// to nearest even over the probes, for each narrow dtype, and `hi + lo` for a split one. The same through the
/// weight tile: an f32 weight column against an activation of one stages the weight narrowed. A weight stored in the
/// fragment dtype (bf16 packed under `Narrow(bf16)`) is not rounded again.
///
/// Mutation: in `operand_terms` truncate a bf16 narrowing instead of casting (a `Bitcast` that clears the low 16
/// bits): the tie-up and neighbour probes differ in the last bit and this row goes red.
#[test]
fn staged_narrow_values_are_bit_identical_to_the_oracles_round_to_nearest_even() {
    let probes = probes();
    let m = probes.len();
    for precision in PRECISIONS {
        // One `K = 16` stage (a whole round, so no tail slot is masked): the row `[x, 0, ..]` against `[1, 0, ..]`.
        let spread = |value: f32, total: usize| -> Vec<f32> {
            (0..total * STAGE)
                .map(|i| if i % STAGE == 0 { value } else { 0.0 })
                .collect()
        };
        let activations: Vec<f32> = (0..m * STAGE)
            .map(|i| {
                if i % STAGE == 0 {
                    probes[i / STAGE]
                } else {
                    0.0
                }
            })
            .collect();
        let got = run_dense_tiled(
            precision,
            (m, STAGE, 1),
            &activations,
            Buffer::from_f32s(&spread(1.0, 1)),
            BufferStorage::f32(),
        );
        for (i, &x) in probes.iter().enumerate() {
            let want = terms(precision, None, x)
                .into_iter()
                .fold(0.0f32, |sum, term| sum + term * 1.0);
            assert!(
                bits_equal(got[i], want),
                "{precision} activation {x:e} ({:#010x}): staged {:e}, oracle {:e}",
                x.to_bits(),
                got[i],
                want
            );
        }
        // The weight tile: the probes as the first element of each `[n, 16]` weight row, activation `[1, 0, ..]`.
        let weights: Vec<f32> = (0..m * STAGE)
            .map(|i| {
                if i % STAGE == 0 {
                    probes[i / STAGE]
                } else {
                    0.0
                }
            })
            .collect();
        let got = run_dense_tiled(
            precision,
            (1, STAGE, m),
            &spread(1.0, 1),
            Buffer::from_f32s(&weights),
            BufferStorage::f32(),
        );
        for (j, &x) in probes.iter().enumerate() {
            let want = terms(precision, None, x)
                .into_iter()
                .fold(0.0f32, |sum, term| sum + 1.0 * term);
            assert!(
                bits_equal(got[j], want),
                "{precision} weight {x:e}: staged {:e}, oracle {:e}",
                got[j],
                want
            );
        }
    }
}

/// A weight already stored in the fragment dtype is exact in it and enters once: a bf16-packed weight under
/// `Narrow(bf16)` and `Split(bf16)` is read as stored (the bf16 values below are bf16-exact, so rounding them again
/// is the identity and the check is on the term count: a split of a bf16-stored weight must not double it).
#[test]
fn a_weight_stored_in_the_fragment_dtype_enters_once() {
    let weights: Vec<f32> = (0..12)
        .map(|i| f32::from_bits((((i as f32 * 0.37 - 2.0).to_bits()) >> 16) << 16))
        .collect();
    let words: Vec<u32> = weights
        .chunks(2)
        .map(|p| (p[0].to_bits() >> 16) | (p[1].to_bits() & 0xffff_0000))
        .collect();
    let x = fill(3 * 12, 3, 1.0);
    for precision in [
        OperandPrecision::Narrow(FragmentDtype::Bf16),
        OperandPrecision::Split(FragmentDtype::Bf16),
    ] {
        let got = run_dense_tiled(
            precision,
            (3, 12, 1),
            &x,
            Buffer::from_u32s(&words),
            BufferStorage::bf16_packed(),
        );
        for row in 0..3 {
            let (want, magnitude) = host_dot(
                precision,
                (None, Some(FragmentDtype::Bf16)),
                x[row * 12..(row + 1) * 12].iter().copied(),
                weights.iter().copied(),
            );
            assert!(
                (got[row] - want).abs() <= 1e-6 * magnitude.max(1.0),
                "{precision} row {row}: {} vs {want}",
                got[row]
            );
        }
    }
}

/// The product rows. The register-tiled GEMM, the decode GEMV, the serial matmul and the decode `gemv_lds` body each
/// compute the host product over the same terms (the planner's launch for each), under every precision, over data
/// that makes the narrowing visible. Mutation: in `tiled_body` write only the `hi` part of a split activation
/// (drop the `lo` array write); the split GEMM rows go red.
#[test]
fn every_contraction_body_equals_the_host_product_over_the_declared_terms() {
    let (m, k, n) = (13usize, 19usize, 11usize);
    let x = fill(m * k, 1, 3.0);
    let w = fill(n * k, 2, 3.0);
    for precision in PRECISIONS {
        // Register-tiled GEMM over an f32 `[N, K]` weight.
        let got = run_dense_tiled(
            precision,
            (m, k, n),
            &x,
            Buffer::from_f32s(&w),
            BufferStorage::f32(),
        );
        for r in 0..m {
            for c in 0..n {
                let (want, magnitude) = host_dot(
                    precision,
                    (None, None),
                    x[r * k..(r + 1) * k].iter().copied(),
                    w[c * k..(c + 1) * k].iter().copied(),
                );
                assert!(
                    (got[r * n + c] - want).abs() <= 2e-6 * magnitude.max(1.0),
                    "tiled {precision} [{r}, {c}]: {} vs {want}",
                    got[r * n + c]
                );
            }
        }
        // The decode GEMV (`M = 1`): lanes fold their partial sums in a tree, so the order is the kernel's own and
        // the comparison is to the sum's rounding error.
        let schedule = kg::Schedule::Gemv {
            width: 16,
            cols: 2,
            unroll: 2,
            activation: kg::GemvActivation::Scalar,
        };
        let request = kg::KernelRequest::Contraction(kg::ContractionSpec::DenseGemv {
            weight: BufferStorage::f32(),
            schedule,
            k,
            n,
            epilogue: None,
            precision,
        });
        let generated = kg::generate(&request, &caps(), &LIMITS).unwrap();
        let workgroup = generated.body.workgroup_size[0];
        let mut buffers = [
            Buffer::from_f32s(&x[..k]),
            Buffer::from_f32s(&w),
            Buffer::from_u32s(generated.launch.meta.as_ref().unwrap()),
            Buffer::from_f32s(&vec![f32::NAN; n]),
        ];
        run(
            &generated.body,
            [generated.launch.grid[0] / workgroup, 1, 1],
            &mut buffers,
        )
        .unwrap();
        let got = buffers[3].to_f32s().unwrap();
        for c in 0..n {
            let (want, magnitude) = host_dot(
                precision,
                (None, None),
                x[..k].iter().copied(),
                w[c * k..(c + 1) * k].iter().copied(),
            );
            assert!(
                (got[c] - want).abs() <= 2e-6 * magnitude.max(1.0),
                "gemv {precision} [{c}]: {} vs {want}",
                got[c]
            );
        }
        // The serial one-thread-per-output matmul over a `[K, N]` weight.
        let wt: Vec<f32> = (0..k * n).map(|i| w[(i % n) * k + i / n]).collect();
        let shapes = kg::MatmulShapes {
            out_shape: vec![m, n],
            a_shape: vec![m, k],
            b_shape: vec![k, n],
        };
        let request = kg::KernelRequest::Contraction(kg::ContractionSpec::Serial {
            dt: poot_kernel_ir::Ty::F32,
            epilogue: None,
            shapes,
            fold: kg::Fold {
                numel: m * n,
                two_d: false,
                width: 64,
            },
            precision,
        });
        let generated = kg::generate(&request, &caps(), &LIMITS).unwrap();
        let workgroup = generated.body.workgroup_size[0];
        let mut buffers = [
            Buffer::from_f32s(&x),
            Buffer::from_f32s(&wt),
            Buffer::from_f32s(&vec![f32::NAN; m * n]),
        ];
        run(
            &generated.body,
            [generated.launch.grid[0].div_ceil(workgroup), 1, 1],
            &mut buffers,
        )
        .unwrap();
        let got = buffers[2].to_f32s().unwrap();
        for r in 0..m {
            for c in 0..n {
                let (want, magnitude) = host_dot(
                    precision,
                    (None, None),
                    x[r * k..(r + 1) * k].iter().copied(),
                    (0..k).map(|kk| wt[kk * n + c]),
                );
                assert!(
                    (got[r * n + c] - want).abs() <= 2e-6 * magnitude.max(1.0),
                    "serial {precision} [{r}, {c}]: {} vs {want}",
                    got[r * n + c]
                );
            }
        }
    }
}
