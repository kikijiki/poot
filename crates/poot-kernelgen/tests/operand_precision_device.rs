//! POOT-1430: the tier-1 storage row on a real device. The narrowed value a generated body stages is bit-identical
//! to the oracle's round to nearest even (`poot_eval::round_to_fragment`) when the body runs on the GPU, not only on
//! the kernel-IR interpreter: a device whose f32 -> f16 conversion truncates, or whose compiler folds the
//! narrow-and-widen pair away, fails here by name whatever the data.
//!
//! Skips (reported) if no Vulkan adapter opens.

use poot_codegen::compile_kernel;
use poot_eval::round_to_fragment;
use poot_graph_ir::{FragmentDtype, OperandPrecision};
use poot_kernelgen as kg;
use poot_target::{Backend, BufferStorage, DeviceCaps};
use poot_wgpu_runtime::{Context, KernelBuffer};

/// One `K = 16` staging round: a probe row `[x, 0, ..]` fills it, so no tail slot is masked.
const STAGE: usize = 16;

fn ctx() -> Option<Context> {
    match Context::new() {
        Ok(c) => Some(c),
        Err(e) => {
            poot_test_util::skip_report("poot-kernelgen", &format_args!("no wgpu context: {e}"));
            None
        }
    }
}

/// Finite values around every edge of the f16 narrowing: ties and their neighbours, the range edge, subnormals,
/// zeros, and a spread of random bit patterns.
fn probes() -> Vec<f32> {
    let mut values = vec![
        0.0,
        -0.0,
        1.0,
        -1.5,
        65504.0,
        65519.0,
        2f32.powi(-24),
        1.5 * 2f32.powi(-25),
        2f32.powi(-14),
    ];
    for odd in [1u32, 3, 5, 0x7f] {
        values.push(f32::from_bits(0x3f80_0000 | (odd << 13) | 0x1000)); // an f16 tie
        values.push(f32::from_bits(0x3f80_0000 | (odd << 13) | 0x1001)); // just above it
        values.push(f32::from_bits(0x3f80_0000 | (odd << 13) | 0x0fff)); // just below it
    }
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    values.extend((0..180).map(|_| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let exponent = 100 + (state % 40) as u32; // |x| in about 2^-27 .. 2^13: inside and below the f16 range
        f32::from_bits(((state >> 20) as u32 & 0x807f_ffff) | (exponent << 23))
    }));
    values
}

/// `precision` staged by the register-tiled GEMM on wgpu: `out[i] = narrow(x_i) * 1` for the probe rows `[x, 0, ..]`
/// against the weight row `[1, 0, ..]`.
fn staged_on_wgpu(ctx: &Context, precision: OperandPrecision, probes: &[f32]) -> Vec<f32> {
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let m = probes.len();
    let schedule = kg::RegisterTiling::choose(
        kg::TiledShape {
            blocks: 1,
            m_per_block: m,
            out_per_block: 1,
            k: STAGE,
            layout: kg::WeightLayout::Nk,
            segments: 1,
            // The scalar tiling: this row holds the staging to the oracle; the fragment tiling has its own rows
            // (`fragment_contraction.rs`).
            terms: kg::StagedTerms::of(precision, None, None).scalar_only(),
        },
        &caps,
    )
    .schedule();
    let (tile_rows, tile_cols) = schedule.output_tile().unwrap();
    let kg::Schedule::RegisterTiled { lanes, .. } = schedule else {
        unreachable!("choose returns a register tiling")
    };
    let tiles = m.div_ceil(tile_rows) * 1usize.div_ceil(tile_cols);
    let request = kg::KernelRequest::Contraction(kg::ContractionSpec::DenseTiled(
        kg::DenseTiledContraction {
            weight: kg::DenseTiledWeight::Nk(BufferStorage::f32()),
            schedule,
            epilogue: None,
            grid: [tiles as u32 * lanes.get() * lanes.get(), 1, 1],
            meta: vec![STAGE as u32, 1, m as u32, tiles as u32, 0],
            precision,
        },
    ));
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    let generated = kg::generate(&request, &caps, &limits).unwrap();
    let dir = std::env::temp_dir()
        .join("poot-kernelgen-test")
        .join("operand_precision_device");
    std::fs::create_dir_all(&dir).unwrap();
    let out = poot_codegen::artifact_path(&dir, "staged", poot_codegen::Target::SpirvVulkan);
    let artifact = compile_kernel(&generated.body, Backend::SpirvVulkan, &out).expect("compile");
    let kernel = poot_codegen::kernel_handle(&generated.body, &artifact).unwrap();

    let activation: Vec<f32> = (0..m * STAGE)
        .map(|i| {
            if i % STAGE == 0 {
                probes[i / STAGE]
            } else {
                0.0
            }
        })
        .collect();
    let weight: Vec<f32> = (0..STAGE).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let mut buffers = [
        KernelBuffer::read_only_f32(&activation),
        KernelBuffer::read_only_f32(&weight),
        KernelBuffer::read_only_u32(generated.launch.meta.as_deref().expect("metadata")),
        KernelBuffer::write_f32(m),
    ];
    ctx.dispatch(
        "staged",
        &kernel,
        generated.body.workgroup_size,
        generated.launch.grid,
        &mut buffers,
    )
    .unwrap();
    buffers[3].as_f32().to_vec()
}

/// The staged f16 value of every probe is the oracle's, bit for bit, on wgpu (RADV), under `Narrow(f16)`; under
/// `Split(f16)` the product is `hi + lo`, also the oracle's. Mutation: in `crate::precision::operand_terms` skip the
/// round trip for f16 (return `x`); every probe off the f16 grid differs and this row goes red.
#[test]
fn staged_f16_values_on_wgpu_are_bit_identical_to_the_oracles_round_to_nearest_even() {
    let Some(ctx) = ctx() else { return };
    let probes = probes();
    let f16 = FragmentDtype::F16;
    let got = staged_on_wgpu(&ctx, OperandPrecision::Narrow(f16), &probes);
    for (i, &x) in probes.iter().enumerate() {
        // The kernel accumulates from `+0.0`, so a `-0.0` product stores as `+0.0`.
        let want = 0.0f32 + round_to_fragment(f16, x) * 1.0;
        assert_eq!(
            got[i].to_bits(),
            want.to_bits(),
            "narrow(f16) {x:e} ({:#010x}): staged {:e}, oracle {:e}",
            x.to_bits(),
            got[i],
            want
        );
    }
    let got = staged_on_wgpu(&ctx, OperandPrecision::Split(f16), &probes);
    for (i, &x) in probes.iter().enumerate() {
        let hi = round_to_fragment(f16, x);
        let lo = round_to_fragment(f16, x - hi);
        // The weight is `1.0` (a split f32 operand: `hi` 1.0, `lo` 0): the products over the four term pairs.
        let want = (((0.0f32 + hi * 1.0) + hi * 0.0) + lo * 1.0) + lo * 0.0;
        assert_eq!(
            got[i].to_bits(),
            want.to_bits(),
            "split(f16) {x:e} ({:#010x}): staged {:e}, oracle {:e}",
            x.to_bits(),
            got[i],
            want
        );
    }
}
