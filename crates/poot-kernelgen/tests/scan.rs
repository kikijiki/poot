//! The streaming scan kernel (POOT-634) on the kernel-IR reference interpreter, no GPU: both recurrences
//! against the plain token-by-token reference, a sequence split into launches that hand the carried state
//! through the packed result, and a body whose size does not depend on the sequence length. Every row builds
//! its body through `generate`, the one seam the planner uses.
//!
//! Mutations (applied once to `scan.rs`, observed red, restored): drop the final-state publish; load a later
//! launch's carried state as zero instead of from the packed result; unroll the sequence loop.

use std::num::NonZeroUsize;

use poot_kernel_ir::Terminator;
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen::{
    BodyLimits, KernelGenError, KernelRequest, ScanKernel, ScanPhase, ScanSpec, generate,
};
use poot_target::DeviceCaps;
use poot_test_util::assert_close;
use poot_test_util::scan::{ScanFamily, ScanFixture};

fn limits(max_instructions: usize) -> BodyLimits {
    BodyLimits {
        max_instructions: NonZeroUsize::new(max_instructions).unwrap(),
        max_locals: NonZeroUsize::new(1 << 18).unwrap(),
    }
}

/// A request for `state_cols` state columns per workgroup, each shared by `row_split` lanes.
fn spec(fixture: &ScanFixture, state_cols: usize, row_split: usize, phase: ScanPhase) -> ScanSpec {
    ScanSpec {
        kernel: match fixture.family {
            ScanFamily::DeltaRule => ScanKernel::DeltaRule,
            ScanFamily::Ssd => ScanKernel::Ssd,
        },
        heads: fixture.heads,
        len: fixture.len,
        rows: fixture.state_rows_of(),
        width: fixture.width,
        cols: state_cols * row_split,
        row_split,
        phase,
    }
}

/// Run the launches `phases` in order over one packed result buffer (NaN to begin with, so a row nobody
/// wrote fails), and return `(y, s_out)`.
fn interpret(
    fixture: &ScanFixture,
    state_cols: usize,
    row_split: usize,
    phases: &[ScanPhase],
) -> (Vec<f32>, Vec<f32>) {
    let inputs = fixture.inputs(17);
    let rows = fixture.state_rows_of();
    let packed_len = fixture.heads * (fixture.len + rows) * fixture.width;
    let mut buffers: Vec<Buffer> = inputs
        .operands
        .iter()
        .map(|o| Buffer::from_f32s(o))
        .collect();
    buffers.push(Buffer::from_f32s(&vec![f32::NAN; packed_len]));
    for &phase in phases {
        let spec = spec(fixture, state_cols, row_split, phase);
        let lanes = state_cols * row_split;
        let generated = generate(
            &KernelRequest::Scan(spec.clone()),
            &DeviceCaps::wgpu_rdna3_igpu(),
            &limits(1 << 16),
        )
        .expect("the scan request generates");
        assert_eq!(generated.body.workgroup_size, [lanes as u32, 1, 1]);
        let grid = generated.launch.grid;
        run(
            &generated.body,
            [grid[0] / lanes as u32, grid[1], grid[2]],
            &mut buffers,
        )
        .unwrap();
    }
    let packed = buffers.last().unwrap().to_f32s().unwrap();
    let (mut y, mut s) = (Vec::new(), Vec::new());
    for head in packed.chunks(((fixture.len + rows) * fixture.width).max(1)) {
        let (head_y, head_s) = head.split_at(fixture.len * fixture.width);
        y.extend_from_slice(head_y);
        s.extend_from_slice(head_s);
    }
    (y, s)
}

fn whole(fixture: &ScanFixture) -> [ScanPhase; 1] {
    [ScanPhase {
        start: 0,
        count: fixture.len,
    }]
}

fn fixtures() -> [ScanFixture; 6] {
    [
        ScanFixture::delta_rule(3, 9, 6, 4),
        ScanFixture::delta_rule(2, 5, 8, 4),
        ScanFixture::ssd(3, 9, 6, 5, 4),
        ScanFixture::ssd(2, 5, 8, 3, 4),
        // Eight state rows: every split of 1, 2, 4 and 8 divides them. `width` 6 leaves a partial column
        // block for the 4- and 16-column workgroups.
        ScanFixture::delta_rule(2, 7, 8, 4),
        ScanFixture::ssd(2, 3, 6, 8, 4),
    ]
}

/// The row splits that divide `fixture`'s state rows.
fn splits(fixture: &ScanFixture) -> impl Iterator<Item = usize> {
    let rows = fixture.state_rows_of();
    [1usize, 2, 4, 8]
        .into_iter()
        .filter(move |s| rows.is_multiple_of(*s))
}

/// Every output and carried-state element of one launch over the whole sequence matches the independent
/// recurrence, with workgroups narrower than the head width (a partial last column block) and wider.
#[test]
fn one_launch_matches_the_reference() {
    for fixture in fixtures() {
        let (want_y, want_s) = fixture.reference(&fixture.inputs(17));
        for cols in [1, 4, 16] {
            for split in splits(&fixture) {
                let (y, s) = interpret(&fixture, cols.min(fixture.width), split, &whole(&fixture));
                assert_close(&y, &want_y, 2e-5);
                assert_close(&s, &want_s, 2e-5);
            }
        }
    }
}

/// A sequence split into launches of a few tokens each, the carried state handed on through the packed
/// result, matches the reference, whatever the split (ragged last launch, one token per launch).
#[test]
fn launches_hand_the_carried_state_through_the_packed_result() {
    for fixture in fixtures() {
        let (want_y, want_s) = fixture.reference(&fixture.inputs(17));
        for per_launch in [1, 2, 4] {
            let phases: Vec<_> = (0..fixture.len)
                .step_by(per_launch)
                .map(|start| ScanPhase {
                    start,
                    count: per_launch.min(fixture.len - start),
                })
                .collect();
            for split in splits(&fixture) {
                let (y, s) = interpret(&fixture, fixture.width.min(4), split, &phases);
                assert_close(&y, &want_y, 2e-5);
                assert_close(&s, &want_s, 2e-5);
            }
        }
    }
}

/// The body is the same size at every sequence length, within a declared limit; a limit below it refuses
/// with a typed error rather than returning an oversized body. Held for every split: a shared-column body is
/// admitted by the same budget as an unsplit one, never past it.
#[test]
fn the_body_does_not_grow_with_the_sequence_and_obeys_its_limit() {
    for base in fixtures() {
        for split in splits(&base) {
            let size = |len: usize| {
                let fixture = ScanFixture { len, ..base };
                let request = KernelRequest::Scan(spec(
                    &fixture,
                    4,
                    split,
                    ScanPhase {
                        start: 0,
                        count: len,
                    },
                ));
                let generated = generate(&request, &DeviceCaps::wgpu_rdna3_igpu(), &limits(768))
                    .unwrap_or_else(|e| {
                        panic!(
                            "{:?} split {split} L={len}: the body must fit 768 instructions: {e}",
                            base.family
                        )
                    });
                generated
                    .body
                    .blocks
                    .iter()
                    .map(|b| b.statements.len() + 1)
                    .sum::<usize>()
            };
            let sizes = [size(8), size(512), size(65_536)];
            assert!(
                sizes.windows(2).all(|w| w[0] == w[1]),
                "{:?} split {split}: body instructions at L=8, 512, 65536 are {sizes:?}",
                base.family
            );
            let tiny = generate(
                &KernelRequest::Scan(spec(&base, 4, split, whole(&base)[0])),
                &DeviceCaps::wgpu_rdna3_igpu(),
                &limits(sizes[0] - 1),
            );
            assert!(
                matches!(tiny, Err(KernelGenError::BodyLimit { .. })),
                "{:?} split {split}: {tiny:?}",
                base.family
            );
        }
    }
}

/// A launch past the end of the sequence, or an empty one, is refused by the generator, not built.
#[test]
fn a_launch_outside_the_sequence_is_refused() {
    let fixture = ScanFixture::delta_rule(1, 4, 4, 4);
    for phase in [
        ScanPhase { start: 2, count: 3 },
        ScanPhase { start: 0, count: 0 },
    ] {
        let refused = generate(
            &KernelRequest::Scan(spec(&fixture, 4, 1, phase)),
            &DeviceCaps::wgpu_rdna3_igpu(),
            &limits(1 << 16),
        );
        assert!(refused.is_err(), "{phase:?}: {refused:?}");
    }
}

/// What a schedule costs in synchronization: an unsplit launch has no barrier and one workgroup array (the
/// state), a split one two barriers per delta-rule token and one per SSD token, and the array of partial
/// sums. A column shared by lanes without its barrier would read other lanes' stale partials.
#[test]
fn a_split_adds_its_barriers_and_an_unsplit_launch_has_none() {
    for fixture in fixtures() {
        for split in splits(&fixture) {
            let request = KernelRequest::Scan(spec(&fixture, 4, split, whole(&fixture)[0]));
            let body = generate(&request, &DeviceCaps::wgpu_rdna3_igpu(), &limits(1 << 16))
                .expect("the scan request generates")
                .body;
            let barriers = body
                .blocks
                .iter()
                .filter(|block| matches!(block.terminator, Terminator::Barrier { .. }))
                .count();
            let (want_barriers, want_arrays) = match (split, fixture.family) {
                (1, _) => (0, 1),
                (_, ScanFamily::DeltaRule) => (2, 2),
                (_, ScanFamily::Ssd) => (1, 2),
            };
            assert_eq!(
                (barriers, body.workgroup_locals.len()),
                (want_barriers, want_arrays),
                "{:?} split {split}: barriers and workgroup arrays",
                fixture.family
            );
        }
    }
}
