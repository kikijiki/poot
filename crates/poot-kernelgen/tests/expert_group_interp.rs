//! The expert grouping kernel on the kernel-IR reference interpreter, without a GPU: the order and its
//! inverse against a stable sort, over routings that break a naive grouping (every row on one expert, experts with
//! no rows, one expert past the table, ids outside it).

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::DeviceCaps;

/// `(order, dest)` the kernel computes for `ids` over `experts` experts.
fn run_group(ids: &[i32], experts: usize) -> (Vec<i32>, Vec<i32>) {
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let generated = kg::generate(
        &kg::KernelRequest::ExpertGroup(kg::ExpertGroupSpec { experts }),
        &caps,
        &limits,
    )
    .unwrap();
    let wg = generated.body.workgroup_size[0];
    let mut buffers = [
        Buffer::from_i32s(ids),
        Buffer::from_i32s(&vec![-1; 2 * ids.len()]),
    ];
    run(
        &generated.body,
        [generated.launch.grid[0].div_ceil(wg), 1, 1],
        &mut buffers,
    )
    .unwrap_or_else(|e| panic!("{ids:?} over {experts} experts: {e}"));
    let out = buffers[1].to_i32s().unwrap();
    let (order, dest) = out.split_at(ids.len());
    (order.to_vec(), dest.to_vec())
}

/// The stable sort by bucket (the id when it is an expert, else `experts`) the kernel must reproduce.
fn reference(ids: &[i32], experts: usize) -> (Vec<i32>, Vec<i32>) {
    let bucket = |id: i32| {
        usize::try_from(id)
            .ok()
            .filter(|&e| e < experts)
            .unwrap_or(experts)
    };
    let mut order: Vec<usize> = (0..ids.len()).collect();
    order.sort_by_key(|&row| bucket(ids[row]));
    let mut dest = vec![0; ids.len()];
    for (position, &row) in order.iter().enumerate() {
        dest[row] = position as i32;
    }
    (order.into_iter().map(|row| row as i32).collect(), dest)
}

fn assert_grouped(label: &str, ids: &[i32], experts: usize) {
    let (order, dest) = run_group(ids, experts);
    assert_eq!((order.clone(), dest), reference(ids, experts), "{label}");
    let buckets: Vec<usize> = order
        .iter()
        .map(|&row| {
            usize::try_from(ids[row as usize])
                .ok()
                .filter(|&e| e < experts)
                .unwrap_or(experts)
        })
        .collect();
    assert!(
        buckets.is_sorted(),
        "{label}: sorted positions hold ascending experts"
    );
}

/// The kernel's order and inverse equal a stable sort on every adversarial routing.
///
/// Mutation (observed red): compare `less` with `Le` instead of `Lt` when counting a bucket's start; every row of a
/// bucket is counted into its own start, the placements land past their slots and the order and inverse differ
/// from the stable sort (`one expert gets every row`: every position stays `-1` and the inverse starts at 37).
#[test]
fn the_grouping_is_the_stable_sort_by_expert_and_its_inverse() {
    let cycle = |rows: usize, experts: i32| -> Vec<i32> {
        (0..rows as i32).map(|r| (r * 5 + 3) % experts).collect()
    };
    assert_grouped("one expert gets every row", &[2; 37], 4);
    assert_grouped("one row", &[1], 4);
    assert_grouped("experts without rows", &[3, 3, 0, 3, 0, 0, 3], 6);
    assert_grouped("last expert only", &[5, 5, 5], 6);
    assert_grouped("ties keep the row order", &[1, 0, 1, 0, 1, 0, 1, 0], 2);
    assert_grouped("a mixed routing past one workgroup", &cycle(301, 7), 7);
    assert_grouped("more experts than rows", &cycle(5, 40), 40);
}

/// A row whose id is outside `0..experts` sorts after every expert, in row order, and is never dropped or given
/// another expert's position: both outputs stay permutations of `0..M`.
///
/// Mutation (observed red): drop the negative test from `bad` (only `id >= experts` is bucketed last); a negative id
/// keys below expert 0, its placement lands outside `0..M` and the result is not a permutation (`left: ([-1, -1,
/// -1, 2, 7, ...])` against the stable sort).
#[test]
fn an_invalid_expert_id_sorts_last_and_is_never_dropped() {
    let ids = [2, -1, 0, 4, i32::MIN, 1, i32::MAX, 0, 1_000_000_000, 3, -7];
    assert_grouped("bad ids", &ids, 4);
    let (order, dest) = run_group(&ids, 4);
    assert_eq!(&order[order.len() - 6..], &[1, 3, 4, 6, 8, 10]);
    let mut seen = dest.clone();
    seen.sort_unstable();
    assert_eq!(seen, (0..ids.len() as i32).collect::<Vec<_>>());
    assert_grouped("only bad ids", &[-1, 9, -4], 4);
}

const EXPERTS: usize = 4;
const K: usize = 6;
const N: usize = 5;

/// The register-tiled indexed contraction (f32 weight, `[E, K, N]`) of `x [rows, K]` through `ids`, on the
/// interpreter: the cost model's tiling for the shape, its expert walk over each tile's rows.
fn run_indexed(x: &[f32], ids: &[i32]) -> Vec<f32> {
    let caps = DeviceCaps::wgpu_rdna3_igpu();
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    };
    let rows = ids.len();
    let schedule = kg::RegisterTiling::choose(
        kg::TiledShape {
            blocks: 1,
            m_per_block: rows,
            out_per_block: N,
            k: K,
            layout: kg::WeightLayout::Kn,
            segments: EXPERTS,
        },
        &caps,
    )
    .schedule();
    let threads = schedule.grid_threads(1, rows, N) as u32;
    let kg::Schedule::RegisterTiled { lanes, .. } = schedule else {
        panic!("an M > 1 tiling is register-tiled: {schedule:?}");
    };
    let tiles = threads / lanes.get().pow(2);
    let request =
        kg::KernelRequest::Contraction(kg::ContractionSpec::Indexed(kg::IndexedContraction {
            schedule,
            layout: kg::WeightLayout::Kn,
            weight: poot_target::BufferStorage::f32(),
            epilogue: None,
            grid: [threads, 1, 1],
            meta: vec![K as u32, N as u32, rows as u32, tiles, 0, EXPERTS as u32],
        }));
    let generated = kg::generate(&request, &caps, &limits).unwrap();
    let wg = generated.body.workgroup_size[0];
    let weights: Vec<f32> = (0..EXPERTS * K * N)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.25 - 1.25)
        .collect();
    let mut buffers = [
        Buffer::from_f32s(x),
        Buffer::from_f32s(&weights),
        Buffer::from_i32s(ids),
        Buffer::from_u32s(generated.launch.meta.as_ref().unwrap()),
        Buffer::from_f32s(&vec![f32::MIN; rows * N]),
    ];
    run(
        &generated.body,
        [generated.launch.grid[0] / wg, generated.launch.grid[1], 1],
        &mut buffers,
    )
    .unwrap();
    buffers[4].to_f32s().unwrap()
}

/// Contracting the rows in expert order and putting them back gives every row the bits of the direct contraction,
/// and a row with an invalid id is NaN in both: grouping neither drops a bad id nor gives it another expert's
/// result.
///
/// Mutation (observed red, with the two above): the grouped result of a bad-id row is read from a position the broken
/// grouping never wrote, so the bit comparison fails.
#[test]
fn a_contraction_through_the_grouping_is_bit_identical_and_an_invalid_id_stays_a_nan_row() {
    let ids: Vec<i32> = (0..30)
        .map(|r| match r {
            3 => -1,
            11 => EXPERTS as i32,
            25 => i32::MAX,
            _ => (r * 5 + 1) % EXPERTS as i32,
        })
        .collect();
    let x: Vec<f32> = (0..ids.len() * K)
        .map(|i| ((i * 7) % 13) as f32 - 6.0)
        .collect();
    let direct = run_indexed(&x, &ids);

    let (order, dest) = run_group(&ids, EXPERTS);
    let sorted_x: Vec<f32> = order
        .iter()
        .flat_map(|&row| x[row as usize * K..(row as usize + 1) * K].iter().copied())
        .collect();
    let sorted_ids: Vec<i32> = order.iter().map(|&row| ids[row as usize]).collect();
    let sorted = run_indexed(&sorted_x, &sorted_ids);
    for (row, &id) in ids.iter().enumerate() {
        let at = dest[row] as usize * N;
        for col in 0..N {
            let (d, g) = (direct[row * N + col], sorted[at + col]);
            assert_eq!(d.to_bits(), g.to_bits(), "row {row} (id {id}) col {col}");
            let valid = (0..EXPERTS as i32).contains(&id);
            assert_eq!(d.is_nan(), !valid, "row {row} (id {id}) col {col}: {d}");
        }
    }
}
