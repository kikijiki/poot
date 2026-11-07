//! The block-aligned expert grouping (POOT-1451) on the kernel-IR reference interpreter, without a GPU: the grouping
//! kernel against an independent layout over routings that break a naive grouping (every row on one expert, experts
//! with no rows, one expert past the table, ids outside it, more experts than rows), and the tiled indexed
//! contraction over the grouping against each row's own expert.

use poot_graph_ir::ExpertGrouping;
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;

fn limits() -> kg::BodyLimits {
    kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 20).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 16).unwrap(),
    }
}

/// The grouping the kernel computes for `ids` over `experts` experts in tiles of `block`.
fn run_group(ids: &[i32], experts: usize, block: usize) -> Vec<i32> {
    let generated = kg::generate(
        &kg::KernelRequest::ExpertGroup(kg::ExpertGroupSpec { experts, block }),
        &poot_test_util::device_caps::wgpu_rdna3_igpu(),
        &limits(),
    )
    .unwrap();
    let wg = generated.body.workgroup_size[0];
    let words = ExpertGrouping {
        rows: ids.len(),
        experts,
        block,
    }
    .len();
    let mut buffers = [Buffer::from_i32s(ids), Buffer::from_i32s(&vec![-7; words])];
    run(
        &generated.body,
        [generated.launch.grid[0].div_ceil(wg), 1, 1],
        &mut buffers,
    )
    .unwrap_or_else(|e| panic!("{ids:?} over {experts} experts, block {block}: {e}"));
    buffers[1].to_i32s().unwrap()
}

/// The layout the kernel must write, built bucket by bucket from the rows of each.
fn reference(ids: &[i32], experts: usize, block: usize) -> Vec<i32> {
    let layout = ExpertGrouping {
        rows: ids.len(),
        experts,
        block,
    };
    let bucket = |id: i32| {
        usize::try_from(id)
            .ok()
            .filter(|&e| e < experts)
            .unwrap_or(experts)
    };
    let sentinel = ids.len() as i32;
    let (mut order, mut tiles) = (Vec::new(), Vec::new());
    for b in 0..=experts {
        let rows: Vec<i32> = (0..ids.len())
            .filter(|&r| bucket(ids[r]) == b)
            .map(|r| r as i32)
            .collect();
        for chunk in rows.chunks(block) {
            order.extend_from_slice(chunk);
            order.resize(order.len() + block - chunk.len(), sentinel);
            tiles.push(b as i32);
        }
    }
    order.resize(layout.positions(), sentinel);
    tiles.resize(layout.tiles(), experts as i32);
    order.extend(tiles);
    order
}

fn assert_grouped(label: &str, ids: &[i32], experts: usize) {
    for block in [1, 3, 4, 16, 64] {
        assert_eq!(
            run_group(ids, experts, block),
            reference(ids, experts, block),
            "{label}, block {block}"
        );
    }
}

/// The kernel's grouping is the layout on every adversarial routing and block: each bucket's rows in row order
/// from its first tile, its run padded with the sentinel, each tile naming its bucket, and the tail past the padded
/// total all sentinels and empty tiles.
///
/// Mutation (observed red): round a bucket's padded size down instead of up in `expert_group` (`padded`'s `block -
/// 1` dropped); the next bucket starts inside the previous one's last tile and the layouts differ (`one expert gets
/// every row, block 4`).
#[test]
fn the_grouping_pads_each_experts_rows_to_whole_tiles() {
    let cycle = |rows: usize, experts: i32| -> Vec<i32> {
        (0..rows as i32).map(|r| (r * 5 + 3) % experts).collect()
    };
    assert_grouped("one expert gets every row", &[2; 37], 4);
    assert_grouped("one row", &[1], 4);
    assert_grouped("experts without rows", &[3, 3, 0, 3, 0, 0, 3], 6);
    assert_grouped("last expert only", &[5, 5, 5], 6);
    assert_grouped("ties keep the row order", &[1, 0, 1, 0, 1, 0, 1, 0], 2);
    assert_grouped("a mixed routing over the lanes", &cycle(301, 7), 7);
    assert_grouped("more experts than rows", &cycle(5, 40), 40);
    assert_grouped("more buckets than lanes", &cycle(700, 300), 300);
}

/// A row whose id is outside `0..experts` lands in the last bucket, whose tiles name no expert, in row order: it is
/// never dropped or given another expert's tile.
///
/// Mutation (observed red): drop the negative test from `bad` (only `id >= experts` is bucketed last); a negative
/// id keys below expert 0, no lane's bucket claims it, and the layouts differ (`bad ids, block 1`).
#[test]
fn an_invalid_expert_id_lands_in_the_last_bucket_and_is_never_dropped() {
    let ids = [2, -1, 0, 4, i32::MIN, 1, i32::MAX, 0, 1_000_000_000, 3, -7];
    assert_grouped("bad ids", &ids, 4);
    let grouping = run_group(&ids, 4, 4);
    let layout = ExpertGrouping {
        rows: ids.len(),
        experts: 4,
        block: 4,
    };
    let invalid: Vec<i32> = (0..layout.tiles())
        .filter(|&t| grouping[layout.positions() + t] == 4)
        .flat_map(|t| grouping[t * 4..(t + 1) * 4].to_vec())
        .filter(|&row| row < ids.len() as i32)
        .collect();
    assert_eq!(invalid, [1, 3, 4, 6, 8, 10]);
    assert_grouped("only bad ids", &[-1, 9, -4], 4);
}

const EXPERTS: usize = 4;
const K: usize = 6;
const N: usize = 5;

fn weights() -> Vec<f32> {
    (0..EXPERTS * K * N)
        .map(|i| ((i * 7 + 3) % 11) as f32 * 0.25 - 1.25)
        .collect()
}

/// The tiled indexed contraction (f32 weight, `[E, K, N]`) of `x [rows, K]` over its rows grouped from `ids`, on
/// the interpreter: the block the cost model picks for the shape, its tiling, and the grouping kernel's grouping.
fn run_grouped(x: &[f32], ids: &[i32]) -> Vec<f32> {
    let caps = poot_test_util::device_caps::wgpu_rdna3_igpu();
    let rows = ids.len();
    let shape = |block| kg::TiledShape {
        blocks: 1,
        m_per_block: rows,
        out_per_block: N,
        k: K,
        layout: kg::WeightLayout::Kn,
        rows: kg::TiledRows::Grouped {
            groups: EXPERTS + 1,
            block,
        },
        terms: kg::StagedTerms::ONE,
        weight: kg::StagedWeight::Dense { bits: 32 },
        activation_bits: 32,
    };
    let block = kg::RegisterTiling::grouped_block(&[shape(0)], &caps).expect("a block");
    let tiling = kg::RegisterTiling::choose(shape(block), &caps);
    assert_eq!(tiling.output_tile().0, block, "{tiling:?}");
    let schedule = tiling.schedule();
    let positions = shape(block).grid_rows();
    let threads = schedule.grid_threads(1, positions, N) as u32;
    let kg::Schedule::RegisterTiled { lanes, .. } = schedule else {
        panic!("the interpreter runs the scalar tiling: {schedule:?}");
    };
    let tiles = threads / lanes.get().pow(2);
    let request =
        kg::KernelRequest::Contraction(kg::ContractionSpec::Indexed(kg::IndexedContraction {
            schedule,
            layout: kg::WeightLayout::Kn,
            weight: poot_target::BufferStorage::f32(),
            epilogue: None,
            grid: [threads, 1, 1],
            meta: vec![
                K as u32,
                N as u32,
                positions as u32,
                tiles,
                0,
                EXPERTS as u32,
            ],
            precision: poot_graph_ir::OperandPrecision::Exact,
        }));
    let generated = kg::generate(&request, &caps, &limits()).unwrap();
    let wg = generated.body.workgroup_size[0];
    let mut buffers = [
        Buffer::from_f32s(x),
        Buffer::from_f32s(&weights()),
        Buffer::from_i32s(&run_group(ids, EXPERTS, block)),
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

/// Every row of the grouped contraction is its own expert's product, written in row order, and a row with an
/// invalid id is NaN: the grouping neither drops a row, nor stores a padding position, nor gives a row another
/// expert's weight.
///
/// Mutation (observed red): store each grouped position at its position instead of the row it holds (`row` left
/// unmapped in `tiled_body`'s store); rows land out of order, padding overwrites rows and `row 0` differs.
#[test]
fn a_grouped_contraction_writes_each_rows_own_product_and_an_invalid_id_stays_a_nan_row() {
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
    let got = run_grouped(&x, &ids);
    let w = weights();
    for (row, &id) in ids.iter().enumerate() {
        for col in 0..N {
            let value = got[row * N + col];
            match usize::try_from(id).ok().filter(|&e| e < EXPERTS) {
                Some(expert) => {
                    let want: f32 = (0..K)
                        .map(|k| x[row * K + k] * w[(expert * K + k) * N + col])
                        .sum();
                    assert_eq!(value, want, "row {row} (id {id}) col {col}");
                }
                None => assert!(value.is_nan(), "row {row} (id {id}) col {col}: {value}"),
            }
        }
    }
}
