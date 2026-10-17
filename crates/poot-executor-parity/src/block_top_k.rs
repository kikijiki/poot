//! Block top-k fixtures (POOT-1143): the indexer's per-query block selection over scores with ties and a
//! block count that leaves a partial last chunk, compared on every executor with the plain selection of
//! `poot_graph_ir::test_support::block_top_k` and with the oracle's primitive definition.

use poot_eval::test_support::StepFixture;
use poot_graph_ir::test_support::block_top_k::{grid_scores, reference_row};
use poot_graph_ir::{BlockTopKAttrs, Builder, Outputs, Slot, TensorType};
use poot_graph_plan::{KernelChoice, Target, production_plan::Submission};
use poot_tensor::{DType, HostTensor};

use crate::{Fixture, store_from_tensors};

/// One case: the op's program over seeded scores, and the packed `[rows, 1, tokens, k + blocks]` words
/// it must produce, row-major.
pub struct BlockTopKCase {
    /// The block top-k program and its constants.
    pub fixture: Fixture,
    /// The words `fixture` must produce.
    pub expected: Vec<i32>,
    /// Whether the oracle's quadratic definition is cheap enough to run on `fixture` too.
    pub oracle: bool,
}

/// The cases:
///
/// - `ties`: five queries at different positions over 100 blocks drawn from 4 distinct scores, ratio 4,
///   `k = 5`: ties straddle the cut at every query with more than five complete blocks, and the first
///   queries have fewer than `k` complete blocks.
/// - `rows_by_tokens`: two rows of three queries over 12 blocks, `k = 4`: the query index is `row * 3 +
///   token`, and the positions differ per query.
/// - `wide`: one query over 32768 blocks (the 128K-position row of the design; a chunk of 128 per lane),
///   `k = 512`, with ties. The device alone: the oracle's pairwise definition over 32768 blocks is refused
///   (and takes minutes at 8192).
/// - `all_equal`: every score equal, so the lowest blocks win.
pub fn cases() -> Vec<BlockTopKCase> {
    let blocks = 100;
    let queries = 5;
    let tied: Vec<Vec<f32>> = (0..queries)
        .map(|q| grid_scores(blocks, 4, 10 + q as u64))
        .collect();
    vec![
        case(
            "block_top_k_ties",
            (queries, 1),
            tied,
            vec![2, 19, 20, 60, 4 * blocks as i32],
            4,
            5,
        ),
        case(
            "block_top_k_rows_by_tokens",
            (2, 3),
            (0..6).map(|q| grid_scores(12, 1 << 20, q)).collect(),
            vec![3, 11, 27, 5, 17, 47],
            2,
            4,
        ),
        case(
            "block_top_k_wide",
            (1, 1),
            vec![grid_scores(32768, 64, 3)],
            vec![4 * 32768 - 1],
            4,
            512,
        ),
        case(
            "block_top_k_all_equal",
            (1, 2),
            vec![vec![0.25; 30], vec![0.25; 30]],
            vec![119, 20],
            4,
            7,
        ),
    ]
}

fn case(
    name: &'static str,
    (rows, tokens): (usize, usize),
    scores: Vec<Vec<f32>>,
    pos: Vec<i32>,
    ratio: usize,
    k: usize,
) -> BlockTopKCase {
    let blocks = scores[0].len();
    let expected = scores
        .iter()
        .zip(&pos)
        .flat_map(|(row, &p)| reference_row(row, p, ratio, k))
        .collect();
    let b = Builder::for_tests();
    let scores_in = b
        .constant("scores", TensorType::f32(vec![rows, 1, tokens, blocks]))
        .unwrap();
    let pos_in = b
        .slot(Slot::Pos, TensorType::new(vec![rows, tokens], DType::I32))
        .unwrap();
    let result = b
        .block_top_k(BlockTopKAttrs { ratio, k }, scores_in, pos_in)
        .unwrap();
    let graph = b.finish(Outputs::new(result.packed)).unwrap();
    let step = || {
        vec![StepFixture {
            key: graph
                .meta(pos_in.id())
                .slot_key()
                .expect("a slot input has a key")
                .clone(),
            tensor: HostTensor::i32(vec![rows, tokens], pos.clone()),
        }]
    };
    let steps = vec![step(), step()];
    BlockTopKCase {
        fixture: Fixture {
            name,
            graph,
            store: store_from_tensors(vec![(
                "scores",
                HostTensor::f32(vec![rows, 1, tokens, blocks], scores.concat()),
            )]),
            steps,
        },
        expected,
        oracle: blocks <= 128,
    }
}

/// `fixture`'s program, compiled for `target`, runs its one selection through the generated block top-k
/// kernel, so a row shows the kernel it ran is the radix select and not only that the words matched.
pub fn assert_runs_the_kernel(fixture: &Fixture, target: Target) {
    let staged = crate::staged(fixture, target, Submission::Replay);
    let kernels = staged
        .stages()
        .flat_map(|(_, _, program)| {
            program.planned().filter(move |(eqn, _)| {
                matches!(
                    program.kernel_choice(eqn),
                    KernelChoice::Generated(request) if request.family() == "block_top_k"
                )
            })
        })
        .count();
    assert_eq!(
        kernels, 1,
        "{}: the selection runs the kernel",
        fixture.name
    );
}

/// Run `case` on `exec`: the program ran the kernel, and the device's words equal the plain selection's at
/// every step. The caller then compares the device with the oracle's definition through `run_parity`.
pub fn assert_device_selects(
    exec: &mut dyn poot_executor::Executor,
    target: Target,
    case: &BlockTopKCase,
) {
    let name = case.fixture.name;
    assert_runs_the_kernel(&case.fixture, target);
    let outputs = crate::run_outputs(exec, target, &case.fixture).unwrap_or_else(|e| panic!("{e}"));
    for (step, got) in outputs.iter().enumerate() {
        let got = got.as_i32().expect("ids and indicator are I32");
        assert_eq!(
            got.len(),
            case.expected.len(),
            "{name} step {step}: result length"
        );
        if let Some(at) = got.iter().zip(&case.expected).position(|(g, w)| g != w) {
            panic!(
                "{name} step {step}: word {at} of the packed result: the device has {}, the plain selection {}",
                got[at], case.expected[at]
            );
        }
    }
}

/// The selection of one query over `blocks` blocks of four, 512 kept, from random scores, as a program for
/// the perf receipt (no expected words: the receipt times the kernel; [`cases`] check it).
pub fn timing_fixture(blocks: usize) -> Fixture {
    let name = match blocks {
        2048 => "block_top_k_2048_blocks",
        8192 => "block_top_k_8192_blocks",
        32768 => "block_top_k_32768_blocks",
        other => panic!("no timing fixture for {other} blocks"),
    };
    case(
        name,
        (1, 1),
        vec![grid_scores(blocks, 1 << 20, 5)],
        vec![(4 * blocks - 1) as i32],
        4,
        512,
    )
    .fixture
}
