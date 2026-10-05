//! Verifies that `fused_row_parallel_views`' 2-D row layout (`row = GroupY * x_groups + GroupX`) computes the
//! row softmax on the kernel-IR reference interpreter, without a GPU. Card 1006: a row count above the
//! target's X grid cap launches as a folded `[x_groups, y_groups]` grid.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernel_ir::{BinOp, MathOp};
use poot_kernelgen as kg;
use poot_test_util::assert_close;

fn softmax_kernel(n_cols: usize) -> kg::RowKernel {
    kg::RowKernel {
        n_leaves: 1,
        n_cols,
        steps: vec![
            kg::RowOp::Reduce {
                op: kg::RowReduce::Max,
                input: kg::FusedInput::Leaf(0),
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Sub),
                inputs: vec![kg::FusedInput::Leaf(0), kg::FusedInput::Step(0)],
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Math(MathOp::Exp),
                inputs: vec![kg::FusedInput::Step(1)],
            },
            kg::RowOp::Reduce {
                op: kg::RowReduce::Sum,
                input: kg::FusedInput::Step(2),
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Div),
                inputs: vec![kg::FusedInput::Step(2), kg::FusedInput::Step(3)],
            },
        ],
        output: 1 + 4,
    }
}

#[test]
fn fused_row_parallel_folded_grid_covers_every_row() {
    // 7 rows over a 3-wide X grid fold to [3, 3, 1] (9 workgroups, two of them padding): every row is
    // computed exactly once, the padding workgroups write nothing.
    let (num_rows, n_cols, width, x_groups, y_groups) = (7usize, 10usize, 4usize, 3usize, 3u32);
    let body = kg::fused_row_parallel_views(
        "frp_folded",
        &[num_rows, n_cols],
        &[&[num_rows, n_cols][..]],
        &[kg::Layout::contiguous(&[num_rows, n_cols])],
        &softmax_kernel(n_cols),
        width,
        x_groups,
    )
    .expect("fused_row_parallel precondition");
    let scores: Vec<f32> = (0..num_rows * n_cols)
        .map(|i| (i as f32 * 0.7).cos() * 3.0)
        .collect();
    let mut buffers = [
        Buffer::from_f32s(&scores),
        Buffer::from_f32s(&vec![f32::NAN; num_rows * n_cols]),
    ];
    run(&body, [x_groups as u32, y_groups, 1], &mut buffers).expect("interpret");
    let got = buffers[1].to_f32s().unwrap();
    let want: Vec<f32> = scores
        .chunks(n_cols)
        .flat_map(|row| {
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = row.iter().map(|x| (x - m).exp()).collect();
            let sum: f32 = e.iter().sum();
            e.into_iter().map(move |x| x / sum)
        })
        .collect();
    assert_close(&got, &want, 1e-5);
}
