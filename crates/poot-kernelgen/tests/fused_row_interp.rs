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
        leaf_storage: vec![poot_target::BufferStorage::f32()],
        n_cols,
        steps: vec![
            kg::RowOp::Reduce {
                op: kg::RowReduce::Max,
                input: kg::FusedInput::Leaf(kg::LeafId::new(0)),
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Sub),
                inputs: vec![
                    kg::FusedInput::Leaf(kg::LeafId::new(0)),
                    kg::FusedInput::Step(kg::StepId::new(0)),
                ],
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Math(MathOp::Exp),
                inputs: vec![kg::FusedInput::Step(kg::StepId::new(1))],
            },
            kg::RowOp::Reduce {
                op: kg::RowReduce::Sum,
                input: kg::FusedInput::Step(kg::StepId::new(2)),
            },
            kg::RowOp::Pointwise {
                op: kg::FusedScalarOp::Binary(BinOp::Div),
                inputs: vec![
                    kg::FusedInput::Step(kg::StepId::new(2)),
                    kg::FusedInput::Step(kg::StepId::new(3)),
                ],
            },
        ],
        output: kg::RowValue::Step(kg::StepId::new(4)),
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

/// One-pass RMSNorm whose scale (leaf 1) is a BF16 weight in the packed lane, widened in registers by a `Widen`
/// step: `out = x / sqrt(mean(x^2) + eps) * widen(w)`.
fn rmsnorm_with_packed_scale(n_cols: usize, eps: f32) -> kg::RowKernel {
    use kg::{FusedInput as In, FusedScalarOp as Op, RowOp};
    let leaves = 2;
    kg::RowKernel {
        n_leaves: leaves,
        leaf_storage: vec![
            poot_target::BufferStorage::f32(),
            poot_target::BufferStorage::bf16_packed(),
        ],
        n_cols,
        steps: vec![
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Mul),
                inputs: vec![In::Leaf(kg::LeafId::new(0)), In::Leaf(kg::LeafId::new(0))],
            },
            RowOp::Reduce {
                op: kg::RowReduce::Sum,
                input: In::Step(kg::StepId::new(0)),
            },
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Mul),
                inputs: vec![In::Step(kg::StepId::new(1)), In::Lit(1.0 / n_cols as f32)],
            },
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Add),
                inputs: vec![In::Step(kg::StepId::new(2)), In::Lit(eps)],
            },
            RowOp::Pointwise {
                op: Op::Math(MathOp::Sqrt),
                inputs: vec![In::Step(kg::StepId::new(3))],
            },
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Div),
                inputs: vec![In::Leaf(kg::LeafId::new(0)), In::Step(kg::StepId::new(4))],
            },
            RowOp::Pointwise {
                op: Op::Widen,
                inputs: vec![In::Leaf(kg::LeafId::new(1))],
            },
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Mul),
                inputs: vec![In::Step(kg::StepId::new(5)), In::Step(kg::StepId::new(6))],
            },
        ],
        output: kg::RowValue::Step(kg::StepId::new(7)),
    }
}

#[test]
fn fused_row_reads_a_packed_bf16_leaf_as_its_widened_values() {
    // An odd row width leaves the scale's last packed word with one element and a zero pad.
    let (num_rows, n_cols, width, eps) = (3usize, 7usize, 4usize, 1e-6f32);
    let x: Vec<f32> = (0..num_rows * n_cols)
        .map(|i| (i as f32 * 0.37).sin() * 2.0 + 0.5)
        .collect();
    // BF16 values exactly: the top half of an f32 whose low half is zero.
    let scale_bits: Vec<u16> = (0..n_cols)
        .map(|j| ((0.5 + j as f32 * 0.3125).to_bits() >> 16) as u16)
        .collect();
    let scale: Vec<f32> = scale_bits
        .iter()
        .map(|&b| f32::from_bits(u32::from(b) << 16))
        .collect();
    let words: Vec<u32> = scale_bits
        .chunks(2)
        .map(|pair| u32::from(pair[0]) | (u32::from(pair.get(1).copied().unwrap_or(0)) << 16))
        .collect();
    let body = kg::fused_row_parallel_views(
        "frp_packed_scale",
        &[num_rows, n_cols],
        &[&[num_rows, n_cols][..], &[n_cols][..]],
        &[
            kg::Layout::contiguous(&[num_rows, n_cols]),
            kg::Layout::contiguous(&[n_cols]),
        ],
        &rmsnorm_with_packed_scale(n_cols, eps),
        width,
        num_rows,
    )
    .expect("a packed leaf is a stated lane");
    let mut buffers = [
        Buffer::from_f32s(&x),
        Buffer::from_u32s(&words),
        Buffer::from_f32s(&vec![f32::NAN; num_rows * n_cols]),
    ];
    run(&body, [num_rows as u32, 1, 1], &mut buffers).expect("interpret");
    let got = buffers[2].to_f32s().unwrap();
    let want: Vec<f32> = x
        .chunks(n_cols)
        .flat_map(|row| {
            let mean = row.iter().map(|v| v * v).sum::<f32>() / n_cols as f32;
            let inv = (mean + eps).sqrt();
            row.iter()
                .zip(&scale)
                .map(move |(v, w)| v / inv * w)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_close(&got, &want, 1e-5);
}
