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
        side_outputs: Vec::new(),
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
        poot_target::RoundingConversions::NONE,
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
        side_outputs: Vec::new(),
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
        poot_target::RoundingConversions::NONE,
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

/// POOT-1456: a row kernel writing three values, a per-row scalar and a row as side outputs and a row as its
/// output: `h = x / sqrt(mean(x^2) + eps)`, `s = max(rowmax(|h|) / 0.5, 1)`, `out = h / s`. The params are the leaf,
/// then the side outputs in order, then the output; lane 0 writes the scale once per row, at the row's index.
fn norm_with_row_scale(n_cols: usize, eps: f32) -> kg::RowKernel {
    use kg::{FusedInput as In, FusedScalarOp as Op, RowOp};
    let step = |s: usize| In::Step(kg::StepId::new(s));
    let point = |op, inputs| RowOp::Pointwise { op, inputs };
    let x = In::Leaf(kg::LeafId::new(0));
    kg::RowKernel {
        n_leaves: 1,
        leaf_storage: vec![poot_target::BufferStorage::f32()],
        n_cols,
        steps: vec![
            point(Op::Binary(BinOp::Mul), vec![x, x]),
            RowOp::Reduce {
                op: kg::RowReduce::Sum,
                input: step(0),
            },
            point(
                Op::Binary(BinOp::Mul),
                vec![step(1), In::Lit(1.0 / n_cols as f32)],
            ),
            point(Op::Binary(BinOp::Add), vec![step(2), In::Lit(eps)]),
            point(Op::Math(MathOp::Sqrt), vec![step(3)]),
            // 5: h
            point(Op::Binary(BinOp::Div), vec![x, step(4)]),
            point(Op::Binary(BinOp::Mul), vec![step(5), In::Lit(-1.0)]),
            point(Op::Binary(BinOp::Max), vec![step(5), step(6)]),
            RowOp::Reduce {
                op: kg::RowReduce::Max,
                input: step(7),
            },
            point(Op::Binary(BinOp::Div), vec![step(8), In::Lit(0.5)]),
            // 10: s
            point(Op::Binary(BinOp::Max), vec![step(9), In::Lit(1.0)]),
            // 11: h / s
            point(Op::Binary(BinOp::Div), vec![step(5), step(10)]),
        ],
        output: kg::RowValue::Step(kg::StepId::new(11)),
        side_outputs: vec![
            kg::RowValue::Step(kg::StepId::new(10)),
            kg::RowValue::Step(kg::StepId::new(5)),
        ],
    }
}

/// POOT-1456 SC-6: the multi-output row kernel writes each value where the region's chain computes it, at one and
/// at four lanes per row: the scale once per row, the normed row and the scaled row per element; nothing else of
/// any buffer is written (each starts NaN). Mutation: in `fused_row_parallel_views` store a per-row scalar at
/// `i_local` instead of `row`; the interpreter refuses the store, whose index is not yet defined on that path.
#[test]
fn a_row_kernel_writes_a_per_row_scale_and_two_rows_in_one_pass() {
    let (num_rows, n_cols, eps) = (5usize, 9usize, 1e-6f32);
    let x: Vec<f32> = (0..num_rows * n_cols)
        .map(|i| (i as f32 * 0.61).sin() * 3.0 + if i % 7 == 0 { 2.5 } else { 0.0 })
        .collect();
    let (mut h, mut s, mut out) = (Vec::new(), Vec::new(), Vec::new());
    for row in x.chunks(n_cols) {
        let mean = row.iter().map(|v| v * v).sum::<f32>() / n_cols as f32;
        let den = (mean + eps).sqrt();
        let normed: Vec<f32> = row.iter().map(|v| v / den).collect();
        let scale = (normed.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v.abs())) / 0.5).max(1.0);
        out.extend(normed.iter().map(|v| v / scale));
        h.extend(normed);
        s.push(scale);
    }
    assert!(
        s.iter().any(|&scale| scale > 1.0),
        "the fixture scales a row"
    );
    for width in [1usize, 4] {
        let body = kg::fused_row_parallel_views(
            "frp_row_scale",
            &[num_rows, n_cols],
            &[&[num_rows, n_cols][..]],
            &[kg::Layout::contiguous(&[num_rows, n_cols])],
            &norm_with_row_scale(n_cols, eps),
            width,
            num_rows,
            poot_target::RoundingConversions::NONE,
        )
        .expect("a multi-output row kernel");
        let mut buffers = [
            Buffer::from_f32s(&x),
            Buffer::from_f32s(&vec![f32::NAN; num_rows]),
            Buffer::from_f32s(&vec![f32::NAN; num_rows * n_cols]),
            Buffer::from_f32s(&vec![f32::NAN; num_rows * n_cols]),
        ];
        run(&body, [num_rows as u32, 1, 1], &mut buffers).expect("interpret");
        assert_close(&buffers[1].to_f32s().unwrap(), &s, 1e-5);
        assert_close(&buffers[2].to_f32s().unwrap(), &h, 1e-5);
        assert_close(&buffers[3].to_f32s().unwrap(), &out, 1e-5);
    }
}

/// POOT-1456: a row kernel whose only output is a per-row scalar walks the full row and writes one value per row
/// (`rowmax(|x|) / 127`, the KV-quant scale). Mutation: skip the lane-0 stores in `fused_row_parallel_views` (`if
/// false && !scalar_writes.is_empty()`); every scale stays NaN.
#[test]
fn a_row_kernel_whose_result_is_a_per_row_scalar_reduces_the_full_row() {
    use kg::{FusedInput as In, FusedScalarOp as Op, RowOp};
    let (num_rows, n_cols) = (6usize, 13usize);
    let x: Vec<f32> = (0..num_rows * n_cols)
        .map(|i| (i as f32 * 0.37).cos() * (1.0 + i as f32 * 0.05))
        .collect();
    let kernel = kg::RowKernel {
        n_leaves: 1,
        leaf_storage: vec![poot_target::BufferStorage::f32()],
        n_cols,
        steps: vec![
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Mul),
                inputs: vec![In::Leaf(kg::LeafId::new(0)), In::Lit(-1.0)],
            },
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Max),
                inputs: vec![In::Leaf(kg::LeafId::new(0)), In::Step(kg::StepId::new(0))],
            },
            RowOp::Reduce {
                op: kg::RowReduce::Max,
                input: In::Step(kg::StepId::new(1)),
            },
            RowOp::Pointwise {
                op: Op::Binary(BinOp::Mul),
                inputs: vec![In::Step(kg::StepId::new(2)), In::Lit(1.0 / 127.0)],
            },
        ],
        output: kg::RowValue::Step(kg::StepId::new(3)),
        side_outputs: Vec::new(),
    };
    let want: Vec<f32> = x
        .chunks(n_cols)
        .map(|row| row.iter().fold(f32::NEG_INFINITY, |m, v| m.max(v.abs())) * (1.0 / 127.0))
        .collect();
    for width in [1usize, 8] {
        let body = kg::fused_row_parallel_views(
            "frp_row_scalar",
            &[num_rows, n_cols],
            &[&[num_rows, n_cols][..]],
            &[kg::Layout::contiguous(&[num_rows, n_cols])],
            &kernel,
            width,
            num_rows,
            poot_target::RoundingConversions::NONE,
        )
        .expect("a per-row scalar result");
        let mut buffers = [
            Buffer::from_f32s(&x),
            Buffer::from_f32s(&vec![f32::NAN; num_rows]),
        ];
        run(&body, [num_rows as u32, 1, 1], &mut buffers).expect("interpret");
        assert_eq!(buffers[1].to_f32s().unwrap(), want, "width {width}");
    }
}
