//! POOT-1576 (W3b of the POOT-1574 design): one layout rule, `operand_demand`. A value is a view where every reader
//! can read a layout, and a movement chain whose tail cannot is one copy that reads the chain's root through the
//! composed layout.
//!
//! The rows plan through the production entries (`compile`, `compute_views` with the per-equation planner) and run
//! the planned bodies on the kernel-IR interpreter against the CPU oracle (`poot_eval::apply_movement`).

use poot_eval::apply_movement;
use poot_graph_ir::op::{BinOp, UnOp};
use poot_graph_ir::{Builder, Graph, OpKind, TensorType, ValueId};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Dispatch, KernelChoice, Plan, Planned, Program, Target, compile,
    compute_views, plan_eqn_choice,
};
use poot_kernel_ir::interp::{self, Buffer};
use poot_kernelgen::{KernelRequest, Layout, MovementOp, MovementRead, MovementSpec};
use poot_target::Backend;
use poot_tensor::{DType, HostTensor};

/// `eqn` planned for `backend` against the graph's promoted `views`, with the kernel choice it was built from.
fn plan_with_views(
    g: &Graph,
    eqn: &poot_graph_ir::Eqn,
    backend: Backend,
    views: &poot_graph_plan::Views,
) -> Planned {
    plan_eqn_choice(
        g,
        eqn,
        &target(backend),
        views,
        &poot_graph_plan::test_support::roomy_body_limits(),
        None,
    )
    .expect("the equation plans")
}

fn options() -> CompileOptions {
    CompileOptions::new(CompileLimits::STANDARD)
}

fn target(backend: Backend) -> Target {
    Target::Gpu {
        backend,
        caps: poot_test_util::device_caps::default_caps_for(backend),
    }
}

/// The equations of `program` that launch a kernel and move data (a transpose, slice, broadcast, reshape or concat).
fn movement_dispatches(program: &Program) -> Vec<&OpKind> {
    program
        .planned()
        .filter(|(eqn, plan)| {
            matches!(
                eqn.op,
                OpKind::Transpose { .. }
                    | OpKind::Slice { .. }
                    | OpKind::Broadcast { .. }
                    | OpKind::Reshape { .. }
                    | OpKind::Concat { .. }
            ) && matches!(plan, Plan::Dispatches(_))
        })
        .map(|(eqn, _)| &eqn.op)
        .collect()
}

fn movement_spec<'p>(program: &'p Program, eqn: &poot_graph_ir::Eqn) -> &'p MovementSpec {
    match program.kernel_choice(eqn) {
        KernelChoice::Generated(KernelRequest::Movement(spec)) => spec,
        other => panic!("expected a generated movement, got {other:?}"),
    }
}

/// `matmul(transpose(reshape(slice(a))), w)`: the slice and the reshape are readable through a layout and the
/// transpose is read by a contraction, which reads flat.
fn chain_into_a_contraction() -> Graph {
    let b = Builder::for_tests();
    let a = b.constant("a", TensorType::f32(vec![4, 6, 8])).unwrap();
    let w = b.constant("w", TensorType::f32(vec![24, 5])).unwrap();
    let sliced = b.slice(a, 2, 1, 5).unwrap();
    let merged = b.reshape(sliced, vec![24, 4]).unwrap();
    let transposed = b.transpose(merged, vec![1, 0]).unwrap();
    let out = b.matmul(transposed, w).unwrap();
    b.finish(out).unwrap()
}

/// SC-b1: a chain into a dense reader is one dispatch. The slice and the reshape are `Plan::View`, and the transpose
/// is one `Copy` that reads the slice's source through the composed layout.
///
/// Mutation: `movement::layout_demand` answers `Dense` for a movement's data input; the slice materializes as well and the
/// chain is two dispatches.
#[test]
fn a_chain_into_a_dense_reader_is_one_copy() {
    for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
        let program = compile(&chain_into_a_contraction(), &target(backend), &options()).unwrap();
        let dispatched = movement_dispatches(&program);
        assert_eq!(dispatched.len(), 1, "{backend:?}: {dispatched:?}");
        assert!(
            matches!(dispatched[0], OpKind::Transpose { .. }),
            "{backend:?}: {dispatched:?}"
        );
        let (eqn, _) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::Transpose { .. }))
            .unwrap();
        // `a[4, 6, 8]` cut at column 1 and read as `[24, 4]` has strides `[8, 1]`; transposed, `[1, 8]`.
        assert_eq!(
            movement_spec(&program, eqn).op,
            MovementOp::Copy {
                source: vec![4, 6, 8],
                from: Layout {
                    strides: vec![1, 8],
                    offset: 1
                },
            },
            "{backend:?}"
        );
    }
}

/// The planned dispatch of `eqn` run on the interpreter over `inputs`, one buffer per value operand.
fn run_dispatch(dispatch: &Dispatch, mut buffers: Vec<Buffer>, out_len: usize) -> Buffer {
    buffers.push(Buffer::from_f32s(&vec![0.0; out_len]));
    let body = dispatch.body();
    assert_eq!(body.params().count(), buffers.len(), "[inputs, out]");
    let grid = dispatch.launch().grid;
    let wg = body.workgroup_size;
    let groups = [
        grid[0].div_ceil(wg[0]),
        grid[1].div_ceil(wg[1]),
        grid[2].div_ceil(wg[2]),
    ];
    interp::run(body, groups, &mut buffers).expect("the planned body runs");
    buffers.pop().unwrap()
}

fn xorshift(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// SC-b2: random two- to four-link chains of transpose, slice, broadcast and reshape whose tail is the graph output
/// (pinned, so it materializes). The tail's one generated `Copy`, run on the interpreter over the chain's root
/// buffer, equals `poot_eval::apply_movement` folded over the chain, bit for bit. The row counts the chains whose
/// copy reads through a view (a link above the tail), so it is not vacuous.
///
/// Mutation: pass B of `compute_views`' composition in the movement planner starts from `Layout::contiguous` of
/// the tail's input instead of the input's layout: the copy reads the wrong elements and the comparison goes red at a
/// reported seed.
#[test]
fn materialized_chain_matches_oracle_fuzz() {
    let mut seed = 0x1357_9bdf_2468_ace0u64;
    let mut through_a_view = 0usize;
    for round in 0..300 {
        let bld = Builder::for_tests();
        let mut shape: Vec<usize> = (0..3)
            .map(|_| 2 + (xorshift(&mut seed) % 3) as usize)
            .collect();
        let a = bld.constant("a", TensorType::f32(shape.clone())).unwrap();
        let mut cur = a;
        let mut links: Vec<ValueId> = Vec::new();
        for _ in 0..(2 + xorshift(&mut seed) as usize % 3) {
            match xorshift(&mut seed) % 4 {
                0 => {
                    let rank = shape.len();
                    let mut perm: Vec<usize> = (0..rank).collect();
                    for i in (1..rank).rev() {
                        perm.swap(i, xorshift(&mut seed) as usize % (i + 1));
                    }
                    cur = bld.transpose(cur, perm.clone()).unwrap();
                    shape = perm.iter().map(|&p| shape[p]).collect();
                }
                1 => {
                    let axis = xorshift(&mut seed) as usize % shape.len();
                    let start = xorshift(&mut seed) as usize % shape[axis];
                    let end = start + 1 + xorshift(&mut seed) as usize % (shape[axis] - start);
                    cur = bld.slice(cur, axis, start, end).unwrap();
                    shape[axis] = end - start;
                }
                2 => {
                    let mut wider = vec![2 + (xorshift(&mut seed) % 3) as usize];
                    wider.extend_from_slice(&shape);
                    cur = bld.broadcast(cur, wider.clone()).unwrap();
                    shape = wider;
                }
                _ => {
                    let axis = xorshift(&mut seed) as usize % shape.len();
                    match xorshift(&mut seed) % 3 {
                        0 if shape.len() > 1 => {
                            let axis = axis.min(shape.len() - 2);
                            let merged = shape[axis] * shape[axis + 1];
                            shape.splice(axis..axis + 2, [merged]);
                        }
                        1 if shape[axis].is_multiple_of(2) => {
                            let half = shape[axis] / 2;
                            shape.splice(axis..=axis, [2, half]);
                        }
                        _ => shape.insert(axis, 1),
                    }
                    cur = bld.reshape(cur, shape.clone()).unwrap();
                }
            }
            links.push(cur.id());
        }
        let g = bld.finish(cur).unwrap();
        let backend = Backend::SpirvVulkan;
        let views = compute_views(&g, &target(backend));

        let a_shape = g.aval(a.id()).shape.clone();
        let a_data: Vec<f32> = (0..a_shape.iter().product::<usize>())
            .map(|i| i as f32 + 0.5)
            .collect();
        // The oracle's value after each link, the root's first: a link that is no view (a reshape its input's
        // layout cannot merge bars that input) is the root of the links above it.
        let mut values = vec![HostTensor::f32(a_shape, a_data)];
        for &id in &links {
            let eqn = g.eqns().iter().find(|e| e.out == id).unwrap();
            let next = apply_movement(&eqn.op, &[values.last().unwrap().clone()]).unwrap();
            values.push(next);
        }
        let oracle = values.pop().unwrap();
        let mut ids = vec![a.id()];
        ids.extend(&links);
        let mut root_at = links.len() - 1;
        while root_at > 0 && views.contains_key(&ids[root_at]) {
            root_at -= 1;
        }
        let root = values[root_at].as_f32().unwrap().to_vec();

        let tail = g.eqns().iter().find(|e| e.out == cur.id()).unwrap();
        let Planned { plan, .. } = plan_with_views(&g, tail, backend, &views);
        let out_len = oracle.as_f32().unwrap().len();
        let got = match plan {
            // A reshape over a dense input is the buffer itself: nothing to compare.
            Plan::Alias(_) => continue,
            plan => {
                let dispatch = plan.single().expect("one dispatch for the tail");
                if root_at + 1 < links.len() {
                    through_a_view += 1;
                }
                // The dispatch reads its input through the layout the view left past its `bound`: the executor
                // binds the root there, so the kernel sees the root from that byte on.
                let input = poot_graph_plan::value_ids(tail)[0];
                let bound = views.get(&input).map_or(0, |view| view.bound as usize);
                let buffer = run_dispatch(
                    dispatch,
                    vec![Buffer::from_f32s(&root[bound / size_of::<f32>()..])],
                    out_len,
                );
                buffer.to_f32s().unwrap()
            }
        };
        let want: Vec<u32> = oracle
            .as_f32()
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let got: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want, "round {round}, seed state {seed:#x}");
    }
    assert!(
        through_a_view > 50,
        "copies through a view: {through_a_view}"
    );
}

/// SC-b3: `concat(neg(slice(x)), slice(x))` (the RoPE `rotate_half`) has no slice dispatch: the first half is a
/// view read by the fused negation, the second a view the concat reads through its layout, and the concat's output
/// equals the oracle's.
///
/// Mutation: `movement::layout_demand` answers `Dense` for a `Concat` part; the second slice materializes and a slice
/// dispatch appears.
#[test]
fn concat_reads_its_parts_through_layouts() {
    let b = Builder::for_tests();
    let x = b.constant("x", TensorType::f32(vec![3, 8])).unwrap();
    let low = b.slice(x, 1, 0, 4).unwrap();
    let high = b.slice(x, 1, 4, 8).unwrap();
    let negated = b.unary(UnOp::Neg, low).unwrap();
    let out = b.concat(1, &[negated, high]).unwrap();
    let g = b.finish(out).unwrap();
    for backend in [Backend::SpirvVulkan, Backend::Nvptx] {
        let program = compile(&g, &target(backend), &options()).unwrap();
        let dispatched = movement_dispatches(&program);
        assert_eq!(dispatched.len(), 1, "{backend:?}: {dispatched:?}");
        assert!(
            matches!(dispatched[0], OpKind::Concat { .. }),
            "{backend:?}: {dispatched:?}"
        );
    }

    // The concat's generated body over the second part's root buffer reads the same elements the oracle joins.
    let backend = Backend::SpirvVulkan;
    let views = compute_views(&g, &target(backend));
    let concat = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Concat { .. }))
        .unwrap();
    let Planned { plan, choice } = plan_with_views(&g, concat, backend, &views);
    let root: Vec<f32> = (0..24).map(|i| i as f32 - 7.25).collect();
    let negated_low: Vec<f32> = (0..3)
        .flat_map(|r| (0..4).map(move |c| (r, c)))
        .map(|(r, c)| -root[r * 8 + c])
        .collect();
    let want: Vec<f32> = (0..3)
        .flat_map(|r| {
            (0..4)
                .map(|c| negated_low[r * 4 + c])
                .chain((4..8).map(|c| root[r * 8 + c]))
                .collect::<Vec<_>>()
        })
        .collect();
    let dispatch = plan.single().unwrap();
    let got = run_dispatch(
        dispatch,
        vec![Buffer::from_f32s(&negated_low), Buffer::from_f32s(&root)],
        24,
    )
    .to_f32s()
    .unwrap();
    assert_eq!(got, want);
    let spec = match choice {
        KernelChoice::Generated(KernelRequest::Movement(spec)) => spec,
        other => panic!("expected a generated concat, got {other:?}"),
    };
    let MovementOp::Concat { inputs, .. } = spec.op else {
        panic!("expected a concat: {spec:?}");
    };
    assert_eq!(inputs[0], MovementRead::contiguous(vec![3, 4]));
    assert_eq!(inputs[1].source, [3, 8]);
    assert_eq!(
        inputs[1].from,
        Layout {
            strides: vec![8, 1],
            offset: 4
        }
    );
}

/// SC-b4: an exact-I32 pointwise reader takes an I32 view. The census `and` case, `and(transpose(x), m)` over I32
/// words, plans the transpose as a view, and the generated `and` over the root buffer equals the oracle's.
///
/// Mutation: restore the f32-region-only gate for I32 leaves (`elementwise::layout_demand` answers `Dense` for an I32 operand
/// of a `Binary`); the transpose dispatches.
#[test]
fn an_exact_i32_reader_takes_an_i32_view() {
    let b = Builder::for_tests();
    let x = b
        .constant("x", TensorType::new(vec![2, 3], DType::I32))
        .unwrap();
    let mask = b
        .constant("mask", TensorType::new(vec![3, 2], DType::I32))
        .unwrap();
    let out = b
        .binary(BinOp::And, b.transpose(x, vec![1, 0]).unwrap(), mask)
        .unwrap();
    let g = b.finish(out).unwrap();
    let backend = Backend::SpirvVulkan;
    let program = compile(&g, &target(backend), &options()).unwrap();
    assert_eq!(
        movement_dispatches(&program).len(),
        0,
        "the transpose of the I32 words is a view"
    );

    let views = compute_views(&g, &target(backend));
    let and = g
        .eqns()
        .iter()
        .find(|e| matches!(e.op, OpKind::Binary(BinOp::And)))
        .unwrap();
    let Planned { plan, .. } = plan_with_views(&g, and, backend, &views);
    let words = [0x0f0f, 0x00ff, 0x1234, -1, 0x7fff_0000, 0x55aa];
    let masks = [0x00ff, 0x0ff0, 0x3333, 0x0f0f, 0x7f00_ffff, 0x5a5a];
    let want: Vec<i32> = [0, 3, 1, 4, 2, 5]
        .iter()
        .zip(&masks)
        .map(|(&i, m)| words[i] & m)
        .collect();
    let dispatch = plan.single().unwrap();
    let mut buffers = vec![Buffer::from_i32s(&words), Buffer::from_i32s(&masks)];
    buffers.push(Buffer::from_i32s(&[0; 6]));
    let body = dispatch.body();
    assert_eq!(body.params().count(), buffers.len());
    let grid = dispatch.launch().grid;
    let wg = body.workgroup_size;
    let groups = [
        grid[0].div_ceil(wg[0]),
        grid[1].div_ceil(wg[1]),
        grid[2].div_ceil(wg[2]),
    ];
    interp::run(body, groups, &mut buffers).unwrap();
    assert_eq!(buffers.pop().unwrap().to_i32s().unwrap(), want);
}
