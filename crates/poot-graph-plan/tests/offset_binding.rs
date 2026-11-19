//! POOT-1566 W1 (card POOT-1572): a view is split once into the byte offset the executor binds its root at and the
//! layout its readers bake, and a reader of flat buffers takes a dense view whose root is bound at the view's own
//! start. The rows plan through the production entries (`compute_views`, the per-equation planner, `compile`).

use std::num::NonZeroU64;

use poot_graph_ir::op::UnOp;
use poot_graph_ir::{Builder, Graph, Slot, TensorType, ValueId};
use poot_graph_plan::{
    CompileLimits, CompileOptions, Plan, Target, compile, compute_views, plan_eqn_views,
};
use poot_target::Backend;
use poot_tensor::DType;
use poot_test_util::device_caps::default_caps_for;

fn target(backend: Backend, align: Option<u64>) -> Target {
    let mut caps = default_caps_for(backend);
    if let Some(align) = align {
        caps.storage_offset_align = NonZeroU64::new(align).unwrap();
    }
    Target::Gpu { backend, caps }
}

/// The plan of the equation that produces `value` under the views of `g` on `target`.
fn plan_of(g: &Graph, target: &Target, value: ValueId) -> Plan {
    let views = compute_views(g, target);
    let Target::Gpu { backend, caps } = target else {
        unreachable!("a GPU target")
    };
    let eqn = g.eqns().iter().find(|eqn| eqn.out == value).unwrap();
    plan_eqn_views(
        g,
        eqn,
        *backend,
        &views,
        caps,
        &poot_graph_plan::test_support::roomy_body_limits(),
    )
    .unwrap_or_else(|error| panic!("the equation plans: {error:?}"))
}

/// `matmul(slice(x, rows start..start + 4), w)` over an `[8, 16]` activation slot, and the slice's id.
fn sliced_matmul(start: usize) -> (Graph, ValueId) {
    let b = Builder::for_tests();
    let x = b
        .slot_named(Slot::Activation, "x", TensorType::f32(vec![8, 16]))
        .unwrap();
    let w = b.constant("w", TensorType::f32(vec![16, 8])).unwrap();
    let rows = b.slice(x, 0, start, start + 4).unwrap();
    let out = b.matmul(rows, w).unwrap();
    (b.finish(out).unwrap(), rows.id())
}

/// SC-a: a slice at an aligned offset into a matmul plans as a view bound at the offset with nothing left over, and
/// its equation launches nothing. Mutation: drop the dense, no-residual arm of `LayoutDemand::Dense` (`admits`
/// answers `false`): the slice dispatches again.
#[test]
fn offset_view_feeds_a_matmul_without_a_copy() {
    let (g, rows) = sliced_matmul(4);
    let target = target(Backend::SpirvVulkan, None);
    let views = compute_views(&g, &target);
    let view = views
        .get(&rows)
        .expect("the slice is a view of the activation");
    // Row 4 of a 16-column f32 slot is 256 bytes in, a multiple of the target's 32-byte binding alignment.
    assert_eq!(view.bound, 256);
    assert_eq!(view.layout.offset, 0);
    match plan_of(&g, &target, rows) {
        Plan::View { bound: 256, .. } => {}
        other => panic!("expected a view bound at byte 256, got {other:?}"),
    }
}

/// SC-b: the slice at row 1 starts 64 bytes in. A device that binds at 16 bytes binds the view at 64; one whose
/// binding alignment is 256 cannot, so the matmul keeps the copy. Mutation: leave `split_offset`'s bound unaligned
/// (`bound = byte_offset`): the 256 case becomes a view bound at byte 64, and `compile` refuses it with
/// `ViewBindingAlignment`.
#[test]
fn misaligned_offset_keeps_the_copy() {
    let (g, rows) = sliced_matmul(1);
    let aligned = target(Backend::SpirvVulkan, Some(16));
    match plan_of(&g, &aligned, rows) {
        Plan::View { bound: 64, .. } => {}
        other => panic!("alignment 16 binds the view at byte 64, got {other:?}"),
    }
    let coarse = target(Backend::SpirvVulkan, Some(256));
    assert!(
        !compute_views(&g, &coarse).contains_key(&rows),
        "alignment 256 cannot bind a view at byte 64"
    );
    assert!(
        matches!(plan_of(&g, &coarse, rows), Plan::Dispatches(_)),
        "the slice keeps its copy"
    );
    compile(&g, &coarse, &CompileOptions::new(CompileLimits::STANDARD))
        .expect("the coarse target compiles with the copy");
}

/// SC-c: the prefix of an I32 slot read flat (the position slot of every model's KV write) is a view on every GPU
/// reference target. Mutation: admit a flat reader only where `bound > 0` (`admits`): the prefix dispatches again.
#[test]
fn prefix_slice_is_a_view_on_every_gpu_target() {
    let b = Builder::for_tests();
    let positions = b
        .slot_named(
            Slot::Activation,
            "positions",
            TensorType::new(vec![8], DType::I32),
        )
        .unwrap();
    let prefix = b.slice(positions, 0, 0, 4).unwrap();
    let out = b.cast(prefix, DType::F32).unwrap();
    let g = b.finish(out).unwrap();
    for backend in [
        Backend::SpirvVulkan,
        Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
        Backend::Nvptx,
    ] {
        let target = target(backend, None);
        let views = compute_views(&g, &target);
        let view = views
            .get(&prefix.id())
            .unwrap_or_else(|| panic!("{backend:?}: the prefix slice is a view"));
        assert_eq!((view.bound, view.layout.offset), (0, 0), "{backend:?}");
        assert!(
            matches!(plan_of(&g, &target, prefix.id()), Plan::View { .. }),
            "{backend:?}: no dispatch for the prefix"
        );
    }
}

/// SC-d: the q, k and v views of one root at aligned offsets have no residual, so their pointwise readers plan one
/// kernel. Mutation: bake the whole offset instead of the residual (`split_offset`'s layout keeps `full.offset`):
/// three kernel keys.
#[test]
fn aligned_views_share_one_kernel() {
    let b = Builder::for_tests();
    let qkv = b
        .slot_named(Slot::Activation, "qkv", TensorType::f32(vec![3, 8]))
        .unwrap();
    let mut readers = Vec::new();
    for row in 0..3 {
        let part = b.slice(qkv, 0, row, row + 1).unwrap();
        readers.push(b.unary(UnOp::Neg, part).unwrap());
    }
    let pair = b
        .binary(poot_graph_ir::BinOp::Add, readers[0], readers[1])
        .unwrap();
    let sum = b
        .binary(poot_graph_ir::BinOp::Add, pair, readers[2])
        .unwrap();
    let g = b.finish(sum).unwrap();
    let readers: Vec<ValueId> = readers.iter().map(|reader| reader.id()).collect();
    let target = target(Backend::SpirvVulkan, None);
    let keys: Vec<String> = readers
        .iter()
        .map(|&reader| match plan_of(&g, &target, reader) {
            Plan::Dispatches(dispatches) => dispatches.single().unwrap().key().to_string(),
            other => panic!("a pointwise reader dispatches, got {other:?}"),
        })
        .collect();
    assert!(
        keys.windows(2).all(|pair| pair[0] == pair[1]),
        "views at 0, 32 and 64 bytes share one kernel, got {keys:?}"
    );
}
