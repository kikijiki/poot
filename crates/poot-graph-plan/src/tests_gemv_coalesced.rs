//! CPU-oracle equivalence for the coalesced imported decode-GEMV bodies. A workgroup-lockstep
//! interpreter (the `poot-kernelgen/tests/dense_q8_0_gemv_interp.rs` pattern) runs every lane of one
//! workgroup against plain `Vec`s - `Barrier` is a real sync point - then sweeps the 2-D
//! `GroupY*x_groups+GroupX` grid that `plan_eqn`'s tiled arm plans, including the
//! over-dispatched tail. The oracle is the plain GEMV `out[col] = sum_k x[k] * W[k,col]` (+ bias),
//! matching `poot_eval`'s matmul. No GPU.
//!
//! Shapes deliberately include N and K that are not multiples of `GEMV_TILE` (32) or `STRIPS` (4), so a
//! partial-tile column or a short last K-strip is exercised.

use std::collections::HashMap;

use poot_kernel_ir::{
    BinOp, Body, Constant, IndexAxis, Operand, Place, ProjectionElem, Rvalue, Statement, Terminator,
};

use crate::imported::ImportedKernel;
use crate::{ExactI32StorageAnalysis, GEMV_TILE, GEMV_WIDTH};

#[derive(Clone, Copy, Debug)]
enum Val {
    F(f32),
    I(i32),
    U(usize),
    B(bool),
}
impl Val {
    fn f(self) -> f32 {
        match self {
            Val::F(x) => x,
            Val::U(x) => x as f32,
            Val::I(x) => x as f32,
            other => panic!("expected number, got {other:?}"),
        }
    }
    fn u(self) -> usize {
        match self {
            Val::U(x) => x,
            Val::I(x) => x as usize,
            other => panic!("expected usize, got {other:?}"),
        }
    }
    fn i(self) -> i32 {
        match self {
            Val::I(x) => x,
            Val::U(x) => x as i32,
            other => panic!("expected i32, got {other:?}"),
        }
    }
    fn b(self) -> bool {
        match self {
            Val::B(x) => x,
            other => panic!("expected bool, got {other:?}"),
        }
    }
}

/// Global buffers keyed by param local index: f32 slices (`f` read-only, `fw` the output), u32 meta
/// (`u`), i32 (`i`).
struct Globals<'a> {
    f: HashMap<u32, &'a [f32]>,
    fw: HashMap<u32, &'a mut Vec<f32>>,
    u: HashMap<u32, &'a [u32]>,
    i: HashMap<u32, &'a [i32]>,
}

struct Lane {
    bb: usize,
    locals: HashMap<u32, Val>,
    done: bool,
}

fn elem_of(p: &Place, locals: &HashMap<u32, Val>) -> (u32, usize) {
    assert_eq!(p.projection.len(), 2, "expected deref+index");
    assert!(matches!(p.projection[0], ProjectionElem::Deref));
    match &p.projection[1] {
        ProjectionElem::Index(l) => (p.local.index, locals.get(&l.index).unwrap().u()),
        other => panic!("proj {other:?}"),
    }
}

fn read(o: &Operand, locals: &HashMap<u32, Val>, g: &Globals) -> Val {
    match o {
        Operand::Copy(p) | Operand::Move(p) => {
            if p.projection.is_empty() {
                *locals.get(&p.local.index).unwrap()
            } else {
                let (slc, idx) = elem_of(p, locals);
                if let Some(v) = g.f.get(&slc) {
                    Val::F(v[idx])
                } else if let Some(v) = g.fw.get(&slc) {
                    Val::F(v[idx])
                } else if let Some(v) = g.u.get(&slc) {
                    Val::U(v[idx] as usize)
                } else {
                    Val::I(g.i.get(&slc).unwrap()[idx])
                }
            }
        }
        Operand::Const(Constant::F32(x)) => Val::F(*x),
        Operand::Const(Constant::I32(x)) => Val::I(*x),
        Operand::Const(Constant::Usize(x)) => Val::U(*x as usize),
        Operand::Const(Constant::U32(x)) => Val::U(*x as usize),
        Operand::Const(c) => panic!("const {c:?}"),
    }
}

fn eval_binop(op: BinOp, a: Val, b: Val) -> Val {
    use BinOp::*;
    match op {
        Add | Sub | Mul | Div | Rem | Min | Max => {
            if matches!(a, Val::F(_)) || matches!(b, Val::F(_)) {
                let (x, y) = (a.f(), b.f());
                Val::F(match op {
                    Add => x + y,
                    Sub => x - y,
                    Mul => x * y,
                    Div => x / y,
                    Min => x.min(y),
                    Max => x.max(y),
                    _ => unreachable!(),
                })
            } else if matches!(a, Val::I(_)) || matches!(b, Val::I(_)) {
                let (x, y) = (a.i(), b.i());
                Val::I(match op {
                    Add => x + y,
                    Sub => x - y,
                    Mul => x * y,
                    Div => x / y,
                    Rem => x % y,
                    _ => unreachable!(),
                })
            } else {
                let (x, y) = (a.u(), b.u());
                Val::U(match op {
                    Add => x + y,
                    Sub => x - y,
                    Mul => x * y,
                    Div => x / y,
                    Rem => x % y,
                    _ => unreachable!(),
                })
            }
        }
        // Integer bit ops the bf16 GEMV body uses (`word >> shift`, `& 0xffff`, `bits << 16`).
        // Operands are U (usize) carrying the u32/u64 pattern; the kernel's u32 ops wrap at 32 bits.
        Shr => Val::U(a.u() >> b.u()),
        Shl => Val::U((a.u() << b.u()) & u32::MAX as usize),
        BitAnd => Val::U(a.u() & b.u()),
        BitOr => Val::U(a.u() | b.u()),
        Lt => Val::B(a.f() < b.f()),
        Le => Val::B(a.f() <= b.f()),
        Gt => Val::B(a.f() > b.f()),
        Ge => Val::B(a.f() >= b.f()),
        Eq => Val::B(a.f() == b.f()),
        Ne => Val::B(a.f() != b.f()),
        other => panic!("binop {other:?}"),
    }
}

/// Run every lane of one workgroup (`w` lanes, group index `(gx, gy)`) in lockstep. A `Barrier`
/// terminator is a real sync: every live lane must reach it before any proceeds. Lanes that already
/// returned are skipped. All lanes of these bodies share the same `col0 < ncols` guard, so barrier
/// targets cannot diverge - the assert catches a body that would deadlock on hardware.
fn run_workgroup(body: &Body, w: usize, gx: usize, gy: usize, g: &mut Globals, lds: &mut [f32]) {
    let mut lanes: Vec<Lane> = (0..w)
        .map(|_| Lane {
            bb: 0,
            locals: HashMap::new(),
            done: false,
        })
        .collect();

    loop {
        if lanes.iter().all(|l| l.done) {
            return;
        }
        let mut barrier_targets: Vec<Option<u32>> = Vec::with_capacity(w);
        for (lane_id, lane) in lanes.iter_mut().enumerate() {
            if lane.done {
                barrier_targets.push(None);
                continue;
            }
            loop {
                let block = &body.blocks[lane.bb];
                for st in &block.statements {
                    match st {
                        Statement::Assign(place, rv) => {
                            if place.projection.is_empty() {
                                let v = match rv {
                                    Rvalue::Use(o) => read(o, &lane.locals, g),
                                    Rvalue::Len(p) => {
                                        let n = g
                                            .fw
                                            .get(&p.local.index)
                                            .map(|v| v.len())
                                            .or_else(|| g.f.get(&p.local.index).map(|v| v.len()))
                                            .or_else(|| g.u.get(&p.local.index).map(|v| v.len()))
                                            .or_else(|| g.i.get(&p.local.index).map(|v| v.len()))
                                            .unwrap();
                                        Val::U(n)
                                    }
                                    Rvalue::Cast { to, operand } => {
                                        let s = read(operand, &lane.locals, g);
                                        match to {
                                            poot_kernel_ir::Ty::F32 => Val::F(s.f()),
                                            poot_kernel_ir::Ty::I32 => Val::I(s.i()),
                                            poot_kernel_ir::Ty::Usize => Val::U(s.u()),
                                            poot_kernel_ir::Ty::U32 => {
                                                Val::U(s.u() as u32 as usize)
                                            }
                                            other => panic!("cast to {other:?}"),
                                        }
                                    }
                                    // `f32::from_bits` in the bf16 GEMV body: bit reinterpret, no conversion.
                                    Rvalue::Bitcast { to, operand } => {
                                        let s = read(operand, &lane.locals, g);
                                        match to {
                                            poot_kernel_ir::Ty::F32 => {
                                                Val::F(f32::from_bits(s.u() as u32))
                                            }
                                            other => panic!("bitcast to {other:?}"),
                                        }
                                    }
                                    Rvalue::BinaryOp(op, x, y) => {
                                        let (a, b) =
                                            (read(x, &lane.locals, g), read(y, &lane.locals, g));
                                        eval_binop(*op, a, b)
                                    }
                                    Rvalue::WorkgroupLocalRead { idx, array } => {
                                        assert_eq!(*array, 0, "single-array LDS kernel");
                                        let i = read(idx, &lane.locals, g).u();
                                        Val::F(lds[i])
                                    }
                                    other => panic!("rvalue {other:?}"),
                                };
                                lane.locals.insert(place.local.index, v);
                            } else {
                                let (slc, idx) = elem_of(place, &lane.locals);
                                let v = match rv {
                                    Rvalue::Use(o) => read(o, &lane.locals, g).f(),
                                    Rvalue::BinaryOp(op, x, y) => {
                                        let (a, b) =
                                            (read(x, &lane.locals, g), read(y, &lane.locals, g));
                                        eval_binop(*op, a, b).f()
                                    }
                                    other => panic!("store rvalue {other:?}"),
                                };
                                g.fw.get_mut(&slc).unwrap()[idx] = v;
                            }
                        }
                        Statement::WorkgroupLocalWrite { idx, value, array } => {
                            assert_eq!(*array, 0, "single-array LDS kernel");
                            let i = read(idx, &lane.locals, g).u();
                            let v = read(value, &lane.locals, g).f();
                            lds[i] = v;
                        }
                        other => panic!("statement {other:?}"),
                    }
                }
                match &block.terminator {
                    Terminator::ThreadIndexCall {
                        destination,
                        dim,
                        target,
                    } => {
                        let v = match dim {
                            IndexAxis::GroupX => Val::U(gx),
                            IndexAxis::GroupY => Val::U(gy),
                            IndexAxis::LocalX => Val::U(lane_id),
                            other => panic!("index axis {other:?}"),
                        };
                        lane.locals.insert(destination.local.index, v);
                        lane.bb = target.index as usize;
                    }
                    Terminator::Goto { target } => lane.bb = target.index as usize,
                    Terminator::SwitchInt { discr, targets } => {
                        let v = read(discr, &lane.locals, g);
                        let taken = if !v.b() {
                            targets
                                .branches
                                .iter()
                                .find(|(c, _)| *c == 0)
                                .map(|(_, t)| t.index)
                                .unwrap()
                        } else {
                            targets.otherwise.index
                        };
                        lane.bb = taken as usize;
                    }
                    Terminator::Barrier { target } => {
                        barrier_targets.push(Some(target.index));
                        break;
                    }
                    Terminator::Return => {
                        lane.done = true;
                        barrier_targets.push(None);
                        break;
                    }
                    Terminator::Trap { .. } => panic!(
                        "unexpected kernel trap: this fixture's kernelgen-built GEMV bodies never assert"
                    ),
                }
            }
        }
        let live_targets: Vec<u32> = barrier_targets.iter().filter_map(|t| *t).collect();
        if let Some(&first) = live_targets.first() {
            assert!(
                live_targets.iter().all(|&t| t == first),
                "divergent barrier targets across lanes in one round: {live_targets:?}"
            );
            for (lane, bt) in lanes.iter_mut().zip(barrier_targets.iter()) {
                if let Some(t) = bt {
                    lane.bb = *t as usize;
                }
            }
        }
    }
}

/// This box's measured wgpu grid cap (card 522), the fixture every test in this file plans
/// against: `plan_eqn`/`decode_gemv_plan`/`gemv_grid` all read it from the caller-supplied
/// `DeviceCaps` now, never a baked constant.
fn test_grid_cap() -> usize {
    poot_target::DeviceCaps::wgpu_rdna3_igpu().max_grid[0] as usize
}

/// The tiled 2-D grid `plan_eqn` plans for `out_numel` elements: one workgroup per
/// `GEMV_TILE` columns, `x_groups = min(nwg, test_grid_cap())` on X, the spill on Y. Returns
/// `(x_groups, y_groups)`; the caller sweeps `gy in 0..y_groups`, `gx in 0..x_groups`.
fn tiled_grid(out_numel: usize) -> (usize, usize) {
    let nwg = out_numel.div_ceil(GEMV_TILE).max(1);
    let x = nwg.min(test_grid_cap());
    let y = nwg.div_ceil(x);
    (x, y)
}

/// Plain GEMV oracle (row-major `W [K, N]`): `out[col] = sum_k x[k]*W[k,col]` (+ `bias[col]`).
fn cpu_ref_b1(k: usize, n: usize, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    (0..n)
        .map(|col| {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[kk] * w[kk * n + col];
            }
            acc + bias.map(|b| b[col]).unwrap_or(0.0)
        })
        .collect()
}

/// Batched oracle: `out[b,n] = sum_k x[b,k]*W[k,n]` (+ `bias[n]`).
fn cpu_ref_batched(
    batch: usize,
    k: usize,
    n: usize,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0f32; batch * n];
    for b in 0..batch {
        for col in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += x[b * k + kk] * w[kk * n + col];
            }
            out[b * n + col] = acc + bias.map(|bd| bd[col]).unwrap_or(0.0);
        }
    }
    out
}

fn fill(seed: u64, count: usize) -> Vec<f32> {
    let mut s = seed.wrapping_add(0x9E3779B97F4A7C15) | 1;
    (0..count)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 0.4 - 0.2
        })
        .collect()
}

fn run_b1(body: &Body, k: usize, n: usize, x: &[f32], w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
    let _ = k;
    let mut out = vec![0f32; n];
    let (x_groups, y_groups) = tiled_grid(n);
    let w_lanes = body.workgroup_size[0] as usize;
    assert_eq!(w_lanes, GEMV_WIDTH, "body workgroup width");
    let x_v = x.to_vec();
    let w_v = w.to_vec();
    let bias_v = bias.map(|b| b.to_vec());
    for gy in 0..y_groups {
        for gx in 0..x_groups {
            let mut fmap: HashMap<u32, &[f32]> = HashMap::new();
            fmap.insert(1, &x_v);
            fmap.insert(2, &w_v);
            if let Some(b) = bias_v.as_ref() {
                fmap.insert(3, b);
            }
            let mut fwmap: HashMap<u32, &mut Vec<f32>> = HashMap::new();
            let out_slot = if bias.is_some() { 4 } else { 3 };
            fwmap.insert(out_slot, &mut out);
            let mut g = Globals {
                f: fmap,
                fw: fwmap,
                u: HashMap::new(),
                i: HashMap::new(),
            };
            let mut lds = vec![0f32; w_lanes];
            run_workgroup(body, w_lanes, gx, gy, &mut g, &mut lds);
        }
    }
    out
}

fn run_batched(
    body: &Body,
    batch: usize,
    k: usize,
    n: usize,
    x: &[f32],
    w: &[f32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let _ = (k, n);
    let total = batch * n;
    let mut out = vec![0f32; total];
    let (x_groups, y_groups) = tiled_grid(total);
    let w_lanes = body.workgroup_size[0] as usize;
    let dims = vec![batch as u32];
    let bias_v = bias.map(|b| b.to_vec());
    for gy in 0..y_groups {
        for gx in 0..x_groups {
            let mut fmap: HashMap<u32, &[f32]> = HashMap::new();
            fmap.insert(1, x);
            fmap.insert(2, w);
            // params: x, w, [bias], dims, out  -> out is last; dims is the u32 meta.
            let (dims_slot, out_slot) = if bias.is_some() { (4, 5) } else { (3, 4) };
            if let Some(b) = bias_v.as_ref() {
                fmap.insert(3, b);
            }
            let mut umap: HashMap<u32, &[u32]> = HashMap::new();
            umap.insert(dims_slot, &dims);
            let mut fwmap: HashMap<u32, &mut Vec<f32>> = HashMap::new();
            fwmap.insert(out_slot, &mut out);
            let mut g = Globals {
                f: fmap,
                fw: fwmap,
                u: umap,
                i: HashMap::new(),
            };
            let mut lds = vec![0f32; w_lanes];
            run_workgroup(body, w_lanes, gx, gy, &mut g, &mut lds);
        }
    }
    out
}

fn assert_close(got: &[f32], want: &[f32], ctx: &str) {
    assert_eq!(got.len(), want.len(), "{ctx}: length");
    let mut max_abs = 0.0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let diff = (g - w).abs();
        max_abs = max_abs.max(diff);
        let tol = 1e-4 + 1e-4 * w.abs();
        assert!(diff <= tol, "{ctx}[{i}]: interp {g} vs cpu {w} diff={diff}");
    }
    eprintln!("{ctx}: ok (n={}, max_abs={max_abs:.2e})", got.len());
}

/// Tile-aligned shapes: N and K multiples of GEMV_TILE / STRIPS.
#[test]
fn gemv_coalesced_tile_aligned_matches_cpu() {
    let body = ImportedKernel::GemvCoalesced.body();
    assert_eq!(body.workgroup_size, [GEMV_WIDTH as u32, 1, 1]);
    let (k, n) = (64usize, 128usize); // K=64=2*STRIPS*8, N=128=4*GEMV_TILE
    let x = fill(1, k);
    let w = fill(2, k * n);
    let want = cpu_ref_b1(k, n, &x, &w, None);
    let got = run_b1(body, k, n, &x, &w, None);
    assert_close(&got, &want, "gemv_coalesced aligned");
}

/// N not a multiple of GEMV_TILE (partial last tile) and K not a multiple of STRIPS (uneven
/// K-stripes). N=100 = 3*32+4: the last workgroup's columns 96..127 are mostly past the end.
#[test]
fn gemv_coalesced_ragged_n_and_k_matches_cpu() {
    let body = ImportedKernel::GemvCoalesced.body();
    let (k, n) = (7usize, 100usize); // K=7 not div by STRIPS=4; N=100 not div by TILE=32
    let x = fill(3, k);
    let w = fill(4, k * n);
    let want = cpu_ref_b1(k, n, &x, &w, None);
    let got = run_b1(body, k, n, &x, &w, None);
    assert_close(&got, &want, "gemv_coalesced ragged");
}

/// N smaller than one tile (a single workgroup with mostly invalid lanes) and K=1 (only strip 0
/// has work; strips 1..3 write zero partials that the LDS fold must still add).
#[test]
fn gemv_coalesced_subtile_n_and_k1_matches_cpu() {
    let body = ImportedKernel::GemvCoalesced.body();
    let (k, n) = (1usize, 5usize);
    let x = fill(5, k);
    let w = fill(6, k * n);
    let want = cpu_ref_b1(k, n, &x, &w, None);
    let got = run_b1(body, k, n, &x, &w, None);
    assert_close(&got, &want, "gemv_coalesced sub-tile");

    let (k2, n2) = (3usize, 1usize); // N=1: one valid lane, STRIPS-1 zero partials
    let x2 = fill(7, k2);
    let w2 = fill(8, k2 * n2);
    let want2 = cpu_ref_b1(k2, n2, &x2, &w2, None);
    let got2 = run_b1(body, k2, n2, &x2, &w2, None);
    assert_close(&got2, &want2, "gemv_coalesced n1");
}

/// Bias epilogue at a ragged shape.
#[test]
fn gemv_coalesced_bias_ragged_matches_cpu() {
    let body = ImportedKernel::GemvCoalescedBias.body();
    let (k, n) = (5usize, 70usize); // N=70 = 2*32+6
    let x = fill(9, k);
    let w = fill(10, k * n);
    let bias = fill(11, n);
    let want = cpu_ref_b1(k, n, &x, &w, Some(&bias));
    let got = run_b1(body, k, n, &x, &w, Some(&bias));
    assert_close(&got, &want, "gemv_coalesced_bias ragged");
}

/// The launch the planner really emits for a `[1, K] @ [K, N]` decode GEMV, against the interpreter's
/// tiled sweep: the plan (card 525: the planner stores the grid on the plan itself, alongside the
/// kernel choice) must carry `GEMV_WIDTH` threads per workgroup, `x_groups` workgroups on X and the
/// spill on Y - exactly the `(x_groups, y_groups)` the sweeps above walk. A launch that disagrees with
/// the body's `col0 = (GroupY*x_groups + GroupX) * GEMV_TILE` leaves columns unwritten or writes past
/// N. Mutation observed red: the production `nwg` uses `div_ceil(GEMV_TILE + 1)`.
///
/// MUTATION (recorded here, never left in the tree; card 525): in
/// `predicates.rs`'s `decode_gemv_plan`, `grid = [(gx * GEMV_WIDTH) as u32, gy as u32, 1]` was changed
/// to `grid = [(gx * GEMV_WIDTH) as u32, 1, 1]` (dropping the Y spill). Rerunning this test then
/// panicked on the AmdGcn Y-spill assertion below: `assertion `left == right` failed: the tiled grid
/// must spill onto Y past the X cap left: [8388480, 1, 1] right: [8388480, 2, 1]`. Restoring the real
/// formula made it green again.
#[test]
fn gemv_coalesced_grid_matches_dispatch_formula() {
    use poot_graph_ir::{Builder, TensorType};

    for n in [1usize, 31, 32, 33, 100, 4096, 70_000, 201_088] {
        let bld = Builder::new();
        let x = bld.constant("x", TensorType::f32(vec![1, 1]));
        let w = bld.constant("w", TensorType::f32(vec![1, n]));
        let y = bld.matmul(x, w);
        let g = bld.finish(y);
        let eqn = &g.eqns[0];
        let plan = crate::plan_eqn_analyzed(
            &ExactI32StorageAnalysis::new(&g),
            &g,
            eqn,
            poot_target::Backend::SpirvVulkan,
            &poot_test_util::device_caps::default_caps_for(poot_target::Backend::SpirvVulkan),
            &poot_test_util::graph_fixtures::roomy_body_limits(),
        )
        .unwrap();
        let grid = match plan {
            crate::Plan::Compute { grid, .. } | crate::Plan::ComputeMeta { grid, .. } => grid,
            other => panic!("decode GEMV must plan as Compute/ComputeMeta, got {other:?}"),
        };
        let (gx, gy) = tiled_grid(n);
        assert_eq!(
            grid,
            [(gx * GEMV_WIDTH) as u32, gy as u32, 1],
            "planned grid for a decode GEMV over N={n}"
        );
    }

    // A Y-spill shape (just over the X cap, so the tiled workgroup count needs a second row): at the
    // real wgpu cap (`test_grid_cap()`), a shape this large is always past `DECODE_GEMV_CHUNK_TRIGGER`
    // too (card 525 own premise), so `Backend::SpirvVulkan`'s own watchdog budget
    // would route it to `Plan::ComputeChunks` instead of the 2-D tile-spill grid this case means to
    // exercise. Card 546b: the grid cap is no longer one constant shared by
    // every backend (`DeviceCaps::rocm_default()`/`ptx_default()` are genuinely uncapped - HSA/CUDA
    // have no `gridDim` ceiling - so AmdGcn no longer spills here at all), so this test now plans
    // against `Backend::SpirvVulkan` itself, with its real `max_grid` but no watchdog budget (a
    // synthetic fixture: no real wgpu/RADV device both caps the grid and has no display watchdog, but
    // the tile-spill grid math this asserts is backend-generic), to reach `Plan::Compute`'s Y-spill
    // path on a real compiled `Plan` rather than just checking `tiled_grid` against itself.
    let n = test_grid_cap() * GEMV_TILE + 3;
    assert!(
        n as u64 > poot_target::DECODE_GEMV_CHUNK_TRIGGER,
        "this Y-spill shape must be within the SpirvVulkan-chunked range - if this ever fails, this \
         test's premise (an uncapped watchdog budget never chunks) may need revisiting too"
    );
    let bld = Builder::new();
    let x = bld.constant("x", TensorType::f32(vec![1, 1]));
    let w = bld.constant("w", TensorType::f32(vec![1, n]));
    let y = bld.matmul(x, w);
    let g = bld.finish(y);
    let eqn = &g.eqns[0];
    let spirv = poot_target::Backend::SpirvVulkan;
    let no_watchdog_caps = poot_target::DeviceCaps {
        watchdog_budget: None,
        ..poot_test_util::device_caps::default_caps_for(spirv)
    };
    let plan = crate::plan_eqn_analyzed(
        &ExactI32StorageAnalysis::new(&g),
        &g,
        eqn,
        spirv,
        &no_watchdog_caps,
        &poot_test_util::graph_fixtures::roomy_body_limits(),
    )
    .unwrap();
    let grid = match plan {
        crate::Plan::Compute { grid, .. } => grid,
        other => panic!(
            "Y-spill decode GEMV with no watchdog budget must plan as Plan::Compute, got {other:?}"
        ),
    };
    assert_eq!(
        grid,
        [(test_grid_cap() * GEMV_WIDTH) as u32, 2, 1],
        "the tiled grid must spill onto Y past the X cap"
    );
}

/// Batched B>1, ragged N/K, with and without bias. Tile ownership can straddle a batch-row
/// boundary when B*N is not tile-aligned relative to N - covered by N=50 (not div 32) and B=3.
#[test]
fn gemv_batched_coalesced_ragged_matches_cpu() {
    let body = ImportedKernel::GemvBatchedCoalesced.body();
    let (batch, k, n) = (3usize, 6usize, 50usize);
    let x = fill(12, batch * k);
    let w = fill(13, k * n);
    let want = cpu_ref_batched(batch, k, n, &x, &w, None);
    let got = run_batched(body, batch, k, n, &x, &w, None);
    assert_close(&got, &want, "gemv_batched_coalesced");

    let body_b = ImportedKernel::GemvBatchedCoalescedBias.body();
    let bias = fill(14, n);
    let want_b = cpu_ref_batched(batch, k, n, &x, &w, Some(&bias));
    let got_b = run_batched(body_b, batch, k, n, &x, &w, Some(&bias));
    assert_close(&got_b, &want_b, "gemv_batched_coalesced_bias");
}

/// Batched B=1 degenerates to the single-sequence oracle (the plan still uses the B=1 body when
/// out_numel == N, but the batched body must agree on that boundary too).
#[test]
fn gemv_batched_coalesced_b1_agrees_with_b1_oracle() {
    let body = ImportedKernel::GemvBatchedCoalesced.body();
    let (batch, k, n) = (1usize, 9usize, 33usize);
    let x = fill(15, batch * k);
    let w = fill(16, k * n);
    let want = cpu_ref_batched(batch, k, n, &x, &w, None);
    let got = run_batched(body, batch, k, n, &x, &w, None);
    assert_close(&got, &want, "gemv_batched_coalesced B=1");
}

/// Card 370 RNE: round an f32 to bf16 bits (round-to-nearest-even), the inverse of the kernel's
/// exact `from_bits(bits << 16)` widen.
fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    if x.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let rounding_bias = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding_bias) >> 16) as u16
}

/// Exact widen of bf16 bits to f32 (low 16 bits zero-fill): the value the f32 reference must use so
/// the bf16 body is bit-identical to it.
fn widen_bf16(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Pack f32 weights through bf16 into the card 380 u32 lane layout (element `i` in lane `i/2` at
/// shift `(i%2)*16`), and return the exact-widened f32 values for the reference.
fn pack_widened(w: &[f32]) -> (Vec<u32>, Vec<f32>) {
    let bf: Vec<u16> = w.iter().map(|&x| f32_to_bf16(x)).collect();
    let mut words = vec![0u32; bf.len().div_ceil(2)];
    for (i, &b) in bf.iter().enumerate() {
        let shift = ((i % 2) * 16) as u32;
        words[i / 2] |= (b as u32) << shift;
    }
    let widened = bf.iter().map(|&b| widen_bf16(b)).collect();
    (words, widened)
}

/// Run the B=1 bf16-weight body: weight is param 1 as packed `u32` lanes (`Globals::u`).
fn run_b1_bf16(
    body: &Body,
    n: usize,
    x: &[f32],
    weight_words: &[u32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let mut out = vec![0f32; n];
    let (x_groups, y_groups) = tiled_grid(n);
    let w_lanes = body.workgroup_size[0] as usize;
    assert_eq!(w_lanes, GEMV_WIDTH, "body workgroup width");
    let bias_v = bias.map(|b| b.to_vec());
    for gy in 0..y_groups {
        for gx in 0..x_groups {
            let mut fmap: HashMap<u32, &[f32]> = HashMap::new();
            fmap.insert(1, x);
            if let Some(b) = bias_v.as_ref() {
                fmap.insert(3, b);
            }
            let mut umap: HashMap<u32, &[u32]> = HashMap::new();
            umap.insert(2, weight_words);
            let mut fwmap: HashMap<u32, &mut Vec<f32>> = HashMap::new();
            let out_slot = if bias.is_some() { 4 } else { 3 };
            fwmap.insert(out_slot, &mut out);
            let mut g = Globals {
                f: fmap,
                fw: fwmap,
                u: umap,
                i: HashMap::new(),
            };
            let mut lds = vec![0f32; w_lanes];
            run_workgroup(body, w_lanes, gx, gy, &mut g, &mut lds);
        }
    }
    out
}

/// Run the batched bf16-weight body: weight is param 1 as packed `u32` lanes, dims is param
/// (2 or 3) as `u32`.
fn run_batched_bf16(
    body: &Body,
    batch: usize,
    n: usize,
    x: &[f32],
    weight_words: &[u32],
    bias: Option<&[f32]>,
) -> Vec<f32> {
    let total = batch * n;
    let mut out = vec![0f32; total];
    let (x_groups, y_groups) = tiled_grid(total);
    let w_lanes = body.workgroup_size[0] as usize;
    let dims = vec![batch as u32];
    let bias_v = bias.map(|b| b.to_vec());
    for gy in 0..y_groups {
        for gx in 0..x_groups {
            let mut fmap: HashMap<u32, &[f32]> = HashMap::new();
            fmap.insert(1, x);
            let (dims_slot, out_slot) = if bias.is_some() { (4, 5) } else { (3, 4) };
            if let Some(b) = bias_v.as_ref() {
                fmap.insert(3, b);
            }
            let mut umap: HashMap<u32, &[u32]> = HashMap::new();
            umap.insert(2, weight_words);
            umap.insert(dims_slot, &dims);
            let mut fwmap: HashMap<u32, &mut Vec<f32>> = HashMap::new();
            fwmap.insert(out_slot, &mut out);
            let mut g = Globals {
                f: fmap,
                fw: fwmap,
                u: umap,
                i: HashMap::new(),
            };
            let mut lds = vec![0f32; w_lanes];
            run_workgroup(body, w_lanes, gx, gy, &mut g, &mut lds);
        }
    }
    out
}

/// Bit-identical to the f32-widened reference: the bf16 body's in-kernel widen must produce exactly
/// the same products as running the **f32 coalesced body** on the exact widenings of the same packed
/// bytes. Both bodies share the strip/LDS accumulation order, so the comparison is bit-for-bit; a
/// sequential-k CPU oracle would differ by summation order and cannot be the bit-identity reference.
#[test]
fn gemv_coalesced_bf16_bit_identical_to_f32_widened_reference() {
    let bf16_body = ImportedKernel::GemvCoalescedBf16.body();
    let f32_body = ImportedKernel::GemvCoalesced.body();
    let (k, n) = (7usize, 100usize); // ragged N and K
    let x = fill(20, k);
    let w = fill(21, k * n);
    let (words, widened) = pack_widened(&w);
    let want = run_b1(f32_body, k, n, &x, &widened, None);
    let got = run_b1_bf16(bf16_body, n, &x, &words, None);
    assert_eq!(got.len(), want.len(), "length");
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "bf16 GEMV[{i}] must be bit-identical to the f32 body on widened weights: got {g}, want {w}"
        );
    }
    eprintln!("gemv_coalesced_bf16 bit-identical to f32-widened body (n={n})");
}

/// Bias epilogue at a ragged shape, same bit-identity contract (bias stays F32).
#[test]
fn gemv_coalesced_bias_bf16_bit_identical_to_f32_widened_reference() {
    let bf16_body = ImportedKernel::GemvCoalescedBiasBf16.body();
    let f32_body = ImportedKernel::GemvCoalescedBias.body();
    let (k, n) = (5usize, 70usize);
    let x = fill(22, k);
    let w = fill(23, k * n);
    let bias = fill(24, n);
    let (words, widened) = pack_widened(&w);
    let want = run_b1(f32_body, k, n, &x, &widened, Some(&bias));
    let got = run_b1_bf16(bf16_body, n, &x, &words, Some(&bias));
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "bias bf16 GEMV[{i}]: got {g}, want {w}"
        );
    }
}

/// Batched B>1 ragged, with and without bias, bit-identical to the f32 body on widened weights.
#[test]
fn gemv_batched_coalesced_bf16_bit_identical_to_f32_widened_reference() {
    let bf16_body = ImportedKernel::GemvBatchedCoalescedBf16.body();
    let f32_body = ImportedKernel::GemvBatchedCoalesced.body();
    let (batch, k, n) = (3usize, 6usize, 50usize);
    let x = fill(25, batch * k);
    let w = fill(26, k * n);
    let (words, widened) = pack_widened(&w);
    let want = run_batched(f32_body, batch, k, n, &x, &widened, None);
    let got = run_batched_bf16(bf16_body, batch, n, &x, &words, None);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "batched bf16 GEMV[{i}]");
    }

    let bf16_body_b = ImportedKernel::GemvBatchedCoalescedBiasBf16.body();
    let f32_body_b = ImportedKernel::GemvBatchedCoalescedBias.body();
    let bias = fill(27, n);
    let want_b = run_batched(f32_body_b, batch, k, n, &x, &widened, Some(&bias));
    let got_b = run_batched_bf16(bf16_body_b, batch, n, &x, &words, Some(&bias));
    for (i, (g, w)) in got_b.iter().zip(&want_b).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "batched bias bf16 GEMV[{i}]");
    }
}

/// Tile-aligned shapes for the bf16 body (covers the even-col packed-lane path fully).
#[test]
fn gemv_coalesced_bf16_tile_aligned_bit_identical() {
    let bf16_body = ImportedKernel::GemvCoalescedBf16.body();
    let f32_body = ImportedKernel::GemvCoalesced.body();
    let (k, n) = (64usize, 128usize);
    let x = fill(28, k);
    let w = fill(29, k * n);
    let (words, widened) = pack_widened(&w);
    let want = run_b1(f32_body, k, n, &x, &widened, None);
    let got = run_b1_bf16(bf16_body, n, &x, &words, None);
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        assert_eq!(g.to_bits(), w.to_bits(), "aligned bf16 GEMV[{i}]");
    }
}
