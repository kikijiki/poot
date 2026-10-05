use super::*;
/// The `f32` unaries the pointwise-fusion phase admits into a region. [`kernel_mapping::map_fused_op`]
/// (`poot-graph-plan`) must lower every one of these inside a fused region: legality here and the
/// kernel table are one contract (card 536b, R481-007), checked by
/// `kernel_mapping::tests::fused_region_kernel_table_covers_every_fusable_unary` against this same
/// list rather than a second hand-copied one.
pub const FUSABLE_FLOAT_UNARY_OPS: &[UnOp] = &[
    UnOp::Neg,
    UnOp::Exp,
    UnOp::Log,
    UnOp::Sqrt,
    UnOp::Recip,
    UnOp::Tanh,
    UnOp::Erf,
];

/// Is this op a pointwise step the fusion pass fuses? Binary and the float unaries `poot-kernelgen`
/// emits inside a fused region's body (the transcendentals `Exp`/`Log`/`Sqrt`/`Recip`/`Tanh`/`Erf`
/// and `Neg`); the activations are compositions over these, so swiglu and geglu fuse as plain chains.
/// Movement/contraction ops are never pointwise.
fn is_fusable<V: ValidationChannel>(g: &Graph<V>, eqn: &Eqn) -> bool {
    match &eqn.op {
        OpKind::Binary(_) => true,
        OpKind::Unary(u) if g.aval(eqn.out).dtype == DType::I32 => {
            matches!(u, UnOp::Not | UnOp::Clz)
        }
        OpKind::Unary(u) => FUSABLE_FLOAT_UNARY_OPS.contains(u),
        OpKind::Select => g.aval(eqn.out).dtype == DType::I32,
        _ => false,
    }
}

fn is_i32_pointwise<V: ValidationChannel>(g: &Graph<V>, eqn: &Eqn) -> bool {
    g.aval(eqn.out).dtype == DType::I32 && is_fusable(g, eqn)
}

fn to_fused_op(op: &OpKind) -> FusedOp {
    match op {
        OpKind::Binary(b) => FusedOp::Binary(*b),
        OpKind::Unary(u) => FusedOp::Unary(*u),
        OpKind::Select => FusedOp::Select,
        other => unreachable!("non-fusable op in a fused region: {}", other.name()),
    }
}

/// Disjoint-set over eqn indices (the fusion grouping). Path-compressing find, union-by-min so the
/// representative is the lowest (earliest, topo-first) eqn index.
struct DisjointSet {
    parent: Vec<usize>,
}
impl DisjointSet {
    fn new(n: usize) -> Self {
        DisjointSet {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]]; // path halving
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb {
            // keep the smaller index as representative (deterministic, topo-first).
            let (lo, hi) = (ra.min(rb), ra.max(rb));
            self.parent[hi] = lo;
        }
    }
}

/// If `eqn` is a last-axis, keepdim reduction (the shape the row synthesizer handles), its op.
fn row_reduce_op<V: ValidationChannel>(
    g: &Graph<V>,
    eqn: &Eqn,
) -> Option<poot_graph_ir::op::RedOp> {
    if let OpKind::Reduce { op, axis, keepdim } = &eqn.op {
        let in_rank = match eqn.inputs.first() {
            Some(Operand::Value(v)) => g.aval(*v).rank(),
            _ => return None,
        };
        if *keepdim && *axis + 1 == in_rank {
            return Some(*op);
        }
    }
    None
}

/// A candidate for row-wise (reduction-rooted) fusion: a pointwise op or a last-axis keepdim reduce.
/// Exact I32 stays out of this phase: I32 reduction is rejected at inference, and I32 row-fusion is not
/// part of the pointwise-DAG fusion contract.
fn is_row_candidate<V: ValidationChannel>(g: &Graph<V>, eqn: &Eqn) -> bool {
    (is_fusable(g, eqn) && g.aval(eqn.out).dtype != DType::I32) || row_reduce_op(g, eqn).is_some()
}

/// `shape` fits a row-region signature `(L, N)`: it is `L ++ [N]` (a row) or `L ++ [1]` (a per-row scalar).
fn row_fits(shape: &[usize], l: &[usize], n: usize) -> bool {
    shape.len() == l.len() + 1
        && &shape[..l.len()] == l
        && (shape[l.len()] == n || shape[l.len()] == 1)
}

/// The leaf/local layout of a fused region: the root eqn (its escaping output), the external value-operand
/// leaves in first-encounter order, and the outer-`ValueId` -> region-local map (leaves `0..n_inputs`,
/// then each member's output). Shared by pointwise and row-wise region building.
fn region_layout<V: ValidationChannel>(
    g: &Graph<V>,
    members: &[usize],
    producer: &[Option<usize>],
) -> (usize, Vec<ValueId>, HashMap<ValueId, usize>) {
    let member_set: HashSet<usize> = members.iter().copied().collect();
    let consumed_internally = |out: ValueId| -> bool {
        members.iter().any(|&m| {
            g.eqns[m]
                .inputs
                .iter()
                .any(|o| matches!(o, Operand::Value(v) if *v == out))
        })
    };
    let root = *members
        .iter()
        .find(|&&m| !consumed_internally(g.eqns[m].out))
        .expect("a fused region has exactly one escaping (root) output");

    let mut local_of: HashMap<ValueId, usize> = HashMap::new();
    let mut leaves: Vec<ValueId> = Vec::new();
    for &m in members {
        for o in &g.eqns[m].inputs {
            if let Operand::Value(v) = o {
                let internal = producer[*v].is_some_and(|p| member_set.contains(&p));
                if !internal && !local_of.contains_key(v) {
                    local_of.insert(*v, leaves.len());
                    leaves.push(*v);
                }
            }
        }
    }
    let n_inputs = leaves.len();
    for (s, &m) in members.iter().enumerate() {
        local_of.insert(g.eqns[m].out, n_inputs + s);
    }
    (root, leaves, local_of)
}

/// Map an outer operand to a region operand, given the local map.
fn to_region_operand(o: &Operand, local_of: &HashMap<ValueId, usize>) -> FusedOperand {
    match o {
        Operand::Value(v) => FusedOperand::Local(local_of[v]),
        Operand::Lit(s) => FusedOperand::Lit(*s),
    }
}

fn i32_unit_last_slice<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &[Option<usize>],
    value: ValueId,
) -> Option<(usize, ValueId, usize)> {
    let eqn_idx = producer[value]?;
    let OpKind::Slice { axis, start, end } = g.eqns[eqn_idx].op else {
        return None;
    };
    if end != start + 1 || g.aval(value).dtype != DType::I32 {
        return None;
    }
    let out_shape = &g.aval(value).shape;
    if out_shape.last() != Some(&1) || axis + 1 != out_shape.len() {
        return None;
    }
    let Operand::Value(parent) = g.eqns[eqn_idx].inputs.first()? else {
        return None;
    };
    let parent_ty = g.aval(*parent);
    if parent_ty.dtype != DType::I32
        || parent_ty.shape.len() != out_shape.len()
        || parent_ty.shape.last().copied().unwrap_or(0) <= start
    {
        return None;
    }
    Some((eqn_idx, *parent, start))
}

fn i32_region_leaves<V: ValidationChannel>(
    g: &Graph<V>,
    members: &[usize],
    producer: &[Option<usize>],
    consumers: &[Vec<usize>],
    pinned: &HashSet<ValueId>,
) -> (Vec<ValueId>, HashMap<ValueId, FusedOperand>, Vec<usize>) {
    let member_set: HashSet<usize> = members.iter().copied().collect();
    let mut leaves: Vec<ValueId> = Vec::new();
    let mut leaf_index: HashMap<ValueId, usize> = HashMap::new();
    let mut operand_of: HashMap<ValueId, FusedOperand> = HashMap::new();
    let mut slice_eqns: Vec<usize> = Vec::new();
    for &member in members {
        for operand in &g.eqns[member].inputs {
            let Operand::Value(value) = operand else {
                continue;
            };
            let internal = producer[*value].is_some_and(|p| member_set.contains(&p));
            if internal || operand_of.contains_key(value) {
                continue;
            }
            if let Some(&input) = leaf_index.get(value) {
                operand_of.insert(*value, FusedOperand::Local(input));
                continue;
            }
            if let Some((slice_eqn, parent, lane)) = i32_unit_last_slice(g, producer, *value) {
                let input = *leaf_index.entry(parent).or_insert_with(|| {
                    let index = leaves.len();
                    leaves.push(parent);
                    index
                });
                operand_of.insert(*value, FusedOperand::PackLane { input, lane });
                slice_eqns.push(slice_eqn);
                continue;
            }
            let input = leaves.len();
            leaf_index.insert(*value, input);
            leaves.push(*value);
            operand_of.insert(*value, FusedOperand::Local(input));
        }
    }
    let n_inputs = leaves.len();
    for (step, &member) in members.iter().enumerate() {
        operand_of.insert(g.eqns[member].out, FusedOperand::Local(n_inputs + step));
    }
    let absorbed_slices = slice_eqns
        .into_iter()
        .filter(|&slice_eqn| {
            let out = g.eqns[slice_eqn].out;
            !pinned.contains(&out)
                && consumers[out]
                    .iter()
                    .all(|&consumer| member_set.contains(&consumer))
        })
        .collect();
    (leaves, operand_of, absorbed_slices)
}

fn to_i32_region_operand(o: &Operand, operand_of: &HashMap<ValueId, FusedOperand>) -> FusedOperand {
    match o {
        Operand::Value(v) => operand_of[v],
        Operand::Lit(s) => FusedOperand::Lit(*s),
    }
}

/// Automatic fusion (G5): group connected eqns into regions and replace each region (>= 2
/// eqns) with one fused eqn carrying the recipe, so the executor synthesizes a single kernel instead of
/// one kernel + a global round-trip per primitive. Two region kinds: reduction-rooted [`OpKind::FusedRow`]
/// (one-pass RMSNorm / fused softmax: last-axis reductions plus their pointwise feed and epilogue) and
/// pure-pointwise [`OpKind::Fused`]. Row regions are formed first; the pointwise pass takes what is left.
///
/// Legality (before any cost model, graph-architecture.md section 8): a producer P fuses into a
/// consumer only when P's output is used exactly once and is not a graph output / cache value, or
/// (pointwise regions) when every use of it is inside the one region it joins.
/// Pointwise regions additionally require equal output shapes; row regions require all member shapes to
/// fit one `(leading dims, N)` signature (rows `[.,N]` and per-row scalars `[.,1]`). The graph output
/// and the KV-cache state aliases are preserved: a region's escaping value reuses the root eqn's
/// `ValueId`, so downstream references stay valid.
///
/// Exact-I32 equations fuse too (card 533): an I32 pointwise chain becomes an [`OpKind::Fused`] region
/// whose steps stay in `Ge`/`GeU`/`Sub`/`Add`/`Mul`, `Not`/`Clz` and `Select`, and whose packed
/// last-axis concat is carried by the region's `pack`. The exact-I32 evaluator has a `Fused` arm for
/// exactly that subset (`poot-eval/src/exact_i32.rs`), so the region is supported rather than dropped.
pub fn fuse<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    // First turn data-no-op transposes into (free) reshapes so they cost no dispatch on the launch-bound
    // decode; the fusion grouping below treats both as movement boundaries, so this only removes launches.
    // Then collapse the reshape chains that elision exposes (seq-1 attention q/k/v), shrinking the graph.
    let g = &collapse_reshape_chains(&elide_noop_transposes(g));
    let n_vals = g.values.len();
    let n_eqns = g.eqns.len();

    let mut producer: Vec<Option<usize>> = vec![None; n_vals];
    let mut uses: Vec<usize> = vec![0; n_vals];
    for (e, eqn) in g.eqns.iter().enumerate() {
        producer[eqn.out] = Some(e);
        for o in &eqn.inputs {
            if let Operand::Value(v) = o {
                uses[*v] += 1;
            }
        }
    }
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    // a producer P fuses into a consumer over value v iff v is used exactly once and not pinned.
    let single_use = |v: ValueId| uses[v] == 1 && !pinned.contains(&v);

    let mut fused_at: HashMap<usize, Eqn> = HashMap::new(); // root eqn idx -> replacement eqn
    let mut absorbed: HashSet<usize> = HashSet::new(); // non-root members dropped
    let mut claimed: HashSet<usize> = HashSet::new(); // every eqn in some region (excluded from later phases)

    // --- Phase A: row-wise (reduction-rooted) regions ------------------------------------------------
    // union candidates whose shapes fit a common (L, N) signature seeded from the reductions.
    let mut ds = DisjointSet::new(n_eqns);
    let mut sig: HashMap<usize, (Vec<usize>, usize)> = HashMap::new();
    for (e, eqn) in g.eqns.iter().enumerate() {
        if row_reduce_op(g, eqn).is_some()
            && let Some(Operand::Value(v)) = eqn.inputs.first()
        {
            let s = &g.aval(*v).shape;
            let k = s.len();
            sig.insert(e, (s[..k - 1].to_vec(), s[k - 1]));
        }
    }
    // grow to a fixpoint: an edge joins two candidates when a consistent signature covers both ends.
    loop {
        let mut changed = false;
        for (e, eqn) in g.eqns.iter().enumerate() {
            if !is_row_candidate(g, eqn) {
                continue;
            }
            for o in &eqn.inputs {
                let Operand::Value(v) = o else { continue };
                if !single_use(*v) {
                    continue;
                }
                let Some(pe) = producer[*v] else { continue };
                if !is_row_candidate(g, &g.eqns[pe]) {
                    continue;
                }
                let (re, rp) = (ds.find(e), ds.find(pe));
                if re == rp {
                    continue;
                }
                let sg = match (sig.get(&re), sig.get(&rp)) {
                    (Some(a), Some(b)) if a == b => Some(a.clone()),
                    (Some(_), Some(_)) => None, // conflicting signatures: do not merge
                    (Some(a), None) | (None, Some(a)) => Some(a.clone()),
                    (None, None) => None,
                };
                let Some((l, n)) = sg else { continue };
                if row_fits(&g.aval(g.eqns[e].out).shape, &l, n)
                    && row_fits(&g.aval(*v).shape, &l, n)
                {
                    ds.union(e, pe);
                    sig.insert(ds.find(e), (l, n));
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    // collect row components (those with a signature, i.e. containing a reduce).
    let mut row_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in 0..n_eqns {
        if is_row_candidate(g, &g.eqns[e]) {
            let r = ds.find(e);
            if sig.contains_key(&r) {
                row_groups.entry(r).or_default().push(e);
            }
        }
    }
    for (rep, members) in &row_groups {
        if members.len() < 2 {
            continue;
        }
        let (l, n) = &sig[rep];
        let (n, axis) = (*n, l.len());
        let (root, leaves, local_of) = region_layout(g, members, &producer);
        // The FusedRow kernel requires its output (root) to be a row-width value: it derives n_cols, the
        // leaf strides, num_rows, and the output layout from the root eqn's shape, assuming its reduced
        // axis is the full row width `n`. If the region output is reduced (axis size != n) the kernel
        // iterates 1 column instead of the full row. Two ways to get a reduced output: (a) the root is a
        // bare reduce, and (b) the root is a scalar pointwise applied to a reduce (e.g. `quant_scale_kv`
        // = `reduce_max(|x|) * (1/127)`, output `[..,1]`); case (b) once slipped past a "root is a
        // reduce" guard and produced garbage on real KV-quant weights (uniform synthetic weights masked
        // it, since reducing 1 column ~= the true absmax). The condition is the output shape's reduced
        // axis == the full width. rmsnorm/softmax are fine: their reduce is internal and the root is a
        // row-width pointwise. When the output is reduced, skip the region: the pointwise members fall
        // to the pointwise-fusion phase and the reduce(s) stay plain `Reduce` ops. Fusing into a
        // reduce-output region would need the reduce width carried separately from the output width.
        if g.aval(g.eqns[root].out).shape.get(axis) != Some(&n) {
            continue;
        }
        let steps: Vec<RowStep> = members
            .iter()
            .map(|&m| {
                let eqn = &g.eqns[m];
                if let Some(op) = row_reduce_op(g, eqn) {
                    RowStep::Reduce {
                        op,
                        input: to_region_operand(&eqn.inputs[0], &local_of),
                    }
                } else {
                    RowStep::Pointwise {
                        op: to_fused_op(&eqn.op),
                        inputs: eqn
                            .inputs
                            .iter()
                            .map(|o| to_region_operand(o, &local_of))
                            .collect(),
                    }
                }
            })
            .collect();
        let region = RowRegion {
            n_inputs: leaves.len(),
            axis,
            steps,
            output: local_of[&g.eqns[root].out],
        };
        fused_at.insert(
            root,
            Eqn {
                op: OpKind::FusedRow(region),
                inputs: leaves.into_iter().map(Operand::Value).collect(),
                out: g.eqns[root].out,
                layer: g.eqns[root].layer,
            },
        );
        for &m in members {
            claimed.insert(m);
            if m != root {
                absorbed.insert(m);
            }
        }
    }

    // --- Phase B: f32 pointwise regions -------------------------------------------------------------
    let f32_pointwise = |e: usize| {
        is_fusable(g, &g.eqns[e])
            && !claimed.contains(&e)
            && g.aval(g.eqns[e].out).dtype != DType::I32
    };
    let mut dp = DisjointSet::new(n_eqns);
    for (e, eqn) in g.eqns.iter().enumerate() {
        if !is_fusable(g, eqn) || claimed.contains(&e) || g.aval(eqn.out).dtype == DType::I32 {
            continue;
        }
        let e_shape = &g.aval(eqn.out).shape;
        for o in &eqn.inputs {
            let Operand::Value(v) = o else { continue };
            let Some(pe) = producer[*v] else { continue };
            if is_fusable(g, &g.eqns[pe])
                && g.aval(g.eqns[pe].out).dtype != DType::I32
                && !claimed.contains(&pe)
                && single_use(*v)
                && &g.aval(*v).shape == e_shape
            {
                dp.union(pe, e);
            }
        }
    }
    // A producer read more than once fuses too when every read is by a member of one region (a diamond:
    // the activation compositions read their input twice, `x / (1 + exp(-x))`). Its value then never
    // escapes the region, so legality is the single-use rule's: no outside reader, no pinned value.
    let mut readers: Vec<Vec<usize>> = vec![Vec::new(); n_vals];
    for (e, eqn) in g.eqns.iter().enumerate() {
        for o in &eqn.inputs {
            if let Operand::Value(v) = o {
                readers[*v].push(e);
            }
        }
    }
    loop {
        let mut changed = false;
        for pe in (0..n_eqns).filter(|&pe| f32_pointwise(pe)) {
            let v = g.eqns[pe].out;
            if uses[v] < 2 || pinned.contains(&v) {
                continue;
            }
            let group = dp.find(readers[v][0]);
            if dp.find(pe) != group
                && readers[v].iter().all(|&r| {
                    f32_pointwise(r)
                        && g.aval(g.eqns[r].out).shape == g.aval(v).shape
                        && dp.find(r) == group
                })
            {
                dp.union(pe, group);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    let mut pw_groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for e in 0..n_eqns {
        if f32_pointwise(e) {
            pw_groups.entry(dp.find(e)).or_default().push(e);
        }
    }
    for members in pw_groups.values() {
        if members.len() < 2 {
            continue;
        }
        let (root, leaves, local_of) = region_layout(g, members, &producer);
        let steps: Vec<FusedStep> = members
            .iter()
            .map(|&m| FusedStep {
                op: to_fused_op(&g.eqns[m].op),
                inputs: g.eqns[m]
                    .inputs
                    .iter()
                    .map(|o| to_region_operand(o, &local_of))
                    .collect(),
            })
            .collect();
        let region = FusedRegion {
            n_inputs: leaves.len(),
            steps,
            output: local_of[&g.eqns[root].out],
            pack: Vec::new(),
        };
        fused_at.insert(
            root,
            Eqn {
                op: OpKind::Fused(region),
                inputs: leaves.into_iter().map(Operand::Value).collect(),
                out: g.eqns[root].out,
                layer: g.eqns[root].layer,
            },
        );
        for &m in members {
            if m != root {
                absorbed.insert(m);
            }
        }
    }

    fuse_i32_pack_concat(g, &producer, &pinned, &mut fused_at, &mut absorbed);
    fuse_closed_i32_pointwise(g, &producer, &pinned, &mut fused_at, &mut absorbed);

    // rebuild the eqn list in order: a root emits its fused eqn, an absorbed member is dropped, the rest
    // pass through.
    let mut eqns: Vec<Eqn> = Vec::with_capacity(n_eqns);
    for e in 0..n_eqns {
        if let Some(fe) = fused_at.remove(&e) {
            eqns.push(fe);
        } else if !absorbed.contains(&e) {
            eqns.push(g.eqns[e].clone());
        }
    }

    Graph { eqns, ..g.clone() }
}

/// The one contraction fuse rule (Card 557, R-557-1): a `MatMul` whose output is read only by a
/// broadcast `Add` of an `[N]` leaf (a bias the graph binds, `N` the matmul's output width) becomes one
/// `MatMulBias(a, b, bias)` at the add's output, so the bias rides in the contraction kernel's epilogue
/// instead of a second dispatch. This is the only producer of `MatMulBias`: tracers state the matmul
/// and the add (`ops::linear`), and Card 727 generalizes this rule into the contraction epilogue.
///
/// The legality is `fuse`'s: the matmul output is used exactly once and is not pinned (a graph output
/// or state value). The bias must be a leaf, so the fused equation reads no value defined after the
/// matmul. The matmul's own equation is dropped; every other equation passes through in order.
pub fn fuse_bias_epilogues<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let mut producer: Vec<Option<usize>> = vec![None; g.values.len()];
    let mut uses: Vec<usize> = vec![0; g.values.len()];
    for (e, eqn) in g.eqns.iter().enumerate() {
        producer[eqn.out] = Some(e);
        for operand in &eqn.inputs {
            if let Operand::Value(v) = operand {
                uses[*v] += 1;
            }
        }
    }
    let pinned: HashSet<ValueId> = g.pinned_values().collect();
    // The `MatMul` equation index and bias value an `Add` folds, if it is the epilogue shape.
    let epilogue = |add: &Eqn| -> Option<(usize, ValueId)> {
        let (OpKind::Binary(BinOp::Add), [Operand::Value(x), Operand::Value(y)]) =
            (&add.op, add.inputs.as_slice())
        else {
            return None;
        };
        [(*x, *y), (*y, *x)]
            .into_iter()
            .find_map(|(product, bias)| {
                let matmul = producer[product]?;
                let width = *g.aval(product).shape.last()?;
                (matches!(g.eqns[matmul].op, OpKind::MatMul)
                    && uses[product] == 1
                    && !pinned.contains(&product)
                    && producer[bias].is_none()
                    && g.aval(bias).shape == [width])
                .then_some((matmul, bias))
            })
    };
    let mut fused_at: HashMap<usize, Eqn> = HashMap::new();
    let mut absorbed: HashSet<usize> = HashSet::new();
    for (e, add) in g.eqns.iter().enumerate() {
        let Some((matmul, bias)) = epilogue(add) else {
            continue;
        };
        let mut inputs = g.eqns[matmul].inputs.clone();
        inputs.push(Operand::Value(bias));
        fused_at.insert(
            e,
            Eqn {
                op: OpKind::MatMulBias,
                inputs,
                out: add.out,
                layer: add.layer,
            },
        );
        absorbed.insert(matmul);
    }
    let eqns = g
        .eqns
        .iter()
        .enumerate()
        .filter(|(e, _)| !absorbed.contains(e))
        .map(|(e, eqn)| fused_at.remove(&e).unwrap_or_else(|| eqn.clone()))
        .collect();
    Graph { eqns, ..g.clone() }
}

fn fuse_i32_pack_concat<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &[Option<usize>],
    pinned: &HashSet<ValueId>,
    fused_at: &mut HashMap<usize, Eqn>,
    absorbed: &mut HashSet<usize>,
) {
    let n_eqns = g.eqns.len();
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); g.values.len()];
    for (index, eqn) in g.eqns.iter().enumerate() {
        for input in &eqn.inputs {
            if let Operand::Value(value) = input {
                consumers[*value].push(index);
            }
        }
    }
    let same_shape = |left: ValueId, right: ValueId| g.aval(left).shape == g.aval(right).shape;
    let mut claimed: HashSet<usize> = fused_at
        .keys()
        .copied()
        .chain(absorbed.iter().copied())
        .collect();

    for concat_idx in 0..n_eqns {
        if claimed.contains(&concat_idx) {
            continue;
        }
        let OpKind::Concat { axis } = g.eqns[concat_idx].op else {
            continue;
        };
        if g.aval(g.eqns[concat_idx].out).dtype != DType::I32 {
            continue;
        }
        let mut parts: Vec<ValueId> = Vec::new();
        let mut part_eqns: Vec<usize> = Vec::new();
        let mut valid = true;
        for input in &g.eqns[concat_idx].inputs {
            let Operand::Value(value) = input else {
                valid = false;
                break;
            };
            let Some(producer_eqn) = producer[*value] else {
                valid = false;
                break;
            };
            if claimed.contains(&producer_eqn)
                || pinned.contains(value)
                || !is_i32_pointwise(g, &g.eqns[producer_eqn])
            {
                valid = false;
                break;
            }
            let shape = &g.aval(*value).shape;
            if shape.last() != Some(&1) || axis + 1 != shape.len() {
                valid = false;
                break;
            }
            if !parts.is_empty() && !same_shape(parts[0], *value) {
                valid = false;
                break;
            }
            parts.push(*value);
            part_eqns.push(producer_eqn);
        }
        if !valid || parts.len() < 2 {
            continue;
        }
        let part_shape_id = parts[0];
        let mut ancestors = vec![false; n_eqns];
        let mut pending = part_eqns.clone();
        for &part_eqn in &part_eqns {
            ancestors[part_eqn] = true;
        }
        while let Some(current) = pending.pop() {
            for input in &g.eqns[current].inputs {
                let Operand::Value(value) = input else {
                    continue;
                };
                let Some(producer_eqn) = producer[*value] else {
                    continue;
                };
                if claimed.contains(&producer_eqn)
                    || ancestors[producer_eqn]
                    || pinned.contains(value)
                    || !is_i32_pointwise(g, &g.eqns[producer_eqn])
                    || !same_shape(*value, part_shape_id)
                {
                    continue;
                }
                ancestors[producer_eqn] = true;
                pending.push(producer_eqn);
            }
        }
        let mut in_region = ancestors;
        let mut changed = true;
        while changed {
            changed = false;
            for index in 0..n_eqns {
                if !in_region[index] {
                    continue;
                }
                let out = g.eqns[index].out;
                let escapes = consumers[out]
                    .iter()
                    .any(|&consumer| consumer != concat_idx && !in_region[consumer]);
                if escapes {
                    in_region[index] = false;
                    changed = true;
                }
            }
        }
        if part_eqns.iter().any(|&index| !in_region[index]) {
            continue;
        }
        let mut members: Vec<usize> = (0..n_eqns).filter(|&index| in_region[index]).collect();
        if members.len() < 2 {
            continue;
        }
        members.sort_unstable();
        let (leaves, operand_of, absorbed_slices) =
            i32_region_leaves(g, &members, producer, &consumers, pinned);
        let n_inputs = leaves.len();
        let steps: Vec<FusedStep> = members
            .iter()
            .map(|&member| FusedStep {
                op: to_fused_op(&g.eqns[member].op),
                inputs: g.eqns[member]
                    .inputs
                    .iter()
                    .map(|operand| to_i32_region_operand(operand, &operand_of))
                    .collect(),
            })
            .collect();
        let pack: Vec<usize> = parts
            .iter()
            .map(|value| match operand_of[value] {
                FusedOperand::Local(id) => id,
                other => panic!("packed concat part must be a region local, got {other:?}"),
            })
            .collect();
        fused_at.insert(
            concat_idx,
            Eqn {
                op: OpKind::Fused(FusedRegion {
                    n_inputs,
                    steps,
                    output: pack[0],
                    pack,
                }),
                inputs: leaves.into_iter().map(Operand::Value).collect(),
                out: g.eqns[concat_idx].out,
                layer: g.eqns[concat_idx].layer,
            },
        );
        claimed.insert(concat_idx);
        for &member in &members {
            claimed.insert(member);
            absorbed.insert(member);
        }
        for slice_eqn in absorbed_slices {
            claimed.insert(slice_eqn);
            absorbed.insert(slice_eqn);
        }
    }
}

fn fuse_closed_i32_pointwise<V: ValidationChannel>(
    g: &Graph<V>,
    producer: &[Option<usize>],
    pinned: &HashSet<ValueId>,
    fused_at: &mut HashMap<usize, Eqn>,
    absorbed: &mut HashSet<usize>,
) {
    let n_eqns = g.eqns.len();
    let mut consumers: Vec<Vec<usize>> = vec![Vec::new(); g.values.len()];
    for (index, eqn) in g.eqns.iter().enumerate() {
        for input in &eqn.inputs {
            if let Operand::Value(value) = input {
                consumers[*value].push(index);
            }
        }
    }
    let same_shape = |left: ValueId, right: ValueId| g.aval(left).shape == g.aval(right).shape;
    let mut claimed: HashSet<usize> = fused_at
        .keys()
        .copied()
        .chain(absorbed.iter().copied())
        .collect();

    let is_i32_root = |index: usize| -> bool {
        if !is_i32_pointwise(g, &g.eqns[index]) {
            return false;
        }
        let out = g.eqns[index].out;
        if pinned.contains(&out) {
            return true;
        }
        let uses = &consumers[out];
        if uses.is_empty() {
            return false;
        }
        uses.iter().any(|&consumer| {
            !is_i32_pointwise(g, &g.eqns[consumer]) || !same_shape(out, g.eqns[consumer].out)
        })
    };

    for root in (0..n_eqns).rev() {
        if claimed.contains(&root) || !is_i32_root(root) {
            continue;
        }
        let root_out = g.eqns[root].out;
        let mut ancestors = vec![false; n_eqns];
        let mut pending = vec![root];
        ancestors[root] = true;
        while let Some(current) = pending.pop() {
            for input in &g.eqns[current].inputs {
                let Operand::Value(value) = input else {
                    continue;
                };
                let Some(producer_eqn) = producer[*value] else {
                    continue;
                };
                if claimed.contains(&producer_eqn)
                    || ancestors[producer_eqn]
                    || !is_i32_pointwise(g, &g.eqns[producer_eqn])
                    || !same_shape(*value, root_out)
                {
                    continue;
                }
                ancestors[producer_eqn] = true;
                pending.push(producer_eqn);
            }
        }
        let mut in_region = ancestors.clone();
        for index in 0..n_eqns {
            if index != root && (pinned.contains(&g.eqns[index].out) || !ancestors[index]) {
                in_region[index] = false;
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for index in 0..n_eqns {
                if index == root || !in_region[index] {
                    continue;
                }
                let out = g.eqns[index].out;
                if consumers[out].iter().any(|&consumer| !in_region[consumer]) {
                    in_region[index] = false;
                    changed = true;
                }
            }
        }
        let mut members: Vec<usize> = (0..n_eqns).filter(|&index| in_region[index]).collect();
        if members.len() < 2 {
            continue;
        }
        members.sort_unstable();
        let region_root = members
            .iter()
            .copied()
            .find(|&index| {
                !members.iter().any(|&member| {
                    g.eqns[member].inputs.iter().any(|operand| {
                        matches!(operand, Operand::Value(value) if *value == g.eqns[index].out)
                    })
                })
            })
            .expect("a fused region has exactly one escaping (root) output");
        let (leaves, operand_of, absorbed_slices) =
            i32_region_leaves(g, &members, producer, &consumers, pinned);
        let steps: Vec<FusedStep> = members
            .iter()
            .map(|&member| FusedStep {
                op: to_fused_op(&g.eqns[member].op),
                inputs: g.eqns[member]
                    .inputs
                    .iter()
                    .map(|operand| to_i32_region_operand(operand, &operand_of))
                    .collect(),
            })
            .collect();
        let FusedOperand::Local(output) = operand_of[&g.eqns[region_root].out] else {
            panic!("fused I32 root must be a region local");
        };
        fused_at.insert(
            region_root,
            Eqn {
                op: OpKind::Fused(FusedRegion {
                    n_inputs: leaves.len(),
                    steps,
                    output,
                    pack: Vec::new(),
                }),
                inputs: leaves.into_iter().map(Operand::Value).collect(),
                out: g.eqns[region_root].out,
                layer: g.eqns[region_root].layer,
            },
        );
        for &member in &members {
            claimed.insert(member);
            if member != region_root {
                absorbed.insert(member);
            }
        }
        for slice_eqn in absorbed_slices {
            claimed.insert(slice_eqn);
            absorbed.insert(slice_eqn);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Builder;
    use poot_graph_ir::types::TensorType;

    fn single_fused_region(g: &Graph) -> &FusedRegion {
        assert_eq!(g.eqns.len(), 1, "expected exactly one surviving eqn: {g:?}");
        match &g.eqns[0].op {
            OpKind::Fused(region) => region,
            other => panic!("expected one Fused region, got {other:?}"),
        }
    }

    /// SC-002 (R481-007): a `Tanh`/`Erf`/`Recip` pointwise chain fuses into one region, matching
    /// what `kernel_mapping::map_fused_op` (`poot-graph-plan`) lowers - `is_fusable` is the gate. Mutation: drop one of the three from
    /// `FUSABLE_FLOAT_UNARY_OPS`; this assertion's step count drops and the row goes red.
    #[test]
    fn recip_tanh_erf_fuse_into_one_region() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, 8]));
        let g = b.unary(UnOp::Tanh, x);
        let e = b.unary(UnOp::Erf, g);
        let r = b.unary(UnOp::Recip, e);
        let graph = b.finish(r);

        let fused = fuse(&graph);
        let region = single_fused_region(&fused);
        assert_eq!(
            region.steps.len(),
            3,
            "expected all three ops in one region: {region:?}"
        );
        assert!(
            region
                .steps
                .iter()
                .any(|s| matches!(s.op, FusedOp::Unary(UnOp::Tanh))),
            "missing Tanh step: {region:?}"
        );
        assert!(
            region
                .steps
                .iter()
                .any(|s| matches!(s.op, FusedOp::Unary(UnOp::Erf))),
            "missing Erf step: {region:?}"
        );
        assert!(
            region
                .steps
                .iter()
                .any(|s| matches!(s.op, FusedOp::Unary(UnOp::Recip))),
            "missing Recip step: {region:?}"
        );
    }
    /// Card 630: `silu` reads its input twice (`x / (1 + exp(-x))`), so the producer of `x` fuses only
    /// through the diamond rule: every read of its value is inside one region. `add -> silu` is then one
    /// dispatch, as it was when `Silu` was a primitive. Mutation: delete the diamond loop in `fuse`; the
    /// `add` stays outside and this row sees two equations.
    #[test]
    fn a_producer_read_twice_inside_one_region_fuses() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, 8]));
        let y = b.constant("y", TensorType::f32(vec![4, 8]));
        let sum = b.binary(poot_graph_ir::BinOp::Add, x, y);
        let out = poot_graph_ir::ops::silu(&b, sum);
        let graph = b.finish(out);

        let fused = fuse(&graph);
        let region = single_fused_region(&fused);
        assert_eq!(
            region.steps.len(),
            5,
            "add + the four silu steps: {region:?}"
        );
    }

    /// The diamond rule's other half: a value also read outside the region (here by a `Reshape`) still
    /// escapes, so its producer stays a plain equation. Mutation: drop the `all(..)` reader check; the
    /// `add` is absorbed although the `Reshape` still reads it, and the plain `Add` count goes to zero.
    #[test]
    fn a_producer_read_outside_the_region_does_not_fuse() {
        let b = Builder::new();
        let x = b.constant("x", TensorType::f32(vec![4, 8]));
        let y = b.constant("y", TensorType::f32(vec![4, 8]));
        let sum = b.binary(poot_graph_ir::BinOp::Add, x, y);
        let act = b.reshape(poot_graph_ir::ops::silu(&b, sum), vec![32]);
        let out = b.binary(poot_graph_ir::BinOp::Mul, act, b.reshape(sum, vec![32]));
        let graph = b.finish(out);

        let fused = fuse(&graph);
        fused.validate().expect("fused graph stays valid");
        let plain_adds = fused
            .eqns
            .iter()
            .filter(|e| matches!(e.op, OpKind::Binary(poot_graph_ir::BinOp::Add)))
            .count();
        assert_eq!(plain_adds, 1, "the escaping add stays outside: {fused:?}");
    }
}
