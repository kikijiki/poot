use super::*;

/// Common-subexpression elimination: collapse structurally identical eqns (same op + same operands)
/// to a single value, rewiring later uses. Pure and order-preserving. On a transformer trace this
/// matters: e.g. the RoPE `gather(cos_table, pos)` is recomputed in every layer's q and k rope, all
/// referencing the same const + slot, so CSE folds them to one.
///
/// The value table is preserved (dead entries are harmless); only the eqn list shrinks and the
/// output is remapped to its canonical value.
pub fn cse<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    let mut remap: HashMap<usize, usize> = HashMap::new(); // old value id -> canonical id
    let mut seen: HashMap<String, usize> = HashMap::new(); // eqn key -> canonical out id
    let mut eqns: Vec<Eqn> = Vec::with_capacity(g.eqns.len());

    let resolve = |o: &Operand, remap: &HashMap<usize, usize>| -> Operand {
        match o {
            Operand::Value(v) => Operand::Value(remap.get(v).copied().unwrap_or(*v)),
            Operand::Lit(s) => Operand::Lit(*s),
        }
    };

    for eqn in &g.eqns {
        let inputs: Vec<Operand> = eqn.inputs.iter().map(|o| resolve(o, &remap)).collect();
        // a deterministic structural key: the op (params included) + the resolved operands.
        let key = format!("{:?}|{:?}", eqn.op, inputs);
        if let Some(&canon) = seen.get(&key) {
            remap.insert(eqn.out, canon);
        } else {
            seen.insert(key, eqn.out);
            eqns.push(Eqn {
                op: eqn.op.clone(),
                inputs,
                out: eqn.out,
                layer: eqn.layer,
            });
        }
    }

    let canon = |v: usize| remap.get(&v).copied().unwrap_or(v);
    let mut out = Graph { eqns, ..g.clone() };
    out.remap_results(canon);
    out
}

/// Dead-code elimination: drop eqns whose output is never (transitively) used by the graph output.
/// Eqns are topologically ordered, so a single reverse liveness sweep suffices.
pub fn dce<V: ValidationChannel>(g: &Graph<V>) -> Graph<V> {
    dce_with_roots(g, &[])
}

/// Dead-code elimination with additional caller-owned liveness roots.
///
/// Extra roots are a transform hint only. They do not change [`Graph::output`] or carried-state
/// semantics and must be re-proved by the caller after this pass.
pub fn dce_with_roots<V: ValidationChannel>(g: &Graph<V>, extra_roots: &[ValueId]) -> Graph<V> {
    let mut live = vec![false; g.values.len()];
    for root in g.liveness_roots() {
        live[root] = true;
    }
    for &root in extra_roots {
        if let Some(live) = live.get_mut(root) {
            *live = true;
        }
    }
    for eqn in g.eqns.iter().rev() {
        if live[eqn.out] {
            for o in &eqn.inputs {
                if let Operand::Value(v) = o {
                    live[*v] = true;
                }
            }
        }
    }
    let eqns: Vec<Eqn> = g.eqns.iter().filter(|e| live[e.out]).cloned().collect();
    Graph { eqns, ..g.clone() }
}
