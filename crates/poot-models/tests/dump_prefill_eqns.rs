//! Throwaway diagnostic (spec 023): print the cse'd prefill graph eqns around the V-bias add (eqn 14) with their
//! input value-ids and each input's producing eqn/op, to verify the wiring the PTX runtime executes.
use poot_executor_parity::dense::{Dense, Family, plain, step};
use poot_graph_ir::Operand;
use poot_models::model::{LogitRows, Phase};

#[test]
#[ignore = "diagnostic"]
fn dump_eqns() {
    let dense = Dense::new(Family::Qwen2)
        .vocab(151936)
        .dims(896, 4864, 24)
        .heads(14, 2)
        .head_dim(64)
        .max_positions(32768);
    let model = dense.model();
    let g0 = plain(
        model
            .model
            .trace(Phase::Prefill, step(1, 864, 896, LogitRows::Last))
            .unwrap(),
    );
    let g = poot_graph_plan::cse(&g0);
    let mut producer = std::collections::HashMap::new();
    for (i, e) in g.eqns.iter().enumerate() {
        producer.insert(e.out, (i, e.op.name()));
    }
    for (i, e) in g.eqns.iter().enumerate().take(17).skip(8) {
        let ins: Vec<String> = e
            .inputs
            .iter()
            .map(|op| match op {
                Operand::Value(v) => {
                    let sh = g.aval(*v).shape.clone();
                    let st = g.meta(*v).storage;
                    match producer.get(v) {
                        Some((pi, pn)) => format!("v{v}(eqn{pi}:{pn},{sh:?})"),
                        None => format!("v{v}({st:?},{sh:?})"),
                    }
                }
                Operand::Lit(s) => format!("lit{s:?}"),
            })
            .collect();
        eprintln!(
            "eqn {i:>3} {} out=v{} {:?} <- [{}]",
            e.op.name(),
            e.out,
            g.aval(e.out).shape,
            ins.join(", ")
        );
    }
}
