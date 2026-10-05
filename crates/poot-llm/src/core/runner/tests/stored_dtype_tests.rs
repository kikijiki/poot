//! Card 1008: a traced graph reaches the oracle and the executors with every dense const at the dtype the
//! loader stores. The fixture stores each family's checkpoint consts the way a BF16 or F16 safetensors
//! loader keeps them (the stored words, every tensor 16-bit; only the loader's computed RoPE tables stay F32)
//! and binds the family's decode and stateless prefill graphs through `Runner::bind_storage`.

use super::super::weights::StoredDtypeError;
use super::super::*;
use crate::core::decode_arch::{DecodeArch, fixtures::runner_for};

use poot_graph_ir::{Graph, OpKind, Operand, Storage, ValueId};
use poot_tensor::{DType, HostTensor};

const CAP: usize = 12;
const PROMPT: usize = 4;

/// The consts a loader stores as checkpoint tensors: every named const except the RoPE tables it computes.
fn checkpoint_consts(g: &Graph) -> Vec<(String, poot_graph_ir::TensorType)> {
    g.consts
        .iter()
        .map(|&id| g.meta(id))
        .filter(|meta| meta.storage == Storage::Const)
        .filter_map(|meta| Some((meta.name.clone()?, meta.aval.clone())))
        .filter(|(name, aval)| !name.starts_with("rope.") && aval.dtype == DType::F32)
        .collect()
}

fn zeros(shape: &[usize], dtype: DType) -> HostTensor {
    let n = shape.iter().product();
    match dtype {
        DType::BF16 => HostTensor::bf16(shape.to_vec(), vec![0; n]),
        DType::F16 => HostTensor::f16(shape.to_vec(), vec![0; n]),
        other => panic!("fixture stores 16-bit words, not {other}"),
    }
}

/// The typed cause of a refused bind: every `RunnerError::Context` those paths build wraps a
/// [`StoredDtypeError`].
pub(super) fn stored_dtype_error(error: &crate::error::RunnerError) -> &StoredDtypeError {
    match error {
        crate::error::RunnerError::Context { source, .. } => source
            .downcast_ref::<StoredDtypeError>()
            .unwrap_or_else(|| panic!("not a StoredDtypeError: {source}")),
        other => panic!("not a context-wrapped StoredDtypeError: {other}"),
    }
}

type GraphOf = fn(&Runner) -> crate::error::Result<Graph>;

fn decode(runner: &Runner) -> crate::error::Result<Graph> {
    runner.decode_masked_graph(CAP)
}

fn prefill(runner: &Runner) -> crate::error::Result<Graph> {
    runner.stateless_prefill_graph(PROMPT)
}

/// How a fixture stores each checkpoint const.
#[derive(Clone, Copy)]
enum Store {
    /// Every const as `dtype` words (a BF16 or F16 safetensors checkpoint).
    All(DType),
    /// 1-D consts F32, everything else `dtype`: a GGUF-style checkpoint whose norms and biases are F32.
    NormsF32(DType),
    /// Every const F32: the stored dtype is the declared one (a gemma folded `+1` norm is stored F32).
    AllF32,
}

impl Store {
    fn dtype_of(self, aval: &poot_graph_ir::TensorType) -> DType {
        match self {
            Store::All(d) => d,
            Store::NormsF32(d) if aval.shape.len() > 1 => d,
            Store::NormsF32(_) | Store::AllF32 => DType::F32,
        }
    }
}

/// The consts of `g` a contraction reads, directly or through views, concats and casts, derived here
/// independently of the transform.
fn contraction_consts(g: &Graph) -> Vec<ValueId> {
    let reads = |eqn: &poot_graph_ir::Eqn, v: ValueId| {
        eqn.inputs
            .iter()
            .any(|o| matches!(o, Operand::Value(x) if *x == v))
    };
    g.consts
        .iter()
        .copied()
        .filter(|&root| {
            let mut seen = vec![root];
            let mut i = 0;
            while i < seen.len() {
                let v = seen[i];
                i += 1;
                for eqn in g.eqns.iter().filter(|eqn| reads(eqn, v)) {
                    match eqn.op {
                        OpKind::MatMul
                        | OpKind::MatMulBias
                        | OpKind::IndexedMatMul
                        | OpKind::DenseContraction { .. } => return true,
                        OpKind::Reshape { .. }
                        | OpKind::Transpose { .. }
                        | OpKind::Slice { .. }
                        | OpKind::Concat { .. }
                        | OpKind::Broadcast { .. }
                        | OpKind::Cast { .. }
                            if !seen.contains(&eqn.out) =>
                        {
                            seen.push(eqn.out);
                        }
                        _ => {}
                    }
                }
            }
            false
        })
        .collect()
}

/// `arch`'s graph with its F32-declared checkpoint consts stored as `store` binds typed: each const the loader
/// stores reaches the oracle at its stored dtype, no `Cast` reads a const a contraction reads, a const stored
/// as declared is left as is, and (for a 16-bit store) the same graph before the bind is refused by name.
fn assert_family_binds_its_stored_dtypes(arch: DecodeArch, graph_of: GraphOf, store: Store) {
    let mut runner = runner_for(arch);
    let raw = graph_of(&runner).expect("trace with no weights: nothing to retype");
    let consts = checkpoint_consts(&raw);
    assert!(
        !consts.is_empty(),
        "{arch:?}: the fixture found no checkpoint const"
    );
    for (name, aval) in &consts {
        let dtype = store.dtype_of(aval);
        let tensor = match dtype {
            DType::F32 => HostTensor::f32(aval.shape.clone(), vec![0.0; aval.numel()]),
            other => zeros(&aval.shape, other),
        };
        runner.weights.insert(name.clone(), tensor.into());
    }
    let mismatched: Vec<_> = consts
        .iter()
        .filter(|(_, aval)| store.dtype_of(aval) != DType::F32)
        .collect();
    assert!(
        mismatched
            .iter()
            .all(|(name, aval)| runner.weight_value(name, aval).is_err()),
        "{arch:?}: every 16-bit-stored const is a stored/declared mismatch before the bind"
    );

    let bound = graph_of(&runner).unwrap_or_else(|e| panic!("{arch:?}: {e}"));
    for (name, aval) in &consts {
        let id = bound
            .consts
            .iter()
            .copied()
            .find(|&id| bound.meta(id).name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("{arch:?}: {name} is still a const"));
        let meta = bound.meta(id);
        assert_eq!(meta.aval.dtype, store.dtype_of(aval), "{arch:?}: {name}");
        runner
            .weight_value(name, &meta.aval)
            .unwrap_or_else(|e| panic!("{arch:?}: oracle bind of {name}: {e}"));
    }
    let reaching = contraction_consts(&bound);
    assert!(
        !reaching.is_empty(),
        "{arch:?}: the fixture found no contraction weight"
    );
    let cast = cast_sources(&bound);
    for id in reaching {
        let name = bound.meta(id).name.clone().unwrap_or_default();
        assert!(
            !cast.contains(&name),
            "{arch:?}: a Cast reads contraction weight {name}"
        );
    }
    if matches!(store, Store::AllF32) {
        assert_eq!(
            bound.eqns.len(),
            raw.eqns.len(),
            "{arch:?}: a graph stored as declared is untouched"
        );
    }
}

/// `arch`'s graph with every checkpoint const stored as `store` words, bound through `Runner::bind_storage`.
fn stored_graph(arch: DecodeArch, graph_of: GraphOf, store: Store) -> Graph {
    let mut runner = runner_for(arch);
    let raw = graph_of(&runner).expect("trace with no weights: nothing to retype");
    for (name, aval) in checkpoint_consts(&raw) {
        let tensor = match store.dtype_of(&aval) {
            DType::F32 => HostTensor::f32(aval.shape.clone(), vec![0.0; aval.numel()]),
            other => zeros(&aval.shape, other),
        };
        runner.weights.insert(name, tensor.into());
    }
    graph_of(&runner).unwrap_or_else(|e| panic!("{arch:?}: {e}"))
}

/// Card 1011 acceptance: a BF16-stored const never reaches an F32 lane. Every family's decode and prefill
/// graph, bound with BF16-stored checkpoint consts, goes through the target preparation `compile` runs on
/// wgpu, ROCm and PTX; every const that is BF16 after the bind is still BF16 after it. (A const the plan
/// retyped to F32 would upload through the executor's widening arm, which no longer exists.)
///
/// Mutation: retype every BF16 const to F32 at the top of `widen_mismatched_matmul_dtypes`; the first family
/// goes red naming the const and the backend.
#[test]
fn no_family_graph_has_a_bf16_const_retyped_by_target_preparation() {
    let backends = [
        poot_target::Backend::SpirvVulkan,
        poot_target::Backend::AmdGcn(poot_target::AmdArch::gfx1151()),
        poot_target::Backend::Nvptx,
    ];
    for &arch in DecodeArch::ALL {
        for graph_of in [decode as GraphOf, prefill as GraphOf] {
            let bound = stored_graph(arch, graph_of, Store::All(DType::BF16));
            let narrow: Vec<(String, ValueId)> = bound
                .consts
                .iter()
                .copied()
                .filter(|&id| bound.aval(id).dtype == DType::BF16)
                .filter_map(|id| Some((bound.meta(id).name.clone()?, id)))
                .collect();
            assert!(
                !narrow.is_empty(),
                "{arch:?}: the fixture binds BF16 consts"
            );
            for backend in backends {
                let caps = poot_test_util::device_caps::default_caps_for(backend);
                let prepared = poot_graph_plan::prepare_target_graph(&bound, backend, &caps);
                for (name, id) in &narrow {
                    let prepared_id = prepared
                        .consts
                        .iter()
                        .copied()
                        .find(|&c| prepared.meta(c).name.as_deref() == Some(name))
                        .unwrap_or_else(|| {
                            panic!("{arch:?}: {name} is still a const on {backend:?}")
                        });
                    assert_eq!(
                        prepared.aval(prepared_id).dtype,
                        DType::BF16,
                        "{arch:?} on {backend:?}: const {name} (v{id}) was retyped"
                    );
                }
            }
        }
    }
}

#[test]
fn every_family_decode_graph_binds_its_stored_bf16_dtypes() {
    for &arch in DecodeArch::ALL {
        assert_family_binds_its_stored_dtypes(arch, decode, Store::All(DType::BF16));
    }
}

#[test]
fn every_family_decode_graph_binds_its_stored_f16_dtypes() {
    for &arch in DecodeArch::ALL {
        assert_family_binds_its_stored_dtypes(arch, decode, Store::All(DType::F16));
    }
}

#[test]
fn every_family_prefill_graph_binds_its_stored_bf16_dtypes() {
    for &arch in DecodeArch::ALL {
        assert_family_binds_its_stored_dtypes(arch, prefill, Store::All(DType::BF16));
    }
}

#[test]
fn every_family_binds_f32_norms_beside_16_bit_projections() {
    for &arch in DecodeArch::ALL {
        assert_family_binds_its_stored_dtypes(arch, decode, Store::NormsF32(DType::BF16));
        assert_family_binds_its_stored_dtypes(arch, prefill, Store::NormsF32(DType::F16));
    }
}

#[test]
fn every_family_graph_stored_as_declared_binds_untouched() {
    for &arch in DecodeArch::ALL {
        assert_family_binds_its_stored_dtypes(arch, decode, Store::AllF32);
        assert_family_binds_its_stored_dtypes(arch, prefill, Store::AllF32);
    }
}

fn with_weights(weights: &[(&str, &[usize], DType)]) -> Runner {
    let mut runner = runner_for(DecodeArch::Mixtral);
    for &(name, shape, dtype) in weights {
        let tensor = match dtype {
            DType::F32 => HostTensor::f32(shape.to_vec(), vec![0.0; shape.iter().product()]),
            other => zeros(shape, other),
        };
        runner.weights.insert(name.to_string(), tensor.into());
    }
    runner
}

/// The names of the consts a `Cast` equation of `g` reads directly.
fn cast_sources(g: &Graph) -> Vec<String> {
    g.eqns
        .iter()
        .filter(|eqn| matches!(eqn.op, OpKind::Cast { .. }))
        .filter_map(|eqn| match eqn.inputs.first() {
            Some(Operand::Value(v)) => g.meta(*v).name.clone(),
            _ => None,
        })
        .collect()
}

/// Card 1008 acceptance (and Card 1007's review row): a const a contraction reads is never cast or widened. A
/// BF16-stored head declared F32 is read by its matmul at the stored dtype through the transpose, with no
/// `Cast` over it, while the norm scale beside it (an elementwise read) gets its explicit cast.
///
/// Mutation: make `contraction_views` return `None`; the head takes the cast arm, a `Cast` reads `w.head`, and
/// this row goes red.
#[test]
fn a_contraction_weight_keeps_its_stored_dtype_and_is_never_cast() {
    let (n, k) = (3, 4);
    let b = poot_graph_ir::Builder::new();
    let x = b.constant("x", poot_graph_ir::TensorType::f32(vec![2, k]));
    let head = b.constant("w.head", poot_graph_ir::TensorType::f32(vec![n, k]));
    let norm = b.constant("w.norm", poot_graph_ir::TensorType::f32(vec![n]));
    let y = b.matmul(x, b.transpose(head, vec![1, 0]));
    let out = b.binary(poot_graph_ir::BinOp::Add, y, b.broadcast(norm, vec![2, n]));
    let g = b.finish(out);
    let runner = with_weights(&[
        ("w.head", &[n, k], DType::BF16),
        ("w.norm", &[n], DType::BF16),
    ]);
    let bound = runner.bind_storage(g).expect("both consts bind");

    let dtype_of = |name: &str| {
        let id = bound
            .consts
            .iter()
            .copied()
            .find(|&id| bound.meta(id).name.as_deref() == Some(name))
            .unwrap_or_else(|| panic!("{name} is still a const"));
        bound.meta(id).aval.dtype
    };
    assert_eq!(dtype_of("w.head"), DType::BF16);
    assert_eq!(dtype_of("w.norm"), DType::BF16);
    assert_eq!(
        cast_sources(&bound),
        ["w.norm"],
        "only the elementwise norm scale is cast"
    );
}

/// A token embedding stays stored: the gather reads the BF16 table and only the gathered rows widen, through a
/// cast after the gather.
///
/// Mutation: read the whole gather table through a cast (drop the cast-after-gather arm in
/// `bind_stored_dtypes`); a `Cast` then reads `w.embed` and this row goes red.
#[test]
fn an_embedding_table_stays_stored_and_only_the_gathered_rows_widen() {
    let (vocab, w) = (6, 4);
    let b = poot_graph_ir::Builder::new();
    let table = b.constant("w.embed", poot_graph_ir::TensorType::f32(vec![vocab, w]));
    let token = b.slot(
        poot_graph_ir::Slot::Token,
        poot_graph_ir::TensorType::scalar(DType::I32),
    );
    let rows = b.gather_scalar(table, 0, token);
    let g = b.finish(rows);
    let runner = with_weights(&[("w.embed", &[vocab, w], DType::BF16)]);
    let bound = runner.bind_storage(g).expect("the embedding binds");

    assert!(
        cast_sources(&bound).is_empty(),
        "no cast reads the table: {:?}",
        cast_sources(&bound)
    );
    let gather = bound
        .eqns
        .iter()
        .position(|eqn| matches!(eqn.op, OpKind::Gather { .. }))
        .expect("the gather stays");
    assert_eq!(bound.aval(bound.eqns[gather].out).dtype, DType::BF16);
    let widen = &bound.eqns[gather + 1];
    assert!(matches!(widen.op, OpKind::Cast { to: DType::F32 }));
    assert_eq!(widen.out, bound.output);
}

/// Anything but a stored BF16/F16 const declared F32 is a typed refusal naming the const and both dtypes.
///
/// Mutation: let `bind_stored_dtypes` skip a mismatching const (`continue` instead of the `Mismatch` error);
/// the bind succeeds and this row goes red.
#[test]
fn every_other_stored_declared_disagreement_is_refused_by_name() {
    for (stored, declared) in [(DType::F32, DType::BF16), (DType::F16, DType::BF16)] {
        let b = poot_graph_ir::Builder::new();
        let w = b.constant("w.scale", poot_graph_ir::TensorType::new(vec![3], declared));
        let g = b.finish(w);
        let runner = with_weights(&[("w.scale", &[3], stored)]);
        let error = runner
            .bind_storage(g)
            .expect_err("a narrowing or re-encoding disagreement is refused");
        assert!(
            matches!(
                stored_dtype_error(&error),
                StoredDtypeError::Mismatch { name, stored: s, declared: d }
                    if name == "w.scale" && *s == stored && *d == declared
            ),
            "{error}"
        );
    }
}

/// A const read both by a contraction and by a gather (a tied embedding and head sharing one name) cannot keep
/// one dtype for both readers: the bind refuses it with the typed graph error instead of widening either side.
///
/// Mutation: make `contraction_views` return `None`; the whole const takes the cast arm and the bind succeeds,
/// so this row goes red.
#[test]
fn a_const_read_by_a_contraction_and_a_gather_is_a_typed_refusal() {
    let (vocab, w) = (6, 4);
    let b = poot_graph_ir::Builder::new();
    let tied = b.constant("w.tied", poot_graph_ir::TensorType::f32(vec![vocab, w]));
    let token = b.slot(
        poot_graph_ir::Slot::Token,
        poot_graph_ir::TensorType::scalar(DType::I32),
    );
    let row = b.gather_scalar(tied, 0, token);
    let row = b.reshape(row, vec![1, w]);
    let logits = b.matmul(row, b.transpose(tied, vec![1, 0]));
    let g = b.finish(logits);
    let runner = with_weights(&[("w.tied", &[vocab, w], DType::BF16)]);
    let error = runner
        .bind_storage(g)
        .expect_err("one const cannot serve both readers");
    assert!(
        matches!(
            stored_dtype_error(&error),
            StoredDtypeError::Graph { consts, .. } if consts == &["w.tied".to_string()]
        ),
        "{error}"
    );
}
