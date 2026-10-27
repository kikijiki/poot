//! POOT-1156: where poot and llama.cpp part ways, site by site through one forward pass.
//!
//! The teacher-forced gate (`llamacpp_kl_tests`) first measured a mean KL of 0.032 between poot and
//! llama.cpp, twenty times llama.cpp's own backend noise. This module bisects such a gap: both engines decode
//! the same token list, and the tensors they hold at the points both expose (the embedding, the residual
//! stream after each decoder layer, the logits) are compared into a typed [`Bisect`].
//!
//! poot's side is the CPU oracle (`poot_eval`) on the traced, packed-bound graph: f32 activations over the
//! dequantized weights, the same graph every executor runs. llama.cpp's side is `llamacpp-dump`
//! (`scripts/llamacpp-dump`), which decodes on the CPU and writes every graph tensor of `llama_decode`.
//!
//! Two llama.cpp runs are the references. The Q8_0 file itself multiplies its Q8_0 weights with the
//! activations quantized to Q8_0 (ggml's `vec_dot_type` for `Q8_0`), as every Q8_0 run does. An F16 copy of
//! the same weights (`llama-quantize --allow-requantize ... F16`) multiplies F16 with F16 and has no
//! activation quantization, so it is the near-exact reference for what poot computes.

use std::collections::HashMap;
use std::fmt;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use poot_eval::{EvalOptions, Value, eval, materialize_dense};
use poot_graph_ir::{PackedSourceName, Storage, ValueId};
use poot_graph_plan::{WeightFormats, bind_packed_weights};
use poot_models::model::{LogitRows, StepShape, StepSpec};
use poot_models::test_support::weight_map::Step;
use poot_quant::PackedComponentRef;
use poot_quant::weights::{WeightEntry, WeightStore};
use poot_tensor::HostTensor;
use poot_weights::WeightId;

use super::{ModelHandle, serving_trace_limits};

/// A point in the forward pass that poot and llama.cpp both expose.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Site {
    /// The token embeddings, before the first layer.
    Embedding,
    /// The residual stream after decoder layer `n`.
    LayerOutput(usize),
    /// The logits of every position.
    Logits,
}

impl Site {
    /// The name llama.cpp gives the graph node holding this site.
    fn llama_name(self) -> String {
        match self {
            Site::Embedding => "embd".to_string(),
            Site::LayerOutput(n) => format!("l_out-{n}"),
            Site::Logits => "result_output".to_string(),
        }
    }
}

impl fmt::Display for Site {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Site::Embedding => f.write_str("embedding"),
            Site::LayerOutput(n) => write!(f, "layer {n}"),
            Site::Logits => f.write_str("logits"),
        }
    }
}

/// Why two engines' tensors could not be compared, or a dump could not be read.
#[derive(Debug, thiserror::Error)]
enum BisectError {
    #[error("{0}: {1}")]
    Io(PathBuf, #[source] std::io::Error),
    #[error("index.tsv line {line}: {reason}")]
    Index { line: usize, reason: &'static str },
    #[error("llama.cpp's dump holds no tensor named {0}")]
    Missing(String),
    #[error("llama.cpp's tensor {name} holds {got} values, its shape says {want}")]
    DumpSize {
        name: String,
        got: usize,
        want: usize,
    },
    #[error("{site}: poot holds {poot} values, llama.cpp {reference}")]
    Length {
        site: Site,
        poot: usize,
        reference: usize,
    },
}

/// A directory written by `llamacpp-dump`: the tensors of one `llama_decode`, widened to f32.
struct LlamaDump {
    dir: PathBuf,
    /// `(sequence number, name, element count)`, in execution order.
    tensors: Vec<(usize, String, usize)>,
}

impl LlamaDump {
    fn read(dir: &Path) -> Result<Self, BisectError> {
        let path = dir.join("index.tsv");
        let index =
            std::fs::read_to_string(&path).map_err(|source| BisectError::Io(path, source))?;
        let mut tensors = Vec::new();
        for (n, line) in index.lines().enumerate() {
            let line_no = n + 1;
            let fields: Vec<&str> = line.split('\t').collect();
            let [seq, name, _op, dims @ ..] = fields.as_slice() else {
                return Err(BisectError::Index {
                    line: line_no,
                    reason: "fewer than three fields",
                });
            };
            let seq = seq.parse().map_err(|_| BisectError::Index {
                line: line_no,
                reason: "the sequence number is not an integer",
            })?;
            let [d0, d1, d2, d3] = dims else {
                return Err(BisectError::Index {
                    line: line_no,
                    reason: "a tensor has four dimensions",
                });
            };
            let mut elements = 1usize;
            for d in [d0, d1, d2, d3] {
                elements *= d.parse::<usize>().map_err(|_| BisectError::Index {
                    line: line_no,
                    reason: "a dimension is not an integer",
                })?;
            }
            tensors.push((seq, (*name).to_string(), elements));
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            tensors,
        })
    }

    /// The values of the last tensor llama.cpp named `name`.
    fn tensor(&self, name: &str) -> Result<Vec<f32>, BisectError> {
        let &(seq, _, elements) = self
            .tensors
            .iter()
            .rev()
            .find(|(_, n, _)| n == name)
            .ok_or_else(|| BisectError::Missing(name.to_string()))?;
        let path = self.dir.join(format!("{seq}.f32"));
        let bytes = std::fs::read(&path).map_err(|source| BisectError::Io(path, source))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        if values.len() != elements {
            return Err(BisectError::DumpSize {
                name: name.to_string(),
                got: values.len(),
                want: elements,
            });
        }
        Ok(values)
    }
}

/// How far poot's tensor at one [`Site`] is from the reference's.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SiteAgreement {
    site: Site,
    /// Root mean square of the element-wise difference.
    rmse: f64,
    /// Root mean square of the reference.
    rms_reference: f64,
    /// Largest element-wise difference.
    max_abs: f64,
}

impl SiteAgreement {
    fn between(site: Site, poot: &[f32], reference: &[f32]) -> Result<Self, BisectError> {
        if poot.len() != reference.len() {
            return Err(BisectError::Length {
                site,
                poot: poot.len(),
                reference: reference.len(),
            });
        }
        let n = poot.len() as f64;
        let (mut squares, mut reference_squares, mut max_abs) = (0.0f64, 0.0f64, 0.0f64);
        for (&p, &r) in poot.iter().zip(reference) {
            let d = f64::from(p) - f64::from(r);
            squares += d * d;
            reference_squares += f64::from(r) * f64::from(r);
            max_abs = max_abs.max(d.abs());
        }
        Ok(Self {
            site,
            rmse: (squares / n).sqrt(),
            rms_reference: (reference_squares / n).sqrt(),
            max_abs,
        })
    }

    /// The RMS difference relative to the RMS of the reference.
    fn relative(&self) -> f64 {
        self.rmse / self.rms_reference
    }
}

/// The agreement at every site, in forward order.
#[derive(Clone, Debug, PartialEq)]
struct Bisect {
    rows: Vec<SiteAgreement>,
}

impl Bisect {
    /// Compare `poot`'s tensors with `reference`'s, site by site.
    fn of(
        poot: &[(Site, Vec<f32>)],
        reference: impl Fn(Site) -> Result<Vec<f32>, BisectError>,
    ) -> Result<Self, BisectError> {
        let rows = poot
            .iter()
            .map(|(site, values)| SiteAgreement::between(*site, values, &reference(*site)?))
            .collect::<Result<_, _>>()?;
        Ok(Self { rows })
    }

    /// The first site whose relative difference exceeds `limit`; a NaN difference exceeds every limit.
    fn first_beyond(&self, limit: f64) -> Option<&SiteAgreement> {
        self.rows
            .iter()
            .find(|row| row.relative().is_nan() || row.relative() > limit)
    }

    fn at(&self, site: Site) -> &SiteAgreement {
        self.rows
            .iter()
            .find(|row| row.site == site)
            .unwrap_or_else(|| panic!("no row for {site}"))
    }
}

impl fmt::Display for Bisect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<10} {:>10} {:>10} {:>9} {:>10}",
            "site", "rmse", "rms ref", "relative", "max abs"
        )?;
        for row in &self.rows {
            writeln!(
                f,
                "{:<10} {:>10.4e} {:>10.4e} {:>9.5} {:>10.4e}",
                row.site.to_string(),
                row.rmse,
                row.rms_reference,
                row.relative(),
                row.max_abs
            )?;
        }
        Ok(())
    }
}

/// The value of every input of the traced, packed-bound prefill `graph` of `ids`: dense weights as f32 host
/// tensors, packed weights by reference, zeroed state, computed constants and the step's slots.
pub(super) fn oracle_inputs(
    handle: &ModelHandle,
    graph: &poot_graph_ir::Graph,
    ids: &[i32],
) -> HashMap<ValueId, Value> {
    let model = handle.model();
    let (store, map) = (handle.shared_store(), model.weights());
    let ids_by_const: HashMap<String, WeightId> =
        map.iter().map(|(id, _, _)| (id.const_name(), id)).collect();
    let mut inputs: HashMap<ValueId, Value> = HashMap::new();
    for &id in graph.inputs() {
        let meta = graph.meta(id);
        let value: Value = match meta.storage {
            Storage::Const => {
                let name = meta.name().expect("a const has a name");
                let name = name.as_ref();
                match ids_by_const.get(name) {
                    Some(&weight) => {
                        let WeightEntry::Dense(dense) = map.materialize(weight, store).unwrap()
                        else {
                            panic!("{name} is packed but the graph declares it dense")
                        };
                        let mut one = WeightStore::builder();
                        one.insert("w", WeightEntry::Dense(dense)).unwrap();
                        Value::Host(materialize_dense(&one.build(), "w").unwrap())
                    }
                    None => {
                        let source = PackedSourceName::parse(name)
                            .unwrap_or_else(|| panic!("the map names no const {name}"));
                        let weight = ids_by_const[source.linear_id()];
                        let WeightEntry::Packed(payload) = map.materialize(weight, store).unwrap()
                        else {
                            panic!("{name} names a dense weight")
                        };
                        Value::Packed(PackedComponentRef::new(Arc::clone(&payload), source.role()))
                    }
                }
            }
            Storage::State => HostTensor::zeros_of(meta.aval.dtype, meta.aval.shape.clone()).into(),
            Storage::Computed(c) => c.tensor().into(),
            Storage::Slot(_) => continue,
            Storage::Device => unreachable!("a graph input is never a device intermediate"),
        };
        inputs.insert(id, value);
    }
    inputs.extend(slot_inputs(graph, ids));
    inputs
}

/// The step slots of `ids` as values, by the graph input each fills: what changes between two windows over
/// the same graph, so a caller keeps the weights of [`oracle_inputs`] and replaces only these.
pub(super) fn slot_inputs(graph: &poot_graph_ir::Graph, ids: &[i32]) -> Vec<(ValueId, Value)> {
    let step = Step::new(ids, 0);
    let slots = step.slots();
    graph
        .inputs()
        .iter()
        .filter_map(|&id| {
            let meta = graph.meta(id);
            let Storage::Slot(_) = meta.storage else {
                return None;
            };
            let key = meta.slot_key().expect("a slot has a key");
            let (_, tensor) = slots.iter().find(|(k, _)| k == key).expect("slot value");
            Some((id, tensor.clone().into()))
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Model-free: the dump format, the metric and the bisection, on tensors built in the test.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod format_tests {
    use super::*;

    fn dump_dir(index: &str, files: &[(usize, Vec<f32>)]) -> poot_test_util::UniqueTempPath {
        let dir = poot_test_util::unique_temp_path("dump");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("index.tsv"), index).unwrap();
        for (seq, values) in files {
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            std::fs::write(dir.join(format!("{seq}.f32")), bytes).unwrap();
        }
        dir
    }

    #[test]
    fn a_dump_reads_back_the_last_tensor_of_a_name() {
        let dir = dump_dir(
            "0\tembd\tGET_ROWS\t2\t2\t1\t1\n1\tnorm-0\tRMS_NORM\t2\t1\t1\t1\n2\tnorm-0\tRMS_NORM\t2\t1\t1\t1\n",
            &[
                (0, vec![1.0, 2.0, 3.0, 4.0]),
                (1, vec![5.0, 6.0]),
                (2, vec![7.0, 8.0]),
            ],
        );
        let dump = LlamaDump::read(&dir).unwrap();
        assert_eq!(dump.tensor("embd").unwrap(), [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(dump.tensor("norm-0").unwrap(), [7.0, 8.0]);
        assert!(matches!(
            dump.tensor("l_out-3"),
            Err(BisectError::Missing(name)) if name == "l_out-3"
        ));
    }

    #[test]
    fn a_malformed_dump_is_a_typed_error() {
        let short = dump_dir(
            "0\tembd\tGET_ROWS\t2\t2\t1\t1\n",
            &[(0, vec![1.0, 2.0, 3.0])],
        );
        assert!(matches!(
            LlamaDump::read(&short).unwrap().tensor("embd"),
            Err(BisectError::DumpSize {
                got: 3,
                want: 4,
                ..
            })
        ));
        let ragged = dump_dir("0\tembd\tGET_ROWS\t2\t2\n", &[]);
        assert!(matches!(
            LlamaDump::read(&ragged),
            Err(BisectError::Index { line: 1, .. })
        ));
        let text = dump_dir("0\tembd\tGET_ROWS\t2\ttwo\t1\t1\n", &[]);
        assert!(matches!(
            LlamaDump::read(&text),
            Err(BisectError::Index { line: 1, .. })
        ));
        let empty = poot_test_util::unique_temp_path("empty");
        assert!(matches!(LlamaDump::read(&empty), Err(BisectError::Io(..))));
    }

    #[test]
    fn agreement_is_the_rms_difference_over_the_rms_reference() {
        let a = SiteAgreement::between(Site::Logits, &[1.0, 2.0, 4.0], &[1.0, 2.0, 2.0]).unwrap();
        assert!((a.rmse - (4.0f64 / 3.0).sqrt()).abs() < 1e-12);
        assert!((a.rms_reference - 3.0f64.sqrt()).abs() < 1e-12);
        assert_eq!(a.max_abs, 2.0);
        assert!((a.relative() - (4.0f64 / 9.0).sqrt()).abs() < 1e-12);
        assert!(matches!(
            SiteAgreement::between(Site::Embedding, &[1.0], &[1.0, 2.0]),
            Err(BisectError::Length {
                site: Site::Embedding,
                poot: 1,
                reference: 2
            })
        ));
    }

    /// Mutation: report the first row whose limit is *not* exceeded (negate the `find` predicate in
    /// `Bisect::first_beyond`); the layer named below is no longer the one found.
    #[test]
    fn the_bisection_names_the_first_site_that_leaves_the_reference() {
        let reference = |_: Site| -> Result<Vec<f32>, BisectError> { Ok(vec![1.0; 4]) };
        // Layers 0 and 1 reproduce the reference, layer 2 drifts by 10%, layer 3 by 50%.
        let at = |drift: f32| vec![1.0 + drift; 4];
        let poot = vec![
            (Site::Embedding, at(0.0)),
            (Site::LayerOutput(0), at(0.0)),
            (Site::LayerOutput(1), at(0.001)),
            (Site::LayerOutput(2), at(0.1)),
            (Site::LayerOutput(3), at(0.5)),
            (Site::Logits, at(f32::NAN)),
        ];
        let report = Bisect::of(&poot, reference).unwrap();
        let first = report.first_beyond(0.01).unwrap();
        assert_eq!(first.site, Site::LayerOutput(2));
        assert!((first.relative() - 0.1).abs() < 1e-6);
        assert_eq!(report.first_beyond(0.2).unwrap().site, Site::LayerOutput(3));
        assert_eq!(report.first_beyond(10.0).unwrap().site, Site::Logits);
        assert_eq!(report.at(Site::Embedding).rmse, 0.0);
        assert!(Bisect { rows: vec![] }.first_beyond(0.0).is_none());
    }
}

// ---------------------------------------------------------------------------------------------
// Checkpoint row: Qwen2.5-0.5B Q8_0 against llama.cpp, layer by layer.
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod checkpoint_rows {
    use super::*;

    /// The first 32 tokens of the KL corpus (`tests/fixtures/kl_corpus.txt`) as llama.cpp tokenizes it
    /// for Qwen2.5, so neither engine's tokenizer is part of the comparison.
    const TOKENS: [i32; 32] = [
        785, 74642, 6290, 315, 647, 783, 3987, 23011, 1380, 279, 14796, 362, 359, 20027, 279,
        18172, 9396, 11, 323, 369, 1429, 315, 279, 1042, 279, 6556, 12033, 448, 30249, 13, 16608,
        59158,
    ];

    /// poot against llama.cpp running an F16 copy of the same weights, per site: the measured worst is
    /// 0.0018 (layer 23), so 0.005 leaves a 2.7x margin and a missing or misplaced op (a wrong RoPE base, a
    /// dropped bias) lands one or two orders of magnitude above it.
    const F16_LIMIT: f64 = 0.005;

    /// The binary `scripts/llamacpp-dump/build.sh` builds: `LLAMACPP_DUMP`, else `llamacpp-dump` on `PATH`.
    fn dump_tool() -> Option<PathBuf> {
        let tool = std::env::var_os("LLAMACPP_DUMP")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("llamacpp-dump"));
        match Command::new(&tool).arg("--help").output() {
            Ok(_) => Some(tool),
            Err(error) => {
                eprintln!(
                    "POOT-SKIP: {} is not runnable ({error}); build it with scripts/llamacpp-dump/build.sh",
                    tool.display()
                );
                None
            }
        }
    }

    fn run(command: &mut Command) {
        let output = command.output().expect("spawn");
        assert!(
            output.status.success(),
            "{command:?} failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    /// llama.cpp's tensors for `tokens` on the model at `gguf`, with the directory that holds them (removed
    /// when it drops).
    fn llama_dump(
        tool: &Path,
        gguf: &Path,
        tokens: &[i32],
    ) -> (poot_test_util::UniqueTempPath, LlamaDump) {
        let scratch = poot_test_util::unique_temp_path("dump");
        std::fs::create_dir_all(&scratch).unwrap();
        let list = scratch.join("tokens.txt");
        let text: Vec<String> = tokens.iter().map(i32::to_string).collect();
        std::fs::write(&list, text.join(" ")).unwrap();
        run(Command::new(tool).arg(gguf).arg(&*scratch).arg(&list));
        let dump = LlamaDump::read(&scratch).expect("read the dump");
        (scratch, dump)
    }

    /// poot's CPU oracle on the traced, packed-bound prefill of `ids`, at every [`Site`].
    fn oracle_sites(handle: &ModelHandle, ids: &[i32]) -> Vec<(Site, Vec<f32>)> {
        let model = handle.model();
        let n = NonZeroUsize::new(ids.len()).unwrap();
        let shape = StepShape::new(NonZeroUsize::new(1).unwrap(), n, n).with_logits(LogitRows::All);
        let graph = model
            .trace(&StepSpec::step(shape), serving_trace_limits(n))
            .expect("trace the prefill");
        let graph = bind_packed_weights(&graph, &WeightFormats::from_weight_map(model.weights()))
            .expect("bind the packed weights");
        let runs = graph.layer_runs().expect("one run per decoder layer");

        let inputs = oracle_inputs(handle, &graph, ids);
        let evaluation = eval(&graph, &inputs, EvalOptions::default().keep_environment())
            .expect("evaluate the prefill");
        let environment = evaluation.environment.as_ref().expect("kept environment");
        let eqns = graph.eqns();
        let host = |eqn: usize| -> Vec<f32> {
            let value = environment
                .get(eqns[eqn].out)
                .and_then(Option::as_ref)
                .expect("the equation has a value");
            value
                .clone()
                .into_host()
                .expect("a dense value")
                .as_f32()
                .expect("an f32 value")
                .to_vec()
        };
        let mut sites = vec![(Site::Embedding, host(runs.runs[0].start - 1))];
        for (layer, run) in runs.runs.iter().enumerate() {
            sites.push((Site::LayerOutput(layer), host(run.end - 1)));
        }
        sites.push((
            Site::Logits,
            evaluation
                .output
                .into_host()
                .expect("dense logits")
                .as_f32()
                .expect("f32 logits")
                .to_vec(),
        ));
        sites
    }

    /// Where poot and llama.cpp part ways on Qwen2.5-0.5B Q8_0, layer by layer. poot follows llama.cpp
    /// running the weights as F16 to under 0.5% at every site; llama.cpp's own Q8_0 run is farther from its
    /// F16 run than poot is from either, by the activation quantization only the Q8_0 run has. That is the
    /// whole of the gap the KL gate first reported as 0.032 (which was in fact the reference file's
    /// clamped tail, see `llamacpp_kl_tests`).
    ///
    /// Mutation: feed poot the token ids shifted by one (`ids` below, `+ 1`); the bisection is red at the
    /// embedding with a relative difference near 1, not at some later layer.
    #[test]
    fn qwen2_5_0_5b_q8_0_follows_llamacpp_through_every_layer() {
        let Some(gguf) = poot_test_util::model_path(poot_test_util::checkpoint!(
            "qwen2.5-0.5b-gguf/qwen2.5-0.5b-instruct-q8_0.gguf"
        )) else {
            return;
        };
        let Some(tool) = dump_tool() else { return };
        let f16 = poot_test_util::unique_temp_path("q8_0_as_f16.gguf");
        let quantize = Command::new("llama-quantize")
            .args(["--allow-requantize"])
            .arg(&gguf)
            .arg(&*f16)
            .arg("F16")
            .output();
        match quantize {
            Ok(output) if output.status.success() => {}
            other => {
                eprintln!("POOT-SKIP: llama-quantize did not make the F16 copy: {other:?}");
                return;
            }
        }

        let handle = ModelHandle::load(
            &gguf,
            &poot_models::registry::Registry::builtin().unwrap(),
            &Default::default(),
        )
        .expect("load the GGUF");
        // llama.cpp first (seconds); the CPU oracle takes minutes.
        let (_f16_dir, reference_f16) = llama_dump(&tool, &f16, &TOKENS);
        let (_q8_dir, reference_q8) = llama_dump(&tool, &gguf, &TOKENS);
        let poot = oracle_sites(&handle, &TOKENS);
        // The embedding, one row per layer, the logits.
        let layers = poot.len() - 2;
        let against_f16 = Bisect::of(&poot, |s| reference_f16.tensor(&s.llama_name())).unwrap();
        let against_q8 = Bisect::of(&poot, |s| reference_q8.tensor(&s.llama_name())).unwrap();
        let q8_against_f16 = Bisect::of(
            &poot
                .iter()
                .map(|(s, _)| Ok((*s, reference_q8.tensor(&s.llama_name())?)))
                .collect::<Result<Vec<_>, BisectError>>()
                .unwrap(),
            |s| reference_f16.tensor(&s.llama_name()),
        )
        .unwrap();
        eprintln!("poot vs llama.cpp (F16 weights):\n{against_f16}");
        eprintln!("poot vs llama.cpp (Q8_0 weights):\n{against_q8}");
        eprintln!("llama.cpp Q8_0 vs llama.cpp F16 weights:\n{q8_against_f16}");

        if let Some(row) = against_f16.first_beyond(F16_LIMIT) {
            panic!(
                "poot leaves llama.cpp (F16 weights) at {}: {row:?}\n{against_f16}",
                row.site
            );
        }
        // The Q8_0 run is the one that is off: it sits farther from the F16 run than poot does, at the
        // last layer and the logits, and by about as much as it sits from poot.
        for site in [Site::LayerOutput(layers - 1), Site::Logits] {
            let (poot_f16, poot_q8, q8_f16) = (
                against_f16.at(site).relative(),
                against_q8.at(site).relative(),
                q8_against_f16.at(site).relative(),
            );
            assert!(q8_f16 > 3.0 * poot_f16, "{site}: {q8_f16} vs {poot_f16}");
            assert!(
                (poot_q8 / q8_f16 - 1.0).abs() < 0.25,
                "{site}: poot is {poot_q8} from the Q8_0 run, the F16 run {q8_f16}"
            );
        }
    }
}
