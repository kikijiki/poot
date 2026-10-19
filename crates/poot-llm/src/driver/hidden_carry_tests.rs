//! The hidden-state carry through the driver (POOT-1109): one value per driver, built at `prepare` and
//! declared identically by every target program; a `Handoff` value leaves a KV-only family KV-only and
//! is never copied by the rollback; a program whose carry is not the one the step asked for is refused
//! typed at `prepare`. The driver, engine and planner are real, the device does nothing, so `prepare`
//! compiles through the production entry. The numeric value of the carry is the families' own rows
//! (`glm53_flash` against its independent reference, `components::standard`).

use std::collections::{BTreeSet, HashSet};
use std::num::{NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_executor::{Device, Dispatch, Engine};
use poot_graph_ir::{BinOp, Graph, StateRole, TensorType};
use poot_graph_plan::{CompileLimits, CompileOptions, Target};
use poot_models::components::standard::{HIDDEN_CARRY_STATE, Step, check_step};
use poot_models::io::{ModelSchema, ProgramId, ProgramSpec, StateSpec};
use poot_models::model::{
    FamilyKey, HiddenCarry, LogitRows, Model, ModelConfig, ModelError, ModelOutput, ShapeReason,
    StepSpec, TraceError,
};
use poot_models::registry::{ConfigSource, FamilyEntry, Fixture, RawConfig, Registry};
use poot_quant::weights::{WeightMap, WeightStore};
use poot_runtime_common::BufferRole;
use poot_target::{Backend, DeviceCaps};
use poot_tensor::DType;
use poot_text::ChatFormat;

use super::caps::PrepareRefused;
use super::error::{DriverError, Unsupported};
use super::tests::tiny_text;
use super::{
    DraftPolicy, Driver, DriverOptions, Head, Layout, ModelHandle, ModelRole, PreparedSetLimits,
    Program, Serving, ServingShapes, StateResidency, Warm, snapshot,
};

const VOCAB: usize = 12;
const WIDTH: usize = 4;
const CAPACITY: usize = 16;
const CHUNK: usize = 4;
const CARRIER: FamilyKey = FamilyKey::new("poot_fixture_carrier");

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

/// What the fake family carries beside its logits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Only the hidden-state carry.
    Kv,
    /// One `Recurrent` state beside the carry.
    Recurrent,
    /// A `Handoff` value one position wider than the step asked for.
    Wide,
    /// Never writes a carry, whatever the step asks.
    Ignoring,
}

impl Kind {
    fn parse(name: &str) -> Self {
        match name {
            "kv" => Self::Kv,
            "recurrent" => Self::Recurrent,
            "wide" => Self::Wide,
            "ignoring" => Self::Ignoring,
            other => panic!("no carrier kind {other}"),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Kv => "kv",
            Self::Recurrent => "recurrent",
            Self::Wide => "wide",
            Self::Ignoring => "ignoring",
        }
    }
}

#[derive(Debug)]
struct Carrier {
    config: ModelConfig,
    weights: WeightMap,
    schema: ModelSchema,
    kind: Kind,
}

impl Model for Carrier {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn schema(&self) -> &ModelSchema {
        &self.schema
    }

    fn weights(&self) -> &WeightMap {
        &self.weights
    }

    fn trace_into(&self, spec: &StepSpec, b: poot_graph_ir::Builder) -> Result<Graph, TraceError> {
        let shape = spec.shape;
        check_step(CARRIER, shape, CAPACITY)?;
        let step = Step::new(&b, shape)?;
        let (n, rows) = (shape.tokens.get(), shape.rows.get());
        let token = b.cast(
            b.guard_index_bounds(step.tokens(), VOCAB)
                .expect("the step's ids are a rank-2 I32 tensor")
                .guarded,
            DType::F32,
        )?;
        let pos = b.cast(step.pos(), DType::F32)?;
        let seed = b.binary(BinOp::Add, token, pos)?;
        // The state a position would hand on: its seed, at every position and width.
        let hidden = b.broadcast(b.reshape(seed, vec![rows, n, 1])?, vec![rows, n, WIDTH])?;
        match self.kind {
            Kind::Kv => step.carry_hidden(&b, hidden)?,
            Kind::Recurrent => {
                let state = step.recurrent_state(&b, "toy.state", DType::F32, &[WIDTH])?;
                let last = b.slice(seed, 1, n - 1, n)?;
                let folded = b.binary(
                    BinOp::Add,
                    state.current(),
                    b.broadcast(last, vec![rows, WIDTH])?,
                )?;
                step.carry_recurrent(&b, state, folded)?;
                step.carry_hidden(&b, hidden)?;
            }
            Kind::Wide => {
                let positions = shape.hidden.map_or(1, |carry| carry.positions.get()) + 1;
                let state = b.state_input(
                    HIDDEN_CARRY_STATE,
                    TensorType::f32(vec![rows, positions, WIDTH]),
                    StateRole::Handoff,
                )?;
                step.carry(state, state);
            }
            Kind::Ignoring => {}
        }
        let out = if shape.logits == LogitRows::All { n } else { 1 };
        let logits = b.broadcast(b.slice(seed, 1, n - out, n)?, vec![rows, out])?;
        let logits = b.broadcast(
            b.reshape(logits, vec![rows, out, 1])?,
            vec![rows, out, VOCAB],
        )?;
        Ok(step.finish(b, logits)?)
    }
}

fn build_carrier(raw: &RawConfig<'_>, _: &WeightStore) -> Result<Box<dyn Model>, ModelError> {
    let RawConfig::HfJson { config, .. } = raw else {
        panic!("the fixture is an HF checkpoint")
    };
    let kind = Kind::parse(config["carrier"].as_str().expect("a carrier kind"));
    // The graph carries the hand-off (a KV-only family has no cache of its own) and, for the
    // recurrent kind, one recurrent state: the schema declares exactly that, as `trace` checks.
    let mut step = ProgramSpec::new(ProgramId::STEP, ModelOutput::Logits { vocab: VOCAB })
        .carrying(StateSpec::hand_off());
    if kind == Kind::Recurrent {
        step = step.carrying(StateSpec::recurrent("toy.state"));
    }
    let schema = ModelSchema::builder().step(step).build_for(CARRIER)?;
    Ok(Box::new(Carrier {
        schema,
        config: ModelConfig {
            family: CARRIER,
            vocab: VOCAB,
            max_positions: CAPACITY,
            eos: BTreeSet::from([VOCAB as u32 - 1]),
            bos: None,
            prompt_bos: None,
            prefill_granule: nz(1),
            chat: ChatFormat::ChatML,
        },
        weights: WeightMap::default(),
        kind,
    }))
}

fn handle_of(kind: Kind) -> Arc<ModelHandle> {
    let mut registry = Registry::empty();
    registry
        .register(FamilyEntry {
            family: CARRIER,
            keys: &[(ConfigSource::HfModelType, "poot_fixture_carrier")],
            build: build_carrier,
            fixture: || Fixture {
                config: serde_json::json!({}),
                generation: None,
                store: WeightStore::default(),
            },
            variants: &[],
        })
        .unwrap();
    let config = serde_json::json!({
        "model_type": "poot_fixture_carrier",
        "carrier": kind.name(),
    });
    let raw = RawConfig::HfJson {
        config: &config,
        generation: None,
    };
    let handle =
        ModelHandle::from_checkpoint(&raw, WeightStore::default(), &registry, |_| Ok(tiny_text()))
            .unwrap();
    Arc::new(handle)
}

#[derive(Debug, thiserror::Error)]
#[error("fake device error")]
struct FakeError;

/// A device that does nothing: the driver, engine and planner are real, so `prepare` compiles.
struct FakeDevice;

impl Device for FakeDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = FakeError;

    fn target(&self) -> Target {
        Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: DeviceCaps::wgpu_rdna3_igpu(),
        }
    }
    fn memory(&self) -> Vec<(BufferRole, poot_runtime_common::MemoryCounterSnapshot)> {
        Vec::new()
    }
    fn allocate(
        &mut self,
        _role: BufferRole,
        _storage: poot_target::BufferStorage,
        _elems: usize,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    /// This fake keeps no storage, so it states no bound: the engine's length check passes every
    /// dispatch it records.
    fn buffer_elems(&self, _buffer: &()) -> usize {
        usize::MAX
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn begin(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn dispatch(&mut self, _d: Dispatch<'_, Self>) -> Result<(), FakeError> {
        Ok(())
    }
    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn finish(&mut self) -> Result<Option<()>, FakeError> {
        Ok(Some(()))
    }
    fn replay(&mut self, _recording: &()) -> Result<(), FakeError> {
        Ok(())
    }
    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(std::ops::Range<usize>, &mut [u8]),
    ) -> Result<(), FakeError> {
        Ok(())
    }
    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), FakeError> {
        out.fill(0);
        Ok(())
    }
    fn synchronize(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
    fn device_time(&self) -> poot_executor::DeviceTime {
        poot_executor::DeviceTime::Unknown
    }
    fn abort(&mut self) -> Result<(), FakeError> {
        Ok(())
    }
}

/// A driver over the carrier `kind` speculating `k` tokens a round, its carry sized for the round's
/// window (`k + 1`), as the policy that consumes the carry sizes it.
fn driver_of(kind: Kind, k: usize, carry: bool) -> Driver {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    let draft = DraftPolicy::PromptLookup {
        ngram: nz(1),
        k: nz(k),
    };
    Driver::new(
        handle_of(kind),
        Box::new(Engine::new(FakeDevice)),
        DriverOptions {
            prefill: compile.clone(),
            decode: compile,
            capacity: nz(CAPACITY),
            prefill_chunk: nz(CHUNK),
            trace: poot_graph_ir::TraceLimits::for_tests_with_tokens(nz(CAPACITY)),
            prepared: PreparedSetLimits {
                max_entries: nz(64),
                max_retained_bytes: NonZeroU64::new(1 << 40).unwrap(),
            },
            charge: super::program_retention,
            draft,
            hidden_window: carry.then(|| nz(k + 1)),
        },
    )
    .unwrap()
}

fn prepare(
    driver: Driver,
    windows: &[NonZeroUsize],
) -> Result<Driver<Serving>, Box<PrepareRefused>> {
    driver.prepare(&ServingShapes {
        layout: Layout::Contiguous,
        rows: &[NonZeroUsize::MIN],
        heads: &[Head::GREEDY],
        warm: Warm::Admitted,
        windows,
        adapters: &[],
    })
}

/// The carry aval (`[rows, positions, width]`) each prepared target program declares, by key.
fn carry_avals(driver: &Driver<Serving>) -> Vec<(super::EntryKey, Option<Vec<usize>>)> {
    driver
        .ledger
        .entries
        .keys()
        .filter(|key| key.role == ModelRole::Target)
        .map(|key| {
            let graph = driver
                .handle
                .model()
                .trace(
                    &StepSpec::new(key.program.id(), key.shape),
                    poot_graph_ir::TraceLimits::for_tests(),
                )
                .expect("a prepared key traces");
            let shapes: Vec<Vec<usize>> = graph
                .state_pairs()
                .filter(|pair| pair.role == StateRole::Handoff)
                .map(|pair| graph.aval(pair.input).shape.clone())
                .collect();
            assert!(shapes.len() <= 1, "one carry per program");
            (*key, shapes.into_iter().next())
        })
        .collect()
}

/// SC-002: a KV-only family that writes the carry stays `KvOnly`; the carry's extent is the one value
/// `max(k + 1, widest prefill piece)` built at `prepare`, and every target program (prefill pieces,
/// decode, the verify window) declares that one aval, so the executable shares one buffer. A window
/// wider than the chunk sets the extent. Mutation: declare the carry `StateRole::Recurrent` in
/// `Step::carry_hidden`; the residency reads `Recurrent` and the first assertion is red. Mutation: size
/// the carry by the window alone (`fold(window, ..)` without the pieces); a four-token chunk exceeds a
/// three-position carry and `prepare` is refused.
#[test]
fn a_kv_only_family_with_the_carry_stays_kv_only_and_every_target_program_declares_one_aval() {
    for (k, positions) in [(2, CHUNK), (5, 6)] {
        let driver =
            prepare(driver_of(Kind::Kv, k, true), &[]).unwrap_or_else(|e| panic!("k = {k}: {e}"));
        assert_eq!(
            driver.caps().unwrap().state,
            StateResidency::KvOnly,
            "the carry is never rolled back, so it is not recurrent state"
        );
        assert_eq!(
            driver.settings.hidden,
            Some(HiddenCarry {
                positions: nz(positions)
            })
        );
        let avals = carry_avals(&driver);
        assert!(
            avals.iter().any(|(key, _)| key.shape.tokens.get() == k + 1),
            "the verify window is among the programs: {avals:?}"
        );
        let distinct: HashSet<_> = avals.iter().map(|(_, aval)| aval.clone()).collect();
        assert_eq!(
            distinct,
            HashSet::from([Some(vec![1, positions, WIDTH])]),
            "every target program declares the one carry aval"
        );
        for (key, _) in &avals {
            assert_eq!(key.shape.hidden, driver.settings.hidden, "{key:?}");
        }
        for key in driver
            .ledger
            .entries
            .keys()
            .filter(|key| key.role != ModelRole::Target)
        {
            assert_eq!(key.shape.hidden, None, "only the target carries: {key:?}");
        }
    }
}

/// SC-002: over a recurrent family the rollback copies the recurrent value and not the carry: the
/// snapshot and restore programs name `toy.state` and its shadow only, so a restore leaves the carry
/// where the window put it. Mutation: declare the carry `StateRole::Recurrent`; the copies name the
/// carry and its shadow too and the assertion is red.
#[test]
fn the_rollback_copies_recurrent_state_and_leaves_the_carry_untouched() {
    let driver =
        prepare(driver_of(Kind::Recurrent, 2, true), &[]).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(driver.caps().unwrap().state, StateResidency::Recurrent);
    assert_eq!(
        driver.settings.hidden,
        Some(HiddenCarry {
            positions: nz(CHUNK)
        })
    );
    let rollback_keys: Vec<_> = driver
        .ledger
        .entries
        .keys()
        .filter(|key| matches!(key.role, ModelRole::Snapshot | ModelRole::Restore))
        .collect();
    assert_eq!(rollback_keys.len(), 2, "a snapshot and a restore entry");
    assert!(
        rollback_keys
            .iter()
            .all(|key| key.shape.hidden == driver.settings.hidden),
        "the rollback copies trace the target's decode program, carry included"
    );
    let decode = driver
        .ledger
        .entries
        .keys()
        .find(|key| {
            key.role == ModelRole::Target && key.program == Program::Step && key.shape.is_decode()
        })
        .copied()
        .expect("a decode program");
    let schema = driver
        .handle
        .model()
        .trace(
            &StepSpec::new(decode.program.id(), decode.shape),
            poot_graph_ir::TraceLimits::for_tests(),
        )
        .unwrap();
    let carried: Vec<String> = schema
        .state_pairs()
        .filter(|pair| pair.role == StateRole::Handoff)
        .map(|pair| schema.meta(pair.input).name.clone().unwrap())
        .collect();
    assert_eq!(
        carried,
        [HIDDEN_CARRY_STATE],
        "the schema declares the carry"
    );
    let rollback = snapshot::snapshot_entries(&schema, poot_graph_ir::TraceLimits::for_tests())
        .expect("recurrent state is rolled back")
        .unwrap();
    for graph in [&rollback.snapshot, &rollback.restore] {
        let mut names: Vec<String> = graph
            .state()
            .map(|(input, _)| graph.meta(input).name.clone().unwrap())
            .collect();
        names.sort();
        assert_eq!(names, ["toy.state", "toy.state@snap"]);
    }
}

/// SC-003: a program whose carry is not the aval the step asked for is refused typed at `prepare`,
/// naming what it found and what was asked for; so is a carry written when none was asked. Mutation:
/// drop the `Handoff` arm of `check_state_rows`; `prepare` succeeds and both assertions are red.
#[test]
fn a_carry_aval_that_differs_from_the_request_is_refused_typed() {
    let positions = CHUNK;
    for (carry, expected) in [(true, Some(positions)), (false, None)] {
        let refused = prepare(driver_of(Kind::Wide, 2, carry), &[])
            .expect_err("a mismatched carry is refused");
        let (driver, error) = (refused.driver, refused.error);
        let DriverError::Unsupported(Unsupported::Trace(TraceError::ShapeUnsupported {
            family,
            reason,
            ..
        })) = error
        else {
            panic!("not a typed shape refusal: {error:?}");
        };
        assert_eq!(family, CARRIER);
        assert_eq!(
            reason,
            match expected {
                Some(expected) => ShapeReason::HiddenCarryPositions {
                    found: expected + 1,
                    expected,
                },
                None => ShapeReason::HiddenCarryUnrequested,
            }
        );
        assert!(
            driver.caps().is_none(),
            "a refused prepare publishes nothing"
        );
    }
}

/// SC-003: a family whose tracer never writes the carry its step asked for is refused typed, rather
/// than the readers binding a buffer nobody writes. Mutation: drop the `HiddenCarryMissing` return;
/// `prepare` succeeds and the assertion is red.
#[test]
fn a_family_that_ignores_the_request_is_refused_typed() {
    let error = prepare(driver_of(Kind::Ignoring, 2, true), &[])
        .expect_err("an unwritten carry is refused")
        .error;
    let DriverError::Unsupported(Unsupported::Trace(TraceError::ShapeUnsupported {
        reason, ..
    })) = error
    else {
        panic!("not a typed shape refusal: {error:?}");
    };
    assert_eq!(reason, ShapeReason::HiddenCarryMissing);
}
