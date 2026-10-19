//! A model's vision towers on drivers of their own: the [`Towers`] a front end runs.
//!
//! A tower is an `encode`-only model of the container ([`ModelHandle::towers`]). Each runs on a
//! [`Driver`] over an executor of its own, with the tower's positions as its capacity, so the engine
//! that serves the text model fits its workload to the memory the towers left.

use std::num::{NonZeroU64, NonZeroUsize};

use poot_executor::Executor;
use poot_graph_plan::{CompileLimits, CompileOptions};
use poot_models::io::RequestInputs;
use poot_models::multimodal::front_end::{TowerError, TowerId, Towers};

use crate::driver::error::DriverError;
use crate::driver::{
    DraftPolicy, Driver, DriverOptions, EncodeRequest, EncodeRow, ModelHandle, PreparedSetLimits,
    program_retention, serving_trace_limits,
};

/// The driver options of a tower of `positions` positions: an `encode` step is stateless and takes the
/// tower's whole extent, so the capacity and the prefill piece are both `positions`.
pub fn tower_options(positions: NonZeroUsize) -> DriverOptions {
    let compile = CompileOptions::new(CompileLimits::STANDARD);
    DriverOptions {
        prefill: compile.clone(),
        decode: compile,
        capacity: positions,
        prefill_chunk: positions,
        trace: serving_trace_limits(positions),
        prepared: PreparedSetLimits {
            max_entries: NonZeroUsize::new(64).expect("nonzero"),
            max_retained_bytes: NonZeroU64::new(1 << 32).expect("nonzero"),
        },
        charge: program_retention,
        draft: DraftPolicy::Off,
        hidden_window: None,
    }
}

/// One driver per tower of a loaded model.
pub struct TowerDrivers {
    drivers: Vec<(TowerId, Driver)>,
}

impl TowerDrivers {
    /// Open a driver for each tower of `handle`. `open` supplies the executor and driver options of a
    /// tower from the tower's name and its positions (usually [`tower_options`] over a fresh executor);
    /// an executor refusal is `open`'s error, a driver refusal converts into `E`.
    pub fn open<E: From<DriverError>>(
        handle: &ModelHandle,
        mut open: impl FnMut(TowerId, NonZeroUsize) -> Result<(Box<dyn Executor>, DriverOptions), E>,
    ) -> Result<Self, E> {
        let mut drivers = Vec::with_capacity(handle.towers().len());
        for (id, tower) in handle.towers() {
            let positions = NonZeroUsize::new(tower.config().max_positions)
                .expect("a tower has at least one position");
            let (executor, options) = open(*id, positions)?;
            drivers.push((*id, Driver::new(tower.clone(), executor, options)?));
        }
        Ok(Self { drivers })
    }

    /// The driver running tower `id`, for its stats and caps.
    pub fn driver(&self, id: TowerId) -> Option<&Driver> {
        self.drivers
            .iter()
            .find(|(tower, _)| *tower == id)
            .map(|(_, driver)| driver)
    }
}

impl Towers for TowerDrivers {
    fn encode(&mut self, tower: TowerId, inputs: &RequestInputs) -> Result<Vec<f32>, TowerError> {
        let Some((_, driver)) = self.drivers.iter_mut().find(|(id, _)| *id == tower) else {
            return Err(TowerError::Failed(
                format!("the model container has no tower {tower}").into(),
            ));
        };
        let encoded = driver
            .encode(&EncodeRequest {
                rows: &[EncodeRow {
                    tokens: None,
                    inputs,
                }],
            })
            .map_err(|error| match error {
                // An item beyond what the tower takes in one step: the client's, not a fault.
                DriverError::Unsupported(_) => TowerError::Refused(error.to_string()),
                other => TowerError::Failed(Box::new(other)),
            })?;
        Ok(encoded
            .rows
            .into_iter()
            .next()
            .expect("one row in, one row out"))
    }
}
