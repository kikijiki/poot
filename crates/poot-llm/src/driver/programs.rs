//! The compiled programs a planning pass hands to the driver that serves (POOT-1603).
//!
//! The startup capacity fit plans a candidate over recording devices ([`super::capacity::Planner`]); the
//! chosen candidate is then prepared on the real executor. Both run the production `prepare`, so both
//! compile the same programs. A program is a function of the driver's scope (model, target, compile
//! options, capacity, placement, expert pool) and the entry's key, and nothing a device does changes it, so
//! the planning pass records what it compiled and the serving driver is built from that record instead of
//! compiling again.
//!
//! The record cannot be applied to a driver of another scope: [`PlannedPrograms`] carries the scope it was
//! compiled under ([`DriverOptions`], handle, placement, pool) and the serving driver is built *from* it
//! ([`Driver::from_plan`], [`Driver::placed_from_plan`]), never from options the caller states a second
//! time. A different capacity is a different plan, so it is a different record.

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use poot_executor::{Collectives, Coordinator, Device, Executor};
use poot_graph_plan::staged::StagedProgram;

use super::error::DriverError;
use super::handle::ModelHandle;
use super::placement::Placement;
use super::{Driver, DriverOptions, EntryKey, State, Unprepared};

/// Where a driver gets the compiled program of an entry it prepares.
#[derive(Default)]
pub(super) enum Programs {
    /// Compile it: the serving driver of a startup that planned nothing, and every runtime-prepared entry.
    #[default]
    Compile,
    /// Compile it and keep it, for the driver that will be built from the record ([`PlannedPrograms`]).
    Record(HashMap<EntryKey, Arc<StagedProgram>>),
    /// Take the recorded program of the key; compile only a key the record never saw.
    Replay(HashMap<EntryKey, Arc<StagedProgram>>),
}

impl Programs {
    /// The recorded program for `key`, removed from the record: an entry is prepared once.
    pub(super) fn replay(&mut self, key: &EntryKey) -> Option<Arc<StagedProgram>> {
        match self {
            Programs::Replay(recorded) => recorded.remove(key),
            Programs::Compile | Programs::Record(_) => None,
        }
    }

    /// Keep `program` when this driver records.
    pub(super) fn record(&mut self, key: EntryKey, program: &Arc<StagedProgram>) {
        if let Programs::Record(recorded) = self {
            recorded.insert(key, Arc::clone(program));
        }
    }
}

/// The programs one planning pass compiled, with the scope it compiled them under.
pub struct PlannedPrograms {
    pub(super) handle: Arc<ModelHandle>,
    pub(super) options: DriverOptions,
    pub(super) placement: Placement,
    pub(super) pool: Option<NonZeroUsize>,
    pub(super) programs: HashMap<EntryKey, Arc<StagedProgram>>,
}

impl std::fmt::Debug for PlannedPrograms {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlannedPrograms")
            .field("programs", &self.programs.len())
            .field("placement", &self.placement)
            .finish_non_exhaustive()
    }
}

impl PlannedPrograms {
    /// The driver scope the record belongs to, as `driver` would build it.
    fn scoped(driver: Driver<Unprepared>, plan: Self) -> Result<Driver<Unprepared>, DriverError> {
        let mut driver = match plan.pool {
            Some(slots) => driver.with_expert_pool(slots)?,
            None => driver,
        };
        driver.ledger.programs = Programs::Replay(plan.programs);
        Ok(driver)
    }
}

impl Driver<Unprepared> {
    /// The serving driver over `executor`'s one device, built from what planning compiled: the plan's own
    /// model, options and expert pool, and its programs replayed instead of compiled again.
    pub fn from_plan(
        executor: Box<dyn Executor>,
        plan: PlannedPrograms,
    ) -> Result<Self, DriverError> {
        let driver = Self::new(Arc::clone(&plan.handle), executor, plan.options.clone())?;
        PlannedPrograms::scoped(driver, plan)
    }

    /// [`Driver::from_plan`] over a multi-device `coordinator`, placed as the plan was.
    pub fn placed_from_plan<D: Device + Collectives + 'static>(
        coordinator: Coordinator<D>,
        plan: PlannedPrograms,
    ) -> Result<Self, DriverError> {
        let driver = Self::placed(
            Arc::clone(&plan.handle),
            coordinator,
            plan.options.clone(),
            plan.placement.clone(),
        )?;
        PlannedPrograms::scoped(driver, plan)
    }

    /// Make this driver keep every program it compiles, for [`Driver::into_programs`].
    pub(super) fn recording(mut self) -> Self {
        self.ledger.programs = Programs::Record(HashMap::new());
        self
    }
}

impl<S: State> Driver<S> {
    /// The programs a recording driver compiled, with the scope they were compiled under. Empty for a
    /// driver that did not record.
    pub(super) fn into_programs(
        self,
        placement: Placement,
        pool: Option<NonZeroUsize>,
    ) -> PlannedPrograms {
        let programs = match self.ledger.programs {
            Programs::Record(recorded) => recorded,
            Programs::Compile | Programs::Replay(_) => HashMap::new(),
        };
        PlannedPrograms {
            handle: self.handle,
            options: self.settings.options,
            placement,
            pool,
            programs,
        }
    }
}
