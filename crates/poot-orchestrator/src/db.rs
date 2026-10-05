//! Durable run-state in SQLite. Every phase transition and per-model flag is committed, so a later
//! invocation can resume a sweep or reap the pod it recorded.
//!
//! Tables: `runs` (one row per sweep or exec, its pod, its phase), `model_runs` (per-model progress) and
//! `pods` (worker records). The pod id is persisted first thing after the create call returns, so a
//! crash cannot leak a pod.
//!
//! Ownership is split by semantic area (ADR 0096): `core` owns the `Db` handle and schema open path,
//! `runs` the sweep/exec run lifecycle and `pods` the worker records.

use anyhow::{Context, Result, bail};
use rusqlite::{Connection, OptionalExtension, params};
use std::io::ErrorKind;

use crate::clock::now_iso;

mod core;
mod pods;
mod runs;

pub use core::*;
pub use pods::*;
pub use runs::*;

/// Every pod this orchestrator creates is named `poot-orch-<state id>-<run id>-a<attempt>`.
const POD_NAME_PREFIX: &str = "poot-orch-";

/// The schema version this build writes. Version 1 is the first without the verify-batch tables.
const SCHEMA_VERSION: i64 = 1;

/// The tables the removed verify-batch command kept. Nothing reads them, and a state database created
/// before they were removed still holds them.
const DROPPED_BATCH_TABLES: [&str; 5] = [
    "batches",
    "batch_jobs",
    "batch_steps",
    "device_leases",
    "batch_phase_intervals",
];

/// Bring a state database's schema up to [`SCHEMA_VERSION`]. The `runs`, `model_runs` and `pods` rows
/// are untouched; a database already at the version is not written to.
fn migrate(conn: &Connection) -> Result<()> {
    let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    if version >= SCHEMA_VERSION {
        return Ok(());
    }
    let tx = conn.unchecked_transaction()?;
    for table in DROPPED_BATCH_TABLES {
        // Dropping a table drops its indexes with it.
        tx.execute_batch(&format!("DROP TABLE IF EXISTS {table}"))?;
    }
    tx.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?;
    tx.commit()?;
    Ok(())
}

/// The name prefix of every pod created against one state database: the only thing `reap` selects by.
///
/// Its field is private to this module and [`Db::pod_owner_prefix`] is the only constructor, so a pod
/// name cannot be built under a prefix the state database did not mint. A pod created under any other
/// name (the old `poot-batch-` pods were) is invisible to `reap` and leaks.
///
/// A string is not a prefix:
///
/// ```compile_fail,E0277
/// use poot_orchestrator::PodOwnerPrefix;
///
/// let _: PodOwnerPrefix = "poot-batch-".into();
/// ```
///
/// and neither is one accepted where the create call wants it:
///
/// ```compile_fail,E0308
/// use poot_orchestrator::CreateSpec;
///
/// let env = std::collections::HashMap::new();
/// let _ = CreateSpec {
///     owner: &"poot-batch-",
///     run_id: "run-1",
///     attempt: 1,
///     image: "ghcr.io/kikijiki/poot-bench:latest",
///     gpu_type_id: "NVIDIA L40S",
///     cloud_type: "SECURE",
///     gpu_count: 1,
///     container_disk_gb: 10,
///     ports: &["22/tcp"],
///     env: &env,
///     network_volume_id: None,
///     data_center_ids: &[],
///     allowed_cuda_versions: &[],
/// };
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PodOwnerPrefix(String);

impl PodOwnerPrefix {
    /// True when `pod_name` was created under this prefix.
    pub fn owns(&self, pod_name: &str) -> bool {
        pod_name.starts_with(&self.0)
    }

    /// The prefix of the pods one run creates (one pod per provisioning attempt).
    pub fn run_prefix(&self, run_id: &str) -> String {
        format!("{}{run_id}-", self.0)
    }

    /// The cloud name of the pod one provisioning `attempt` of `run_id` creates.
    pub fn pod_name(&self, run_id: &str, attempt: u32) -> String {
        format!("{}a{attempt}", self.run_prefix(run_id))
    }
}

impl Db {
    /// The identity of this state database: eight lowercase hex digits, minted the first time it is
    /// asked for and committed before this returns, so every later open of the same file sees the same
    /// value.
    ///
    /// Every pod the orchestrator creates carries it in its cloud name, so `reap` can tell this
    /// database's pods from another database's (worktrees and tests each have their own).
    fn state_id(&self) -> Result<String> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS state_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )",
        )?;
        // INSERT OR IGNORE then SELECT: two processes minting at once converge on the row that won.
        self.conn.execute(
            "INSERT OR IGNORE INTO state_meta (key, value) VALUES ('state_id', ?1)",
            params![mint_state_id()],
        )?;
        Ok(self.conn.query_row(
            "SELECT value FROM state_meta WHERE key = 'state_id'",
            [],
            |row| row.get(0),
        )?)
    }
}

impl Db {
    /// The prefix every pod created against this database carries. The state id is committed here, before
    /// the first create call, so a pod created and then lost to a crash is still named under a prefix
    /// `reap` can find.
    pub fn pod_owner_prefix(&self) -> Result<PodOwnerPrefix> {
        Ok(PodOwnerPrefix(format!(
            "{POD_NAME_PREFIX}{}-",
            self.state_id()?
        )))
    }
}

/// A fresh state id. `RandomState` is seeded from the OS per process, so this needs no extra dependency.
fn mint_state_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let bits = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish();
    format!("{:08x}", bits as u32)
}

#[cfg(test)]
mod tests;
