use super::*;

/// A run record. `status` drives resume/teardown. `kind` is the resumable bench `sweep` (the `run`
/// command) or a one-off `exec`; both are recorded so every pod is tracked, but only `sweep` rows are
/// eligible for resume (see `active_run`).
#[derive(Debug, Clone)]
pub struct Run {
    pub id: String,
    pub kind: String, // sweep | exec
    pub git_ref: String,
    pub scenario: String,
    /// Runs of the whole matrix per model, one results directory each. Fixed when the run is created: a
    /// resumed run sweeps, gathers and reports the number it started with, whatever the new command line says.
    pub repeats: u32,
    pub models: Vec<String>,
    pub gpu_types: String,
    pub image: String,
    pub status: String, // new | provisioned | setup | sweeping | gathering | running | done | failed | aborted
    pub pod_id: Option<String>,
}

impl Run {
    /// The results directories one model's sweep wrote, given the `bench run --run-id` it recorded: the run
    /// itself for one repeat, else `<run_id>-r1` .. `-rN`, each a valid `bench compare` input.
    pub fn model_snapshot_dirs(&self, run_id: &str) -> Vec<String> {
        repeat_dirs(run_id, self.repeats)
    }
}

fn repeat_dirs(run_id: &str, repeats: u32) -> Vec<String> {
    if repeats == 1 {
        vec![run_id.to_string()]
    } else {
        (1..=repeats).map(|n| format!("{run_id}-r{n}")).collect()
    }
}

/// A kept-warm exec pod a later exec may adopt (see [`Db::find_warm_exec`]).
#[derive(Debug, Clone)]
pub struct WarmExec {
    pub run_id: String,
    pub image: String,
    pub pod_id: String,
    pub host: String,
    pub port: u16,
    pub warm_until: String, // UTC ISO; expired once <= now_iso()
}

/// Identity of the local process that owns a nonterminal `exec` row. A PID can be reused, so the
/// process start tick and boot id must also match before the owner counts as live.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecOwner {
    pub hostname: String,
    pub pid: i64,
    pub boot_id: String,
    pub start_ticks: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnerLiveness {
    /// The recorded owner is still the same local process.
    Live,
    /// The recorded owner is provably gone, or the PID now names a different process.
    Dead,
    /// Reap cannot prove dead safely, so normal reap must keep the row protected.
    Unknown,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ExecReapReport {
    pub aborted: usize,
    pub kept_live: usize,
    pub kept_unknown: usize,
}

/// Final DB-side protection snapshot for one normal reap pass.
///
/// `run_ids` protect newly provisioned pods by their run-specific cloud name before `pod_id` is
/// recorded. `has_unrecorded` covers an older concurrent orchestrator whose generic pod name cannot be
/// mapped back to its run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReapProtection {
    pub pod_ids: Vec<String>,
    pub run_ids: Vec<String>,
    pub has_unrecorded: bool,
}

#[derive(Debug, Clone)]
struct ExecReapCandidate {
    id: String,
    status: String,
    warm_until: Option<String>,
    owner_hostname: Option<String>,
    owner_pid: Option<i64>,
    owner_boot_id: Option<String>,
    owner_start_ticks: Option<String>,
}

impl ExecReapCandidate {
    fn owner(&self) -> Option<ExecOwner> {
        match (
            self.owner_hostname.clone(),
            self.owner_pid,
            self.owner_boot_id.clone(),
            self.owner_start_ticks.clone(),
        ) {
            (Some(hostname), Some(pid), Some(boot_id), Some(start_ticks)) => Some(ExecOwner {
                hostname,
                pid,
                boot_id,
                start_ticks,
            }),
            _ => None,
        }
    }
}

impl ExecOwner {
    pub fn current() -> Result<Self> {
        let pid = std::process::id() as i64;
        Ok(Self {
            hostname: local_hostname(),
            pid,
            boot_id: local_boot_id().context("read local boot id")?,
            start_ticks: process_start_ticks(pid)
                .context("read current process start ticks")?
                .context("current process disappeared while reading start ticks")?,
        })
    }

    pub fn local_liveness(&self) -> OwnerLiveness {
        let hostname = local_hostname();
        if self.hostname != hostname {
            return OwnerLiveness::Unknown;
        }
        let boot_id = match local_boot_id() {
            Ok(id) => id,
            Err(_) => return OwnerLiveness::Unknown,
        };
        if self.boot_id != boot_id {
            return OwnerLiveness::Dead;
        }
        process_start_liveness(process_start_ticks(self.pid), &self.start_ticks)
    }
}

fn local_hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| std::env::var("HOSTNAME").ok())
        .unwrap_or_else(|| "unknown".to_string())
}

fn local_boot_id() -> Result<String> {
    Ok(std::fs::read_to_string("/proc/sys/kernel/random/boot_id")?
        .trim()
        .to_string())
}

fn process_start_ticks(pid: i64) -> Result<Option<String>> {
    if pid <= 0 {
        bail!("invalid pid {pid}");
    }
    let stat = match std::fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(stat) => stat,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let (_, rest) = stat
        .rsplit_once(") ")
        .ok_or_else(|| anyhow::anyhow!("malformed /proc/{pid}/stat"))?;
    let Some(start) = rest.split_whitespace().nth(19) else {
        bail!("missing starttime in /proc/{pid}/stat");
    };
    Ok(Some(start.to_string()))
}

pub(crate) fn process_start_liveness(
    observed: Result<Option<String>>,
    expected_start_ticks: &str,
) -> OwnerLiveness {
    match observed {
        Ok(Some(ticks)) if ticks == expected_start_ticks => OwnerLiveness::Live,
        Ok(Some(_)) | Ok(None) => OwnerLiveness::Dead,
        Err(_) => OwnerLiveness::Unknown,
    }
}

/// Per-model progress inside a run.
#[derive(Debug, Clone, Default)]
pub struct ModelState {
    pub setup_done: bool,
    pub sweep_done: bool,
    pub pod_run_id: Option<String>, // the harness results dir the sweep wrote on the pod
    pub gathered: bool,
}

/// A run plus its per-model progress, serialized to JSON for the web dashboard.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RunView {
    pub id: String,
    pub kind: String,
    pub created_at: String,
    pub updated_at: String,
    pub git_ref: String,
    pub scenario: String,
    pub gpu_types: String,
    pub image: String,
    pub status: String,
    pub pod_id: Option<String>,
    pub ssh_host: Option<String>,
    pub ssh_port: Option<u16>,
    pub models: Vec<ModelView>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ModelView {
    pub model: String,
    pub setup_done: bool,
    pub sweep_done: bool,
    pub gathered: bool,
    pub pod_run_id: Option<String>,
}

impl Db {
    pub fn create_run(&self, r: &Run) -> Result<()> {
        self.create_run_with_owner(r, None)
    }

    pub fn create_run_with_owner(&self, r: &Run, owner: Option<&ExecOwner>) -> Result<()> {
        let now = now_iso();
        self.conn.execute(
            "INSERT INTO runs (
                id, created_at, updated_at, git_ref, scenario, models, gpu_types, image, status, kind,
                owner_hostname, owner_pid, owner_boot_id, owner_start_ticks, repeats
             )
             VALUES (?1, ?2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                r.id,
                now,
                r.git_ref,
                r.scenario,
                r.models.join(","),
                r.gpu_types,
                r.image,
                r.status,
                r.kind,
                owner.map(|o| o.hostname.as_str()),
                owner.map(|o| o.pid),
                owner.map(|o| o.boot_id.as_str()),
                owner.map(|o| o.start_ticks.as_str()),
                r.repeats,
            ],
        )?;
        for m in &r.models {
            self.conn.execute(
                "INSERT OR IGNORE INTO model_runs (run_id, model) VALUES (?1, ?2)",
                params![r.id, m],
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn set_exec_owner(&self, run_id: &str, owner: &ExecOwner) -> Result<()> {
        self.conn.execute(
            "UPDATE runs
             SET owner_hostname=?2, owner_pid=?3, owner_boot_id=?4, owner_start_ticks=?5, updated_at=?6
             WHERE id=?1 AND kind='exec'",
            params![
                run_id,
                owner.hostname.as_str(),
                owner.pid,
                owner.boot_id.as_str(),
                owner.start_ticks.as_str(),
                now_iso()
            ],
        )?;
        Ok(())
    }

    /// The single in-progress run, if any. One sweep at a time: a second would fight over pod-name reaping.
    pub fn active_run(&self) -> Result<Option<Run>> {
        self.conn
            .query_row(
                "SELECT id, git_ref, scenario, models, gpu_types, image, status, pod_id, kind, repeats
                 FROM runs WHERE status NOT IN ('done','failed','aborted') AND kind='sweep'
                 ORDER BY created_at DESC LIMIT 1",
                [],
                Self::row_to_run,
            )
            .optional()
            .map_err(Into::into)
    }

    fn row_to_run(row: &rusqlite::Row) -> rusqlite::Result<Run> {
        let models: String = row.get(3)?;
        Ok(Run {
            id: row.get(0)?,
            git_ref: row.get(1)?,
            scenario: row.get(2)?,
            models: models
                .split(',')
                .filter(|s| !s.is_empty())
                .map(String::from)
                .collect(),
            gpu_types: row.get(4)?,
            image: row.get(5)?,
            status: row.get(6)?,
            pod_id: row.get(7)?,
            kind: row.get(8)?,
            repeats: row.get(9)?,
        })
    }

    /// Card 437: refuses to overwrite a row still inside its keep-warm window. [`Self::set_warm`] stores
    /// warm-ness in `status`, so an unguarded write would leave a terminal status with `warm_until` in the
    /// future: unadoptable ([`Self::find_warm_exec`] filters `status='warm'`) and reapable
    /// ([`Self::reap_protection`] treats terminal statuses as reapable), destroying a deliberately kept pod.
    ///
    /// Legitimate exits from 'warm' use their own guarded SQL ([`Self::try_adopt_warm_exec`],
    /// [`Self::try_abort_warm_exec`], `abort_exec_candidate`). An expired warm row is not protected.
    pub fn set_status(&self, run_id: &str, status: &str) -> Result<()> {
        let now = now_iso();
        self.conn.execute(
            "UPDATE runs SET status=?2, updated_at=?3
             WHERE id=?1
               AND NOT (status='warm' AND warm_until IS NOT NULL AND warm_until > ?3)",
            params![run_id, status, now],
        )?;
        Ok(())
    }

    pub fn set_pod(
        &self,
        run_id: &str,
        pod_id: &str,
        host: Option<&str>,
        port: Option<u16>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE runs SET pod_id=?2, ssh_host=?3, ssh_port=?4, updated_at=?5 WHERE id=?1",
            params![run_id, pod_id, host, port.map(|p| p as i64), now_iso()],
        )?;
        Ok(())
    }

    pub fn model_state(&self, run_id: &str, model: &str) -> Result<ModelState> {
        self.conn
            .query_row(
                "SELECT setup_done, sweep_done, pod_run_id, gathered FROM model_runs WHERE run_id=?1 AND model=?2",
                params![run_id, model],
                |row| {
                    Ok(ModelState {
                        setup_done: row.get::<_, i64>(0)? != 0,
                        sweep_done: row.get::<_, i64>(1)? != 0,
                        pod_run_id: row.get(2)?,
                        gathered: row.get::<_, i64>(3)? != 0,
                    })
                },
            )
            .optional()
            .map(|o| o.unwrap_or_default())
            .map_err(Into::into)
    }

    pub fn set_model_setup(&self, run_id: &str, model: &str, done: bool) -> Result<()> {
        self.conn.execute(
            "UPDATE model_runs SET setup_done=?3 WHERE run_id=?1 AND model=?2",
            params![run_id, model, done as i64],
        )?;
        Ok(())
    }

    pub fn set_model_sweep(
        &self,
        run_id: &str,
        model: &str,
        done: bool,
        pod_run_id: Option<&str>,
    ) -> Result<()> {
        self.conn.execute(
            "UPDATE model_runs SET sweep_done=?3, pod_run_id=COALESCE(?4, pod_run_id) WHERE run_id=?1 AND model=?2",
            params![run_id, model, done as i64, pod_run_id],
        )?;
        Ok(())
    }

    pub fn set_model_gathered(&self, run_id: &str, model: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE model_runs SET gathered=1 WHERE run_id=?1 AND model=?2",
            params![run_id, model],
        )?;
        Ok(())
    }

    /// The results-snapshot dirs this run produced, each a `benchmarks/results/<dir>`: every repeat of every
    /// model's sweep (see [`Run::model_snapshot_dirs`]). Used by the dashboard's Results tab and the commit message.
    pub fn snapshot_dirs(&self, run_id: &str) -> Result<Vec<String>> {
        let repeats: u32 = self
            .conn
            .query_row(
                "SELECT repeats FROM runs WHERE id=?1",
                params![run_id],
                |row| row.get(0),
            )
            .optional()?
            .unwrap_or(1);
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT pod_run_id FROM model_runs WHERE run_id=?1 AND pod_run_id IS NOT NULL ORDER BY model",
        )?;
        let recorded = stmt
            .query_map(params![run_id], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(recorded
            .iter()
            .flat_map(|base| repeat_dirs(base, repeats))
            .collect())
    }

    /// Finalize nonterminal `exec` rows only when their owner is gone. An `exec` is non-resumable, but
    /// another process may be running it, so normal reap checks the owner identity before releasing its
    /// pod from `live_pod_ids`. `force` is the `reap --all` override.
    pub fn abort_stale_execs(&self, force: bool) -> Result<ExecReapReport> {
        self.abort_stale_execs_with(force, |owner| owner.local_liveness())
    }

    pub fn abort_stale_execs_with(
        &self,
        force: bool,
        liveness: impl Fn(&ExecOwner) -> OwnerLiveness,
    ) -> Result<ExecReapReport> {
        let now = now_iso();
        let candidates = {
            let mut stmt = self.conn.prepare(
                "SELECT id, status, warm_until, owner_hostname, owner_pid, owner_boot_id, owner_start_ticks
                 FROM runs
                 WHERE kind='exec' AND status NOT IN ('done','failed','aborted')",
            )?;
            stmt.query_map([], |row| {
                Ok(ExecReapCandidate {
                    id: row.get(0)?,
                    status: row.get(1)?,
                    warm_until: row.get(2)?,
                    owner_hostname: row.get(3)?,
                    owner_pid: row.get(4)?,
                    owner_boot_id: row.get(5)?,
                    owner_start_ticks: row.get(6)?,
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?
        };

        let mut report = ExecReapReport::default();
        for candidate in candidates {
            if !force
                && candidate.status == "warm"
                && candidate
                    .warm_until
                    .as_deref()
                    .is_some_and(|until| until > now.as_str())
            {
                report.kept_live += 1;
                continue;
            }

            let abort = if force {
                true
            } else {
                match candidate.owner().as_ref().map(&liveness) {
                    Some(OwnerLiveness::Live) => {
                        report.kept_live += 1;
                        false
                    }
                    Some(OwnerLiveness::Dead) => true,
                    Some(OwnerLiveness::Unknown) | None => {
                        report.kept_unknown += 1;
                        false
                    }
                }
            };

            if abort {
                let n = self.abort_exec_candidate(&candidate, &now)?;
                report.aborted += n;
            }
        }
        Ok(report)
    }

    fn abort_exec_candidate(&self, candidate: &ExecReapCandidate, now: &str) -> Result<usize> {
        self.conn
            .execute(
                "UPDATE runs
                 SET status='aborted', updated_at=?2
                 WHERE id=?1
                   AND kind='exec'
                   AND status NOT IN ('done','failed','aborted')
                   AND status IS ?3
                   AND warm_until IS ?4
                   AND owner_hostname IS ?5
                   AND owner_pid IS ?6
                   AND owner_boot_id IS ?7
                   AND owner_start_ticks IS ?8",
                params![
                    candidate.id.as_str(),
                    now,
                    candidate.status.as_str(),
                    candidate.warm_until.as_deref(),
                    candidate.owner_hostname.as_deref(),
                    candidate.owner_pid,
                    candidate.owner_boot_id.as_deref(),
                    candidate.owner_start_ticks.as_deref(),
                ],
            )
            .map_err(Into::into)
    }

    /// Keep a finished exec's pod warm until `warm_until` (UTC ISO); a later exec with the same image
    /// adopts it via [`find_warm_exec`].
    pub fn set_warm(&self, run_id: &str, warm_until: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE runs SET status='warm', warm_until=?2, updated_at=?3 WHERE id=?1",
            params![run_id, warm_until, now_iso()],
        )?;
        Ok(())
    }

    /// Atomically move a still-matching warm pod claim to this exec row, so two concurrent execs cannot
    /// adopt the same pod.
    pub fn try_adopt_warm_exec(
        &self,
        warm: &WarmExec,
        new_run_id: &str,
        new_owner: &ExecOwner,
    ) -> Result<bool> {
        let now = now_iso();
        let tx = self.conn.unchecked_transaction()?;
        let old_changed = tx.execute(
            "UPDATE runs
             SET status='done', updated_at=?7
             WHERE id=?1
               AND kind='exec'
               AND status='warm'
               AND image=?2
               AND pod_id=?3
               AND ssh_host=?4
               AND ssh_port IS ?5
               AND warm_until IS ?6",
            params![
                warm.run_id.as_str(),
                warm.image.as_str(),
                warm.pod_id.as_str(),
                warm.host.as_str(),
                warm.port as i64,
                warm.warm_until.as_str(),
                now.as_str(),
            ],
        )?;
        if old_changed == 0 {
            tx.rollback()?;
            return Ok(false);
        }

        let new_changed = tx.execute(
            "UPDATE runs
             SET pod_id=?2, ssh_host=?3, ssh_port=?4, status='running', updated_at=?5
             WHERE id=?1
               AND kind='exec'
               AND status NOT IN ('done','failed','aborted')
               AND owner_hostname IS ?6
               AND owner_pid IS ?7
               AND owner_boot_id IS ?8
               AND owner_start_ticks IS ?9",
            params![
                new_run_id,
                warm.pod_id.as_str(),
                warm.host.as_str(),
                warm.port as i64,
                now.as_str(),
                new_owner.hostname.as_str(),
                new_owner.pid,
                new_owner.boot_id.as_str(),
                new_owner.start_ticks.as_str(),
            ],
        )?;
        if new_changed == 0 {
            tx.rollback()?;
            return Ok(false);
        }
        tx.commit()?;
        Ok(true)
    }

    /// Mark a still-matching warm row aborted before deleting its pod. If another process already
    /// changed the row, the caller must not delete the pod.
    pub fn try_abort_warm_exec(&self, warm: &WarmExec) -> Result<bool> {
        let n = self.conn.execute(
            "UPDATE runs
             SET status='aborted', updated_at=?7
             WHERE id=?1
               AND kind='exec'
               AND status='warm'
               AND image=?2
               AND pod_id=?3
               AND ssh_host=?4
               AND ssh_port IS ?5
               AND warm_until IS ?6",
            params![
                warm.run_id.as_str(),
                warm.image.as_str(),
                warm.pod_id.as_str(),
                warm.host.as_str(),
                warm.port as i64,
                warm.warm_until.as_str(),
                now_iso(),
            ],
        )?;
        Ok(n > 0)
    }

    /// The newest kept-warm exec pod for `image` with a recorded endpoint. The caller checks expiry and
    /// SSH reachability.
    pub fn find_warm_exec(&self, image: &str) -> Result<Option<WarmExec>> {
        self.conn
            .query_row(
                "SELECT id, image, pod_id, ssh_host, ssh_port, warm_until FROM runs
                 WHERE kind='exec' AND status='warm' AND image=?1 AND pod_id IS NOT NULL
                   AND ssh_host IS NOT NULL AND ssh_port IS NOT NULL
                 ORDER BY updated_at DESC LIMIT 1",
                params![image],
                |row| {
                    Ok(WarmExec {
                        run_id: row.get(0)?,
                        image: row.get(1)?,
                        pod_id: row.get(2)?,
                        host: row.get(3)?,
                        port: row.get::<_, i64>(4)? as u16,
                        warm_until: row.get(5)?,
                    })
                },
            )
            .optional()
            .map_err(Into::into)
    }

    /// Test-only view of recorded pod ids of nonterminal runs. Production reap uses [`Self::reap_protection`].
    #[cfg(test)]
    pub fn live_pod_ids(&self) -> Result<Vec<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT pod_id FROM runs WHERE pod_id IS NOT NULL AND status NOT IN ('done','failed','aborted')",
        )?;
        let ids = stmt
            .query_map([], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(ids)
    }

    /// Snapshot every nonterminal run claim after the cloud pod list has been read (the caller owns that
    /// ordering). A pod that appears before `on_create` stores `pod_id` is protected by `run_ids` when its
    /// name is run-specific; with a generic name from an older binary, any unrecorded live claim makes
    /// normal reap defer deletion for this pass.
    pub fn reap_protection(&self) -> Result<ReapProtection> {
        let mut stmt = self.conn.prepare(
            "SELECT id, pod_id FROM runs WHERE status NOT IN ('done','failed','aborted')",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        let mut protection = ReapProtection::default();
        for (run_id, pod_id) in rows {
            protection.run_ids.push(run_id);
            match pod_id {
                Some(id) => protection.pod_ids.push(id),
                None => protection.has_unrecorded = true,
            }
        }
        Ok(protection)
    }

    /// The pod_id recorded for a run, if any (card 230). The SIGINT/SIGTERM handler calls this on a fresh
    /// connection because the main thread's `Connection` is not `Sync`. `Ok(None)` when there is no pod
    /// or row; an error makes the handler skip teardown.
    pub fn pod_id_for_run(&self, run_id: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row(
                "SELECT pod_id FROM runs WHERE id = ?",
                params![run_id],
                |row| row.get::<_, Option<String>>(0),
            )
            .optional()?
            .flatten())
    }

    #[cfg(test)]
    pub fn run_status(&self, run_id: &str) -> Result<Option<String>> {
        self.conn
            .query_row(
                "SELECT status FROM runs WHERE id=?1",
                params![run_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    /// Recent runs with per-model progress for the web dashboard: one query for runs, one per run for models.
    pub fn dashboard(&self, limit: i64) -> Result<Vec<RunView>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, created_at, updated_at, git_ref, scenario, gpu_types, image, status, pod_id, ssh_host, ssh_port, kind
             FROM runs ORDER BY created_at DESC LIMIT ?1",
        )?;
        let mut runs = stmt
            .query_map([limit], |row| {
                Ok(RunView {
                    id: row.get(0)?,
                    created_at: row.get(1)?,
                    updated_at: row.get(2)?,
                    git_ref: row.get(3)?,
                    scenario: row.get(4)?,
                    gpu_types: row.get(5)?,
                    image: row.get(6)?,
                    status: row.get(7)?,
                    pod_id: row.get(8)?,
                    ssh_host: row.get(9)?,
                    ssh_port: row.get::<_, Option<i64>>(10)?.map(|p| p as u16),
                    kind: row.get(11)?,
                    models: Vec::new(),
                })
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for r in &mut runs {
            let mut ms = self.conn.prepare(
                "SELECT model, setup_done, sweep_done, gathered, pod_run_id
                 FROM model_runs WHERE run_id=?1 ORDER BY model",
            )?;
            r.models = ms
                .query_map([&r.id], |row| {
                    Ok(ModelView {
                        model: row.get(0)?,
                        setup_done: row.get::<_, i64>(1)? != 0,
                        sweep_done: row.get::<_, i64>(2)? != 0,
                        gathered: row.get::<_, i64>(3)? != 0,
                        pod_run_id: row.get(4)?,
                    })
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
        }
        Ok(runs)
    }

    /// All recent runs for the `status` command.
    pub fn recent_runs(&self, limit: i64) -> Result<Vec<(Run, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, git_ref, scenario, models, gpu_types, image, status, pod_id, kind, repeats, updated_at
             FROM runs ORDER BY created_at DESC LIMIT ?1",
        )?;
        let rows = stmt
            .query_map([limit], |row| {
                let run = Self::row_to_run(row)?;
                let updated: String = row.get(10)?;
                Ok((run, updated))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }
}
