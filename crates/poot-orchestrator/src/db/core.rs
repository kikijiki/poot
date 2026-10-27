use super::{Connection, Result, SCHEMA_VERSION, bail};

pub(crate) struct Db {
    pub(super) conn: Connection,
}

impl Db {
    pub(crate) fn open(path: &str) -> Result<Db> {
        let conn = Connection::open(path)?;
        let version: i64 = conn.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        let has_runs: bool = conn.query_row(
            "SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs')",
            [],
            |row| row.get(0),
        )?;
        // No migrations: a database from another schema is rejected, not upgraded.
        if has_runs && version != SCHEMA_VERSION {
            bail!(
                "state database {path} has schema version {version}, this build needs {SCHEMA_VERSION}: \
                 delete the file (it only holds run history) and run again"
            );
        }
        conn.execute_batch(
            r#"
            -- The default state DB is machine-wide shared (see main.rs `state_dir_from`), so several
            -- orchestrator processes from different git worktrees write concurrently. WAL lets readers
            -- run during a write, but two writers still serialize on the write lock; without a busy
            -- timeout the loser fails instantly with "database is locked". Wait instead - our writes are
            -- tiny single-row commits, so 10s is far more than enough. Set before journal_mode: taking
            -- WAL needs an exclusive lock itself, so it is the first statement that can hit SQLITE_BUSY.
            PRAGMA busy_timeout=10000;
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS runs (
                id          TEXT PRIMARY KEY,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                git_ref     TEXT NOT NULL,
                scenario    TEXT NOT NULL,
                models      TEXT NOT NULL,   -- comma-separated
                gpu_types   TEXT NOT NULL,
                image       TEXT NOT NULL,
                status      TEXT NOT NULL,
                pod_id      TEXT,
                ssh_host    TEXT,
                ssh_port    INTEGER,
                kind        TEXT NOT NULL DEFAULT 'sweep',   -- sweep | exec
                warm_until  TEXT,
                -- the local process that created the run; a run is resumed only once this process is gone
                owner_hostname    TEXT,
                owner_pid         INTEGER,
                owner_boot_id     TEXT,
                owner_start_ticks TEXT,
                owner_pid_ns      TEXT,
                repeats     INTEGER NOT NULL DEFAULT 1,
                set_id      TEXT             -- the name of the result sets (the failed run this one resumes)
            );
            CREATE TABLE IF NOT EXISTS model_runs (
                run_id      TEXT NOT NULL,
                model       TEXT NOT NULL,
                setup_done  INTEGER NOT NULL DEFAULT 0,
                sweep_done  INTEGER NOT NULL DEFAULT 0,
                pod_run_id  TEXT,
                gathered    INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (run_id, model)
            );
            -- spec 030: pods (workers) get their own lifecycle, decoupled from runs. A pod can serve many
            -- runs over its life; a run is ASSIGNED a pod (runs.pod_id) rather than owning provisioning.
            CREATE TABLE IF NOT EXISTS pods (
                id          TEXT PRIMARY KEY,
                gpu_type    TEXT NOT NULL,
                cloud       TEXT NOT NULL,
                cost_per_hr REAL,            -- USD per hour as RunPod reported it at create; NULL if it did not
                image       TEXT NOT NULL,
                status      TEXT NOT NULL,   -- provisioning | idle | busy | terminated
                ssh_host    TEXT,
                ssh_port    INTEGER,
                current_run TEXT,
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL
            );
            "#,
        )?;
        conn.execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION}"))?;
        Ok(Db { conn })
    }
}
