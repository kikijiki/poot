use super::*;

pub struct Db {
    pub(super) conn: Connection,
}

fn add_column_if_missing(conn: &Connection, sql: &str, context: &str) -> Result<()> {
    match conn.execute(sql, []) {
        Ok(_) => Ok(()),
        Err(e) if e.to_string().contains("duplicate column name") => Ok(()),
        Err(e) => Err(e).with_context(|| context.to_string()),
    }
}

impl Db {
    pub fn open(path: &str) -> Result<Db> {
        let conn = Connection::open(path)?;
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
                ssh_port    INTEGER
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
        // SQLite has no ADD COLUMN IF NOT EXISTS: ignore duplicate-column only; any other failure must
        // be reported.
        add_column_if_missing(
            &conn,
            "ALTER TABLE runs ADD COLUMN kind TEXT NOT NULL DEFAULT 'sweep'",
            "add runs.kind",
        )?;
        add_column_if_missing(
            &conn,
            "ALTER TABLE runs ADD COLUMN warm_until TEXT",
            "add runs.warm_until",
        )?;
        add_column_if_missing(
            &conn,
            "ALTER TABLE runs ADD COLUMN owner_hostname TEXT",
            "add runs.owner_hostname",
        )?;
        add_column_if_missing(
            &conn,
            "ALTER TABLE runs ADD COLUMN owner_pid INTEGER",
            "add runs.owner_pid",
        )?;
        add_column_if_missing(
            &conn,
            "ALTER TABLE runs ADD COLUMN owner_boot_id TEXT",
            "add runs.owner_boot_id",
        )?;
        add_column_if_missing(
            &conn,
            "ALTER TABLE runs ADD COLUMN owner_start_ticks TEXT",
            "add runs.owner_start_ticks",
        )?;
        add_column_if_missing(
            &conn,
            "ALTER TABLE runs ADD COLUMN repeats INTEGER NOT NULL DEFAULT 1",
            "add runs.repeats",
        )?;
        migrate(&conn)?;
        Ok(Db { conn })
    }
}
