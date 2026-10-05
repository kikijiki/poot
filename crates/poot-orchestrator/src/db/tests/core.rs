use super::*;

/// The state DB is shared across worktrees, so concurrent writers need a busy timeout to avoid an
/// instant "database is locked"; assert the connection carries it.
#[test]
fn open_sets_busy_timeout_and_wal() {
    let tmp = std::env::temp_dir().join("poot-orch-busy-timeout-test.db");
    let _ = std::fs::remove_file(&tmp);
    let db = Db::open(tmp.to_str().unwrap()).unwrap();
    let ms: i64 = db
        .conn
        .query_row("PRAGMA busy_timeout", [], |r| r.get(0))
        .unwrap();
    assert_eq!(ms, 10_000);
    let mode: String = db
        .conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!(mode, "wal"); // a file-backed db keeps WAL; :memory: cannot
    drop(db);
    let _ = std::fs::remove_file(&tmp);
}

#[test]
fn open_migrations_are_idempotent() {
    let tmp = std::env::temp_dir().join(format!(
        "poot-orch-migration-idempotent-{}.db",
        std::process::id()
    ));
    let _ = std::fs::remove_file(&tmp);
    drop(Db::open(tmp.to_str().unwrap()).unwrap());
    let db = Db::open(tmp.to_str().unwrap()).unwrap();
    db.create_run_with_owner(
        &exec_run("exec-owned-after-reopen", "running"),
        Some(&owner(7, "8")),
    )
    .unwrap();
    assert_eq!(
        db.abort_stale_execs_with(false, |_| OwnerLiveness::Unknown)
            .unwrap()
            .kept_unknown,
        1
    );
    drop(db);
    let _ = std::fs::remove_file(&tmp);
}

/// The schema as a state database created before the verify-batch tables were removed has it: the run,
/// model and pod tables plus the five batch tables, at `user_version` 0.
const PRE_CARD_508_SCHEMA: &str = r#"
    CREATE TABLE runs (
        id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
        git_ref TEXT NOT NULL, scenario TEXT NOT NULL, models TEXT NOT NULL, gpu_types TEXT NOT NULL,
        image TEXT NOT NULL, status TEXT NOT NULL, pod_id TEXT, ssh_host TEXT, ssh_port INTEGER,
        kind TEXT NOT NULL DEFAULT 'sweep', warm_until TEXT, owner_hostname TEXT, owner_pid INTEGER,
        owner_boot_id TEXT, owner_start_ticks TEXT
    );
    CREATE TABLE model_runs (
        run_id TEXT NOT NULL, model TEXT NOT NULL, setup_done INTEGER NOT NULL DEFAULT 0,
        sweep_done INTEGER NOT NULL DEFAULT 0, pod_run_id TEXT, gathered INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY (run_id, model)
    );
    CREATE TABLE pods (
        id TEXT PRIMARY KEY, gpu_type TEXT NOT NULL, cloud TEXT NOT NULL, image TEXT NOT NULL,
        status TEXT NOT NULL, ssh_host TEXT, ssh_port INTEGER, current_run TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE batches (
        id TEXT PRIMARY KEY, run_id TEXT NOT NULL, pod_id TEXT, status TEXT NOT NULL,
        phase TEXT NOT NULL DEFAULT 'admitted', owner_hostname TEXT, owner_pid INTEGER,
        owner_boot_id TEXT, owner_start_ticks TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
    );
    CREATE TABLE batch_jobs (
        batch_id TEXT NOT NULL, job_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, cache_root TEXT,
        PRIMARY KEY (batch_id, job_id)
    );
    CREATE TABLE batch_steps (
        batch_id TEXT NOT NULL, step_id TEXT NOT NULL, job_id TEXT NOT NULL, resource TEXT NOT NULL,
        phase TEXT NOT NULL DEFAULT 'pending', blocked_by TEXT, log_path TEXT,
        created_at TEXT NOT NULL, updated_at TEXT NOT NULL, PRIMARY KEY (batch_id, step_id)
    );
    CREATE TABLE device_leases (
        batch_id TEXT NOT NULL, device_id TEXT NOT NULL, step_id TEXT, holder_token TEXT,
        generation INTEGER NOT NULL DEFAULT 0, held INTEGER NOT NULL DEFAULT 0, acquired_at TEXT,
        updated_at TEXT NOT NULL, PRIMARY KEY (batch_id, device_id)
    );
    CREATE TABLE batch_phase_intervals (
        id INTEGER PRIMARY KEY AUTOINCREMENT, batch_id TEXT NOT NULL, step_id TEXT NOT NULL,
        kind TEXT NOT NULL, started_at_ms INTEGER NOT NULL, ended_at_ms INTEGER
    );
    CREATE INDEX idx_batch_phase_intervals ON batch_phase_intervals(batch_id, kind, started_at_ms);

    INSERT INTO runs (id, created_at, updated_at, git_ref, scenario, models, gpu_types, image, status,
                      pod_id, ssh_host, ssh_port, kind)
        VALUES ('exec-old', 't0', 't0', 'abc', '--test', '', 'NVIDIA L40S', 'img', 'running',
                'pod-old', '1.2.3.4', 22, 'exec');
    INSERT INTO pods VALUES ('pod-old', 'NVIDIA L40S', 'SECURE', 'img', 'busy', '1.2.3.4', 22,
                             'exec-old', 't0', 't0');
    INSERT INTO batches (id, run_id, pod_id, status, created_at, updated_at)
        VALUES ('batch-old', 'exec-old', 'pod-batch', 'running', 't0', 't0');
    INSERT INTO batch_jobs (batch_id, job_id, created_at, updated_at) VALUES ('batch-old', 'j', 't0', 't0');
    INSERT INTO batch_steps (batch_id, step_id, job_id, resource, created_at, updated_at)
        VALUES ('batch-old', 's', 'j', 'gpu-exclusive', 't0', 't0');
    INSERT INTO device_leases (batch_id, device_id, held, updated_at) VALUES ('batch-old', 'gpu-0', 1, 't0');
    INSERT INTO batch_phase_intervals (batch_id, step_id, kind, started_at_ms)
        VALUES ('batch-old', 's', 'prep', 1);
"#;

/// The batch and lease tables `PRE_CARD_508_SCHEMA` holds, as SQLite reports them.
fn batch_tables(db: &Db) -> Vec<String> {
    let mut stmt = db
        .conn
        .prepare(
            "SELECT name FROM sqlite_master WHERE type='table'
             AND (name LIKE 'batch%' OR name='device_leases') ORDER BY name",
        )
        .unwrap();
    stmt.query_map([], |row| row.get(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<String>>>()
        .unwrap()
}

/// SC-005: a state database created before the batch tables were removed, holding batch rows and one
/// `exec` run and pod, opens: the batch tables are gone, the run and pod rows survive, and a second open
/// changes nothing. Leaving out the drops keeps the five tables and the first assertion fails.
#[test]
fn a_pre_card_508_state_database_loses_its_batch_tables_and_keeps_its_exec_run_and_pod() {
    let tmp =
        std::env::temp_dir().join(format!("poot-orch-pre-card-508-{}.db", std::process::id()));
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", tmp.to_str().unwrap()));
    }
    Connection::open(&tmp)
        .unwrap()
        .execute_batch(PRE_CARD_508_SCHEMA)
        .unwrap();
    let before = Db::open(tmp.to_str().unwrap()).unwrap();
    drop(before);

    let db = Db::open(tmp.to_str().unwrap()).unwrap();
    assert_eq!(
        batch_tables(&db),
        Vec::<String>::new(),
        "the batch tables are dropped"
    );
    let runs = db.recent_runs(10).unwrap();
    assert_eq!(runs.len(), 1, "the exec run survives");
    let run = &runs[0].0;
    assert_eq!(
        (run.id.as_str(), run.kind.as_str(), run.status.as_str()),
        ("exec-old", "exec", "running")
    );
    assert_eq!(run.pod_id.as_deref(), Some("pod-old"));
    let pods = db.all_pods(10).unwrap();
    assert_eq!(pods.len(), 1, "the pod row survives");
    assert_eq!(
        (
            pods[0].id.as_str(),
            pods[0].status.as_str(),
            pods[0].current_run.as_deref()
        ),
        ("pod-old", "busy", Some("exec-old"))
    );
    let version = |db: &Db| -> i64 {
        db.conn
            .query_row("PRAGMA user_version", [], |row| row.get(0))
            .unwrap()
    };
    assert_eq!(version(&db), 1, "the migration recorded its version");
    drop(db);

    // A second open changes nothing: same tables, same rows, same version.
    let db = Db::open(tmp.to_str().unwrap()).unwrap();
    assert_eq!(batch_tables(&db), Vec::<String>::new());
    assert_eq!(db.recent_runs(10).unwrap().len(), 1);
    assert_eq!(db.all_pods(10).unwrap().len(), 1);
    assert_eq!(version(&db), 1);
    drop(db);
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", tmp.to_str().unwrap()));
    }
}

/// The state id is minted once per database file, committed, and different per database. Returning
/// a fresh id on every call, or one shared id, turns this red.
#[test]
fn state_id_is_stable_per_database_and_distinct_between_databases() {
    let dir = std::env::temp_dir().join(format!("poot-orch-state-id-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path_a = dir.join("a.db");
    let path_b = dir.join("b.db");
    let first = Db::open(path_a.to_str().unwrap())
        .unwrap()
        .state_id()
        .unwrap();
    let db_a = Db::open(path_a.to_str().unwrap()).unwrap();
    assert_eq!(db_a.state_id().unwrap(), first, "reopening keeps the id");
    assert_eq!(db_a.state_id().unwrap(), first, "asking twice keeps the id");
    assert!(
        first.len() == 8
            && first
                .bytes()
                .all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()),
        "eight lowercase hex digits, got {first:?}"
    );
    let other = Db::open(path_b.to_str().unwrap())
        .unwrap()
        .state_id()
        .unwrap();
    // Two 32-bit ids collide with probability 2^-32.
    assert_ne!(other, first, "another database has its own id");
    assert_eq!(
        db_a.pod_owner_prefix().unwrap(),
        db_a.pod_owner_prefix().unwrap(),
        "the prefix is the same on every ask"
    );
    std::fs::remove_dir_all(&dir).unwrap();
}
