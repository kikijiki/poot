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

/// There are no migrations: a database file of another schema version is refused with the way out, and a fresh
/// one records the current version and reopens unchanged. Mutation: skip the version check and the old file opens.
#[test]
fn a_state_database_of_another_schema_is_rejected_and_a_fresh_one_reopens() {
    let tmp = std::env::temp_dir().join(format!("poot-orch-schema-{}.db", std::process::id()));
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{}{suffix}", tmp.to_str().unwrap()));
    }
    let path = tmp.to_str().unwrap();
    Connection::open(&tmp)
        .unwrap()
        .execute_batch("CREATE TABLE runs (id TEXT PRIMARY KEY, status TEXT NOT NULL);")
        .unwrap();
    let error = Db::open(path).err().expect("an old schema is refused");
    assert!(
        format!("{error:#}").contains("delete the file"),
        "{error:#}"
    );
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{path}{suffix}"));
    }

    drop(Db::open(path).unwrap());
    let db = Db::open(path).unwrap();
    let version: i64 = db
        .conn
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(version, SCHEMA_VERSION);
    drop(db);
    for suffix in ["", "-wal", "-shm"] {
        let _ = std::fs::remove_file(format!("{path}{suffix}"));
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
