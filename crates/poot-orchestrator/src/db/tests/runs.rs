use super::*;

#[test]
fn exec_owner_roundtrips_on_create_and_update() {
    let db = Db::open(":memory:").unwrap();
    let first = owner(123, "456");
    db.create_run_with_owner(&exec_run("exec-owned", "running"), Some(&first))
        .unwrap();
    db.set_pod("exec-owned", "pod-owned", Some("1.2.3.4"), Some(22))
        .unwrap();

    let report = db
        .abort_stale_execs_with(false, |seen| {
            assert_eq!(seen, &first);
            OwnerLiveness::Live
        })
        .unwrap();
    assert_eq!(
        report,
        ExecReapReport {
            aborted: 0,
            kept_live: 1,
            kept_unknown: 0,
        }
    );
    assert_eq!(
        db.live_pod_ids().unwrap(),
        vec!["pod-owned".to_string()],
        "a live owner keeps its pod protected"
    );

    let second = owner(124, "999");
    db.set_exec_owner("exec-owned", &second).unwrap();
    assert_eq!(
        db.abort_stale_execs_with(false, |_| OwnerLiveness::Dead)
            .unwrap()
            .aborted,
        1
    );
    assert_eq!(
        db.run_status("exec-owned").unwrap().as_deref(),
        Some("aborted")
    );
}

#[test]
fn local_process_identity_uses_start_ticks_not_pid_alone() {
    let current = ExecOwner::current().unwrap();
    assert_eq!(current.local_liveness(), OwnerLiveness::Live);

    let mut reused_pid = current;
    reused_pid.start_ticks.push('9');
    assert_eq!(
        reused_pid.local_liveness(),
        OwnerLiveness::Dead,
        "same PID with a different process start identity is not the original owner"
    );
}

#[test]
fn process_start_tick_failures_are_conservative_except_missing_process() {
    assert_eq!(
        process_start_liveness(Ok(Some("10".to_string())), "10"),
        OwnerLiveness::Live
    );
    assert_eq!(
        process_start_liveness(Ok(Some("11".to_string())), "10"),
        OwnerLiveness::Dead,
        "PID reuse with a different start tick is definitive"
    );
    assert_eq!(
        process_start_liveness(Ok(None), "10"),
        OwnerLiveness::Dead,
        "a missing /proc/<pid>/stat is definitive"
    );
    assert_eq!(
        process_start_liveness(Err(anyhow::anyhow!("malformed stat")), "10"),
        OwnerLiveness::Unknown,
        "permission, I/O, and parse failures must not release pod protection"
    );
}

#[test]
fn normal_reap_aborts_dead_exec_owner_but_keeps_sweeps() {
    let db = Db::open(":memory:").unwrap();
    db.create_run_with_owner(&exec_run("exec-dead", "running"), Some(&owner(1, "10")))
        .unwrap();
    db.set_pod("exec-dead", "pod-dead", Some("1.2.3.4"), Some(22))
        .unwrap();
    let sweep = Run {
        id: "run-live".into(),
        kind: "sweep".into(),
        git_ref: "abc".into(),
        scenario: "decode".into(),
        repeats: 1,
        models: vec![],
        gpu_types: "g".into(),
        image: "img".into(),
        status: "sweeping".into(),
        pod_id: None,
    };
    db.create_run(&sweep).unwrap();

    let report = db
        .abort_stale_execs_with(false, |_| OwnerLiveness::Dead)
        .unwrap();
    assert_eq!(report.aborted, 1);
    assert_eq!(
        db.run_status("exec-dead").unwrap().as_deref(),
        Some("aborted")
    );
    assert_eq!(
        db.run_status("run-live").unwrap().as_deref(),
        Some("sweeping")
    );
    assert!(db.live_pod_ids().unwrap().is_empty());
}

#[test]
fn normal_reap_keeps_unknown_or_legacy_exec_owner_protected() {
    let db = Db::open(":memory:").unwrap();
    db.create_run(&exec_run("exec-legacy", "running")).unwrap();
    db.set_pod("exec-legacy", "pod-legacy", Some("1.2.3.4"), Some(22))
        .unwrap();

    let report = db
        .abort_stale_execs_with(false, |_| OwnerLiveness::Dead)
        .unwrap();
    assert_eq!(
        report,
        ExecReapReport {
            aborted: 0,
            kept_live: 0,
            kept_unknown: 1,
        }
    );
    assert_eq!(
        db.live_pod_ids().unwrap(),
        vec!["pod-legacy".to_string()],
        "ownerless legacy rows are not released by normal reap"
    );
}

#[test]
fn pid_reuse_identity_mismatch_is_reclaimable() {
    let db = Db::open(":memory:").unwrap();
    let original = owner(321, "111");
    db.create_run_with_owner(&exec_run("exec-reused-pid", "running"), Some(&original))
        .unwrap();
    db.set_pod("exec-reused-pid", "pod-reused", Some("1.2.3.4"), Some(22))
        .unwrap();

    let report = db
        .abort_stale_execs_with(false, |seen| {
            assert_eq!(seen.pid, 321);
            assert_eq!(seen.start_ticks, "111");
            OwnerLiveness::Dead
        })
        .unwrap();
    assert_eq!(report.aborted, 1);
    assert!(db.live_pod_ids().unwrap().is_empty());
}

#[test]
fn stale_exec_abort_uses_compare_and_set_identity() {
    let db = Db::open(":memory:").unwrap();
    db.create_run_with_owner(&exec_run("exec-race", "running"), Some(&owner(321, "111")))
        .unwrap();
    db.set_pod("exec-race", "pod-race", Some("1.2.3.4"), Some(22))
        .unwrap();

    let report = db
        .abort_stale_execs_with(false, |seen| {
            assert_eq!(seen.start_ticks, "111");
            db.set_exec_owner("exec-race", &owner(654, "222")).unwrap();
            OwnerLiveness::Dead
        })
        .unwrap();
    assert_eq!(report.aborted, 0);
    assert_eq!(
        db.run_status("exec-race").unwrap().as_deref(),
        Some("running"),
        "a row whose owner changed after selection must not be overwritten"
    );
    assert_eq!(db.live_pod_ids().unwrap(), vec!["pod-race".to_string()]);
}

#[test]
fn stale_exec_abort_does_not_overwrite_terminal_or_rewarmed_rows() {
    let db = Db::open(":memory:").unwrap();
    db.create_run_with_owner(
        &exec_run("exec-terminal-race", "running"),
        Some(&owner(1, "10")),
    )
    .unwrap();
    db.set_pod(
        "exec-terminal-race",
        "pod-terminal",
        Some("1.2.3.4"),
        Some(22),
    )
    .unwrap();
    let report = db
        .abort_stale_execs_with(false, |_| {
            db.set_status("exec-terminal-race", "done").unwrap();
            OwnerLiveness::Dead
        })
        .unwrap();
    assert_eq!(report.aborted, 0);
    assert_eq!(
        db.run_status("exec-terminal-race").unwrap().as_deref(),
        Some("done")
    );

    db.create_run_with_owner(
        &exec_run("exec-warm-race", "running"),
        Some(&owner(2, "20")),
    )
    .unwrap();
    db.set_pod("exec-warm-race", "pod-warm-race", Some("1.2.3.4"), Some(22))
        .unwrap();
    db.set_warm("exec-warm-race", &crate::clock::iso_from_secs(1))
        .unwrap();
    let report = db
        .abort_stale_execs_with(false, |_| {
            db.set_warm("exec-warm-race", &crate::clock::iso_in_minutes(30))
                .unwrap();
            OwnerLiveness::Dead
        })
        .unwrap();
    assert_eq!(report.aborted, 0);
    assert_eq!(
        db.run_status("exec-warm-race").unwrap().as_deref(),
        Some("warm")
    );
    assert_eq!(
        db.live_pod_ids().unwrap(),
        vec!["pod-warm-race".to_string()]
    );
}

#[test]
fn warm_exec_policy_and_force_semantics_are_explicit() {
    let db = Db::open(":memory:").unwrap();
    db.create_run_with_owner(&exec_run("exec-warm", "running"), Some(&owner(1, "10")))
        .unwrap();
    db.set_pod("exec-warm", "pod-warm", Some("1.2.3.4"), Some(22))
        .unwrap();
    db.set_warm("exec-warm", &crate::clock::iso_in_minutes(30))
        .unwrap();

    let report = db
        .abort_stale_execs_with(false, |_| OwnerLiveness::Dead)
        .unwrap();
    assert_eq!(report.aborted, 0);
    assert_eq!(report.kept_live, 1);
    assert_eq!(db.live_pod_ids().unwrap(), vec!["pod-warm".to_string()]);

    let report = db
        .abort_stale_execs_with(true, |_| OwnerLiveness::Live)
        .unwrap();
    assert_eq!(report.aborted, 1);
    assert!(db.live_pod_ids().unwrap().is_empty());
}

#[test]
fn warm_exec_adoption_is_single_winner() {
    let db = Db::open(":memory:").unwrap();
    db.create_run_with_owner(&exec_run("exec-warm", "running"), Some(&owner(1, "10")))
        .unwrap();
    db.set_pod("exec-warm", "pod-warm", Some("1.2.3.4"), Some(22))
        .unwrap();
    db.set_warm("exec-warm", &crate::clock::iso_in_minutes(30))
        .unwrap();
    db.create_run_with_owner(&exec_run("exec-new-a", "new"), Some(&owner(2, "20")))
        .unwrap();
    db.create_run_with_owner(&exec_run("exec-new-b", "new"), Some(&owner(3, "30")))
        .unwrap();

    let observed_by_a = db.find_warm_exec("img").unwrap().unwrap();
    let observed_by_b = observed_by_a.clone();
    assert!(
        db.try_adopt_warm_exec(&observed_by_a, "exec-new-a", &owner(2, "20"))
            .unwrap()
    );
    assert!(
        !db.try_adopt_warm_exec(&observed_by_b, "exec-new-b", &owner(3, "30"))
            .unwrap(),
        "a stale warm observation must not adopt a pod already claimed by another exec"
    );
    assert_eq!(db.run_status("exec-warm").unwrap().as_deref(), Some("done"));
    assert_eq!(
        db.run_status("exec-new-a").unwrap().as_deref(),
        Some("running")
    );
    assert_eq!(
        db.pod_id_for_run("exec-new-a").unwrap().as_deref(),
        Some("pod-warm")
    );
    assert_eq!(db.run_status("exec-new-b").unwrap().as_deref(), Some("new"));
    assert_eq!(db.pod_id_for_run("exec-new-b").unwrap(), None);
}

#[test]
fn warm_exec_cleanup_is_cas_protected() {
    let db = Db::open(":memory:").unwrap();
    db.create_run_with_owner(&exec_run("exec-warm", "running"), Some(&owner(1, "10")))
        .unwrap();
    db.set_pod("exec-warm", "pod-warm", Some("1.2.3.4"), Some(22))
        .unwrap();
    db.set_warm("exec-warm", &crate::clock::iso_in_minutes(30))
        .unwrap();
    db.create_run_with_owner(&exec_run("exec-new", "new"), Some(&owner(2, "20")))
        .unwrap();

    let stale_cleanup_view = db.find_warm_exec("img").unwrap().unwrap();
    assert!(
        db.try_adopt_warm_exec(&stale_cleanup_view, "exec-new", &owner(2, "20"))
            .unwrap()
    );
    assert!(
        !db.try_abort_warm_exec(&stale_cleanup_view).unwrap(),
        "cleanup from a stale observation must not abort an adopted warm pod"
    );
    assert_eq!(db.run_status("exec-warm").unwrap().as_deref(), Some("done"));
    assert_eq!(
        db.run_status("exec-new").unwrap().as_deref(),
        Some("running")
    );
    assert_eq!(db.live_pod_ids().unwrap(), vec!["pod-warm".to_string()]);
}

/// Every repeat of every model's sweep is a snapshot dir of the run: the dashboard and the commit read this list.
#[test]
fn a_run_of_three_repeats_has_three_snapshot_dirs_per_model() {
    let db = Db::open(":memory:").unwrap();
    let run = Run {
        id: "run-1".into(),
        kind: "sweep".into(),
        git_ref: "abc".into(),
        scenario: "decode".into(),
        repeats: 3,
        models: vec!["m1".into(), "m2".into()],
        gpu_types: "g".into(),
        image: "img".into(),
        status: "sweeping".into(),
        pod_id: None,
    };
    db.create_run(&run).unwrap();
    assert!(
        db.snapshot_dirs("run-1").unwrap().is_empty(),
        "nothing swept yet"
    );
    for model in ["m1", "m2"] {
        db.set_model_sweep("run-1", model, true, Some(&format!("run-1-{model}")))
            .unwrap();
    }

    assert_eq!(
        db.snapshot_dirs("run-1").unwrap(),
        [
            "run-1-m1-r1",
            "run-1-m1-r2",
            "run-1-m1-r3",
            "run-1-m2-r1",
            "run-1-m2-r2",
            "run-1-m2-r3"
        ]
    );
    assert_eq!(db.active_run().unwrap().unwrap().repeats, 3);
    assert_eq!(
        run.model_snapshot_dirs("run-1-m1"),
        ["run-1-m1-r1", "run-1-m1-r2", "run-1-m1-r3"]
    );
}

/// One repeat is the run directory itself, the layout every earlier run used.
#[test]
fn a_single_repeat_run_has_one_snapshot_dir_per_model() {
    let db = Db::open(":memory:").unwrap();
    let run = Run {
        id: "run-1".into(),
        kind: "sweep".into(),
        git_ref: "abc".into(),
        scenario: "decode".into(),
        repeats: 1,
        models: vec!["m1".into()],
        gpu_types: "g".into(),
        image: "img".into(),
        status: "sweeping".into(),
        pod_id: None,
    };
    db.create_run(&run).unwrap();
    db.set_model_sweep("run-1", "m1", true, Some("run-1-m1"))
        .unwrap();

    assert_eq!(db.snapshot_dirs("run-1").unwrap(), ["run-1-m1"]);
}
