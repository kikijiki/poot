//! poot-orchestrator: a local, resumable runner for poot benchmark sweeps on rented RunPod GPUs.
//!
//! Builds the poot runner from a git ref, provisions a GPU pod (with retry), sets it up (binary,
//! harness, models, GGUFs), runs the decode-curve sweep per model in tmux, copies the result snapshots
//! into benchmarks/results/ (which the Docusaurus dashboard ingests), and always tears the pod down.
//!
//! Every phase and per-model flag is committed to a SQLite run-state DB, so a crash resumes. The sweep
//! runs in tmux on the pod and keeps going while the orchestrator is down; resume reattaches. The pod id
//! is persisted as soon as the pod is created, and `reap` (run at the start of every `run`, or on demand)
//! terminates any pod with our name prefix that is not tied to an in-progress run.
//!
//! Decision logic (manifest parse, model selection, reap targeting, resume) is in pure, unit-tested
//! functions; the I/O wrappers (db, runpod, ssh) are thin.
//!
//! The binary is a thin `main`; the crate is a library so the pod-name types have doctests that prove a
//! pod cannot be created under a name `reap` does not own.

mod db;
mod image_resolve;
mod runpod;
mod ssh;
mod web;

mod cli;
mod clock;
mod logging;
mod manifest;
mod workflow;

use anyhow::Result;
use clap::Parser;

use cli::{Cli, Command};
use db::Db;
use logging::{log, state_dir};
use runpod::RunPod;
use workflow::{cmd_exec, cmd_image, cmd_logs, cmd_run, cmd_status, reap, require_key};

pub use db::PodOwnerPrefix;
pub use runpod::CreateSpec;

/// The orchestrator command line: parse `argv` and run the command.
pub fn run_from_args() -> Result<()> {
    run(Cli::parse())
}

fn run(cli: Cli) -> Result<()> {
    // Machine-wide state dir, not derived from the checkout (see `state_dir_from`).
    let state_db = cli
        .state_db
        .unwrap_or_else(|| state_dir().join("state.db").to_string_lossy().into_owned());
    let db = Db::open(&state_db)?;
    match cli.command {
        Command::Status => cmd_status(&db, &state_db),
        Command::Reap(a) => {
            let rp = RunPod::new(require_key(a.runpod_api_key)?)?;
            let n = reap(&db, &rp, a.all)?;
            log(format!("reaped {n} orphan pod(s)"));
            Ok(())
        }
        Command::Run(a) => cmd_run(db, state_db, a),
        Command::Serve(s) => web::serve(db, &s.addr),
        Command::Image(a) => cmd_image(&a),
        Command::Logs(a) => cmd_logs(&a),
        Command::Exec(a) => cmd_exec(&db, a),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::{exec_run_id_from, iso_from_secs, iso_in_minutes, now_iso};
    use crate::logging::{
        LogKind, LogState, classify_log_kind, classify_log_state, safe_log_name, state_dir_from,
    };
    use crate::manifest::{parse_manifest_models, select_models};
    use crate::workflow::{
        Resume, build_exec_create_env, cuda_floor_to_allowed, finish_status_output, parse_env_pair,
        reap_targets, resume_decision,
    };
    use std::path::PathBuf;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use crate::db::{ExecOwner, ReapProtection, Run};
    use crate::runpod::{CreateSpec, Pod};

    /// SC-001: the `verify-batch` subcommand no longer exists, so the parser refuses it as an unknown
    /// subcommand. Re-adding a `VerifyBatch` arm to `Command` makes the parse succeed and this red.
    #[test]
    fn verify_batch_is_an_unknown_subcommand() {
        let err = match Cli::try_parse_from([
            "poot-orchestrator",
            "verify-batch",
            "--manifest",
            "batch.toml",
            "--remote-available-parallelism",
            "4",
        ]) {
            Ok(_) => panic!("verify-batch must not parse"),
            Err(err) => err,
        };
        assert_eq!(err.kind(), clap::error::ErrorKind::InvalidSubcommand);
        assert!(err.to_string().contains("verify-batch"), "{err}");
    }

    const MANIFEST: &str = r#"
[suite]
models_dir = "/x"
[[scenario]]
id = "decode-curve"
[[model]]
id = "qwen2.5-0.5b"
hf_repo = "Qwen/Qwen2.5-0.5B-Instruct"
local_dir = "qwen2.5-0.5b-instruct"
frameworks.poot = { support = true, precision = "bf16" }
frameworks.candle = { support = true, precision = "bf16" }
[[model]]
id = "deepseek-v2"
hf_repo = "deepseek-ai/DeepSeek-V2-Lite"
frameworks.poot = { support = false, reason = "MoE host route" }
[[model]]
id = "phi-4"
hf_repo = "microsoft/phi-4"
frameworks.poot = { support = false, reason = "no Phi3 decode tracer" }
frameworks.transformers = { support = true, precision = "bf16" }
"#;

    fn pod(id: &str, name: &str, ip: Option<&str>, port22: Option<u32>) -> Pod {
        let mut pm = std::collections::HashMap::new();
        if let Some(p) = port22 {
            pm.insert("22".to_string(), p);
        }
        Pod {
            id: id.into(),
            name: name.into(),
            desired_status: "RUNNING".into(),
            public_ip: ip.map(String::from),
            port_mappings: if port22.is_some() { Some(pm) } else { None },
            machine: serde_json::Value::Null,
        }
    }

    #[test]
    fn state_dir_prefers_xdg_state_home() {
        assert_eq!(
            state_dir_from(Some("/var/lib/xdg"), Some("/users/u")),
            PathBuf::from("/var/lib/xdg/poot-orchestrator")
        );
    }

    #[test]
    fn status_treats_a_closed_output_pipe_as_a_successful_short_read() {
        let broken_pipe = std::io::Error::from(std::io::ErrorKind::BrokenPipe);
        assert!(finish_status_output(Err(broken_pipe)).is_ok());

        let other = std::io::Error::other("status sink failed");
        assert!(finish_status_output(Err(other)).is_err());
    }

    #[test]
    fn state_dir_falls_back_to_home_local_state() {
        assert_eq!(
            state_dir_from(None, Some("/users/u")),
            PathBuf::from("/users/u/.local/state/poot-orchestrator")
        );
    }

    #[test]
    fn state_dir_ignores_empty_or_relative_xdg() {
        // XDG requires an absolute path; anything else is treated as unset.
        for bad in ["", "   ", "relative/state"] {
            assert_eq!(
                state_dir_from(Some(bad), Some("/users/u")),
                PathBuf::from("/users/u/.local/state/poot-orchestrator"),
                "XDG_STATE_HOME={bad:?} should be ignored"
            );
        }
    }

    #[test]
    fn state_dir_last_resort_when_no_home() {
        assert_eq!(
            state_dir_from(None, None),
            PathBuf::from("/tmp/poot-orchestrator")
        );
        assert_eq!(
            state_dir_from(None, Some("")),
            PathBuf::from("/tmp/poot-orchestrator")
        );
    }

    /// Builds from different worktrees must resolve to the same state dir so `reap` sees every pod;
    /// only the env decides.
    #[test]
    fn state_dir_is_identical_across_worktrees() {
        let a = state_dir_from(None, Some("/users/u"));
        let b = state_dir_from(None, Some("/users/u"));
        assert_eq!(a, b);
        // not under any checkout
        assert!(!a.to_string_lossy().contains("worktree"));
        assert!(!a.to_string_lossy().contains(".orchestrator"));
    }

    #[test]
    fn parse_env_pair_splits_on_first_equals() {
        assert_eq!(
            parse_env_pair("NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics").unwrap(),
            (
                "NVIDIA_DRIVER_CAPABILITIES".into(),
                "compute,utility,graphics".into()
            )
        );
        assert_eq!(
            parse_env_pair("FOO=a=b=c").unwrap(),
            ("FOO".into(), "a=b=c".into())
        );
    }

    #[test]
    fn parse_env_pair_rejects_missing_equals_or_empty_key() {
        assert!(parse_env_pair("NOEQUALS").is_err());
        assert!(parse_env_pair("=novaluekey").is_err());
    }

    #[test]
    fn build_exec_create_env_defaults_driver_caps_and_keeps_public_key() {
        let env = build_exec_create_env("ssh-ed25519 AAAA", &[]).unwrap();
        assert_eq!(
            env.get("PUBLIC_KEY").map(String::as_str),
            Some("ssh-ed25519 AAAA")
        );
        assert_eq!(
            env.get("NVIDIA_DRIVER_CAPABILITIES").map(String::as_str),
            Some("compute,utility,graphics")
        );
    }

    #[test]
    fn build_exec_create_env_merges_overrides_and_protects_public_key() {
        let extra = vec![
            "NVIDIA_DRIVER_CAPABILITIES=all".to_string(),
            "FOO=bar".to_string(),
            "PUBLIC_KEY=should-not-win".to_string(),
        ];
        let env = build_exec_create_env(" real-key \n", &extra).unwrap();
        assert_eq!(env.get("PUBLIC_KEY").map(String::as_str), Some("real-key"));
        assert_eq!(
            env.get("NVIDIA_DRIVER_CAPABILITIES").map(String::as_str),
            Some("all")
        );
        assert_eq!(env.get("FOO").map(String::as_str), Some("bar"));
    }

    /// `build_exec_create_env` -> `create_pod_body` must put the default driver-caps pair on the REST v1
    /// `env` object (not a GraphQL key/value array).
    #[test]
    fn exec_create_pod_body_includes_default_driver_caps_as_rest_object() {
        use runpod::create_pod_body;
        let env = build_exec_create_env("ssh-ed25519 AAAA", &[]).unwrap();
        let owner = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let spec = CreateSpec {
            owner: &owner,
            run_id: "exec-test",
            attempt: 1,
            image: "ghcr.io/kikijiki/poot-bench:latest",
            gpu_type_id: "NVIDIA RTX A5000",
            cloud_type: "SECURE",
            gpu_count: 1,
            container_disk_gb: 40,
            ports: &["22/tcp"],
            env: &env,
            network_volume_id: None,
            data_center_ids: &[],
            allowed_cuda_versions: &[],
        };
        let body = create_pod_body(&spec);
        let env_val = &body["env"];
        assert!(
            env_val.is_object(),
            "REST env must be object, got {env_val}"
        );
        assert!(
            !env_val.is_array(),
            "REST env must not be a GraphQL-style [{{key,value}}] array: {env_val}"
        );
        assert_eq!(
            env_val["NVIDIA_DRIVER_CAPABILITIES"].as_str(),
            Some("compute,utility,graphics")
        );
        assert_eq!(env_val["PUBLIC_KEY"].as_str(), Some("ssh-ed25519 AAAA"));
    }

    #[test]
    fn parse_manifest_picks_models_and_support() {
        let ms = parse_manifest_models(MANIFEST).unwrap();
        assert_eq!(ms.len(), 3);
        let q = ms.iter().find(|m| m.id == "qwen2.5-0.5b").unwrap();
        assert!(q.poot_supported);
        assert!(q.any_framework_supported);
        assert_eq!(q.hf_repo.as_deref(), Some("Qwen/Qwen2.5-0.5B-Instruct"));
        // dir_name follows local_dir (matches the harness's MODELS_DIR/<dir>).
        assert_eq!(q.dir_name(), "qwen2.5-0.5b-instruct");
        let ds = ms.iter().find(|m| m.id == "deepseek-v2").unwrap();
        assert!(!ds.poot_supported);
        assert!(!ds.any_framework_supported); // only frameworks.poot is declared, and it's false
        assert_eq!(ds.dir_name(), "deepseek-v2"); // no local_dir -> falls back to id
        // phi-4: poot cannot run it, but transformers can; select_models accepts it when requested.
        let p4 = ms.iter().find(|m| m.id == "phi-4").unwrap();
        assert!(!p4.poot_supported);
        assert!(p4.any_framework_supported);
    }

    #[test]
    fn select_models_default_is_supported_only() {
        let ms = parse_manifest_models(MANIFEST).unwrap();
        let sel = select_models(&ms, None).unwrap();
        assert_eq!(
            sel.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["qwen2.5-0.5b"]
        );
    }

    #[test]
    fn select_models_rejects_unknown_and_unsupported() {
        let ms = parse_manifest_models(MANIFEST).unwrap();
        assert!(select_models(&ms, Some("nope")).is_err());
        // deepseek-v2 has no supporting framework, so it is rejected.
        assert!(select_models(&ms, Some("deepseek-v2")).is_err());
        assert_eq!(select_models(&ms, Some("qwen2.5-0.5b")).unwrap().len(), 1);
    }

    #[test]
    fn select_models_accepts_explicit_poot_unsupported_model_with_other_framework_support() {
        // phi-4: poot support = false but transformers = true. An explicit --models request succeeds;
        // only the default selection is poot-supported-only.
        let ms = parse_manifest_models(MANIFEST).unwrap();
        let sel = select_models(&ms, Some("phi-4")).unwrap();
        assert_eq!(
            sel.iter().map(|m| m.id.as_str()).collect::<Vec<_>>(),
            ["phi-4"]
        );
    }

    #[test]
    fn exec_run_ids_keep_seconds_prefix_but_are_unique_within_a_second() {
        let t0 = UNIX_EPOCH + Duration::new(1_788_523_412, 42);
        let t1 = UNIX_EPOCH + Duration::new(1_788_523_412, 43);
        let a = exec_run_id_from(t0, 1234, 0);
        let b = exec_run_id_from(t0, 1234, 1);
        let c = exec_run_id_from(t1, 1234, 0);
        let d = exec_run_id_from(t0, 5678, 0);

        assert_eq!(a, "exec-1788523412-000000042-1234-0");
        assert!(a.starts_with("exec-1788523412-"));
        assert_ne!(
            a, b,
            "same-process exec starts in the same second use the sequence"
        );
        assert_ne!(a, c, "same-second timestamps keep subsecond precision");
        assert_ne!(a, d, "parallel exec processes use the pid in the run id");
    }

    #[test]
    fn pod_parses_from_runpod_rest_json() {
        // Validates the serde field renames (publicIp, portMappings, desiredStatus).
        let json = r#"{
            "id": "abc123",
            "name": "poot-bench-run-a1",
            "desiredStatus": "RUNNING",
            "publicIp": "194.26.0.7",
            "portMappings": {"22": 40123},
            "imageName": "ghcr.io/kikijiki/poot-bench:latest",
            "extraFieldWeIgnore": 42
        }"#;
        let pod: Pod = serde_json::from_str(json).unwrap();
        assert_eq!(pod.id, "abc123");
        assert_eq!(pod.desired_status, "RUNNING");
        assert_eq!(pod.ssh_endpoint(), Some(("194.26.0.7".into(), 40123)));
        // A booting pod (no ip/ports) reports no endpoint.
        let booting: Pod =
            serde_json::from_str(r#"{"id":"x","name":"poot-bench-x","desiredStatus":"PENDING"}"#)
                .unwrap();
        assert_eq!(booting.ssh_endpoint(), None);
    }

    #[test]
    fn ssh_endpoint_needs_ip_and_port() {
        assert_eq!(
            pod("p", "n", Some("1.2.3.4"), Some(40000)).ssh_endpoint(),
            Some(("1.2.3.4".into(), 40000))
        );
        assert_eq!(pod("p", "n", None, Some(40000)).ssh_endpoint(), None);
        assert_eq!(pod("p", "n", Some("1.2.3.4"), None).ssh_endpoint(), None);
        assert_eq!(pod("p", "n", Some(""), Some(40000)).ssh_endpoint(), None);
    }

    #[test]
    fn reap_targets_kills_prefixed_orphans_only() {
        let prefix = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let pods = vec![
            pod("a", &prefix.pod_name("run1", 1), None, None), // orphan (prefixed, not live)
            pod("b", &prefix.pod_name("run2", 1), None, None), // live
            pod("c", "someones-other-pod", None, None),        // not ours - never touch
        ];
        let protection = ReapProtection {
            pod_ids: vec!["b".to_string()],
            ..ReapProtection::default()
        };
        assert_eq!(
            reap_targets(&pods, &prefix, &protection, false),
            vec!["a".to_string()]
        );
        // force kills all prefixed pods, never the foreign one.
        let mut forced = reap_targets(&pods, &prefix, &protection, true);
        forced.sort();
        assert_eq!(forced, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn reap_protects_cloud_visible_pod_before_on_create_records_id() {
        let run_id = "exec-1788530213-892720615-3077622-0";
        let prefix = Db::open(":memory:").unwrap().pod_owner_prefix().unwrap();
        let pods = vec![pod(
            "4dmn0aykzpftwj",
            &prefix.pod_name(run_id, 1),
            None,
            None,
        )];
        let protection = ReapProtection {
            run_ids: vec![run_id.to_string()],
            ..ReapProtection::default()
        };
        assert!(
            reap_targets(&pods, &prefix, &protection, false).is_empty(),
            "the run-specific name protects a cloud-visible pod before its id callback"
        );

        let old_binary_pod = vec![pod("old-pod", &prefix.pod_name("exec", 1), None, None)];
        let old_binary_protection = ReapProtection {
            has_unrecorded: true,
            ..protection
        };
        assert!(
            reap_targets(&old_binary_pod, &prefix, &old_binary_protection, false).is_empty(),
            "an unrecorded live claim conservatively protects a generic old-binary pod"
        );
    }

    #[test]
    fn post_cloud_db_snapshot_closes_incident_ordering_race() {
        let db = Db::open(":memory:").unwrap();
        let run_id = "exec-1788530213-892720615-3077622-0";
        let exec = Run {
            id: run_id.into(),
            kind: "exec".into(),
            git_ref: "885c0279".into(),
            scenario: "card 149 live provisioning".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA L40S".into(),
            image: "img".into(),
            status: "new".into(),
            pod_id: None,
        };
        db.create_run_with_owner(&exec, Some(&ExecOwner::current().unwrap()))
            .unwrap();

        // Stale pre-list snapshot (the old ID-only implementation): the live row exists but the create
        // callback has not stored its pod id yet.
        let stale = ReapProtection {
            pod_ids: db.live_pod_ids().unwrap(),
            ..ReapProtection::default()
        };
        let prefix = db.pod_owner_prefix().unwrap();
        let cloud = vec![pod(
            "4dmn0aykzpftwj",
            &prefix.pod_name("exec", 1),
            None,
            None,
        )];
        assert_eq!(
            reap_targets(&cloud, &prefix, &stale, false),
            vec!["4dmn0aykzpftwj".to_string()],
            "the old ID-only snapshot deterministically reproduces the wrongful target"
        );

        // The callback commits after the cloud snapshot; current reap reads protection afterwards.
        db.set_pod(run_id, "4dmn0aykzpftwj", None, None).unwrap();
        let current = db.reap_protection().unwrap();
        assert!(reap_targets(&cloud, &prefix, &current, false).is_empty());
        assert_eq!(db.run_status(run_id).unwrap().as_deref(), Some("new"));
    }

    #[test]
    fn resume_decision_branches() {
        let mut r = Run {
            id: "r".into(),
            kind: "sweep".into(),
            git_ref: "x".into(),
            scenario: "s".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "g".into(),
            image: "i".into(),
            status: "sweeping".into(),
            pod_id: Some("p".into()),
        };
        assert_eq!(
            resume_decision(&r, Some(("h".into(), 22))),
            Resume::Reattach("h".into(), 22)
        );
        assert_eq!(resume_decision(&r, None), Resume::AbortStale);
        r.pod_id = None;
        assert_eq!(resume_decision(&r, None), Resume::Reprovision);
    }

    #[test]
    fn db_state_machine_roundtrip() {
        let tmp = std::env::temp_dir().join(format!("pbo-test-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let db = Db::open(tmp.to_str().unwrap()).unwrap();
        let run = Run {
            id: "run-id-column".into(),
            kind: "sweep".into(),
            git_ref: "git-ref-column".into(),
            scenario: "scenario-column".into(),
            repeats: 1,
            models: vec!["model-column-a".into(), "model-column-b".into()],
            gpu_types: "gpu-types-column".into(),
            image: "image-column".into(),
            status: "new".into(),
            pod_id: None,
        };
        db.create_run(&run).unwrap();
        let assert_run_fields = |actual: &Run, pod_id: Option<&str>| {
            assert_eq!(actual.id, run.id);
            assert_eq!(actual.git_ref, run.git_ref);
            assert_eq!(actual.scenario, run.scenario);
            assert_eq!(actual.models, run.models);
            assert_eq!(actual.gpu_types, run.gpu_types);
            assert_eq!(actual.image, run.image);
            assert_eq!(actual.status, run.status);
            assert_eq!(actual.pod_id.as_deref(), pod_id);
            assert_eq!(actual.kind, run.kind);
        };
        let active = db.active_run().unwrap().unwrap();
        assert_run_fields(&active, None);
        // pod recorded -> shows up as a live id
        db.set_pod(
            "run-id-column",
            "pod-id-column",
            Some("stale-db-host-column"),
            Some(40000),
        )
        .unwrap();
        assert_run_fields(&db.active_run().unwrap().unwrap(), Some("pod-id-column"));
        assert_eq!(
            db.live_pod_ids().unwrap(),
            vec!["pod-id-column".to_string()]
        );
        let recent = db.recent_runs(1).unwrap();
        assert_eq!(recent.len(), 1);
        assert_run_fields(&recent[0].0, Some("pod-id-column"));
        assert_eq!(recent[0].1, db.dashboard(1).unwrap()[0].updated_at);
        // per-model flags roundtrip
        assert!(
            !db.model_state("run-id-column", "model-column-b")
                .unwrap()
                .setup_done
        );
        db.set_model_setup("run-id-column", "model-column-b", true)
            .unwrap();
        db.set_model_sweep(
            "run-id-column",
            "model-column-b",
            true,
            Some("result-dir-column"),
        )
        .unwrap();
        let st = db.model_state("run-id-column", "model-column-b").unwrap();
        assert!(st.setup_done && st.sweep_done);
        assert_eq!(st.pod_run_id.as_deref(), Some("result-dir-column"));
        // terminal status -> no longer active, no longer a live pod id
        db.set_status("run-id-column", "done").unwrap();
        assert!(db.active_run().unwrap().is_none());
        assert!(db.live_pod_ids().unwrap().is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn pod_id_for_run_resolves_the_current_pod_for_the_sigterm_handler() {
        // Card 230: unlike `live_pod_ids`, `pod_id_for_run` must not filter on run status.
        let tmp = std::env::temp_dir().join(format!("pbo-pidfor-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let db = Db::open(tmp.to_str().unwrap()).unwrap();
        let run = Run {
            id: "run-pid".into(),
            kind: "sweep".into(),
            git_ref: "abc".into(),
            scenario: "decode-128".into(),
            repeats: 1,
            models: vec!["qwen2.5-0.5b".into()],
            gpu_types: "RTX 3090".into(),
            image: "img".into(),
            status: "new".into(),
            pod_id: None,
        };
        db.create_run(&run).unwrap();
        // No pod yet -> None (handler skips teardown).
        assert_eq!(db.pod_id_for_run("run-pid").unwrap(), None);
        // After provisioning the pod resolves.
        db.set_pod("run-pid", "pod-abc", Some("1.2.3.4"), Some(22))
            .unwrap();
        assert_eq!(
            db.pod_id_for_run("run-pid").unwrap(),
            Some("pod-abc".to_string())
        );
        // Still resolvable once the row is terminal (the column is not cleared).
        db.set_status("run-pid", "done").unwrap();
        assert_eq!(
            db.pod_id_for_run("run-pid").unwrap(),
            Some("pod-abc".to_string())
        );
        // Unknown run id -> None, not an error.
        assert_eq!(db.pod_id_for_run("no-such-run").unwrap(), None);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn exec_run_is_tracked_but_never_resumed_as_a_sweep() {
        let tmp = std::env::temp_dir().join(format!("pbo-exec-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let db = Db::open(tmp.to_str().unwrap()).unwrap();
        let exec = Run {
            id: "exec-123".into(),
            kind: "exec".into(),
            git_ref: "abc".into(),
            scenario: "poot-bench-runner --prefill".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA H100 80GB HBM3".into(),
            image: "img".into(),
            status: "running".into(),
            pod_id: None,
        };
        db.create_run(&exec).unwrap();
        // Tracked: pod recorded and protected from reap while non-terminal, shown on the dashboard.
        db.set_pod("exec-123", "pod-exec", Some("5.6.7.8"), Some(22))
            .unwrap();
        assert_eq!(db.live_pod_ids().unwrap(), vec!["pod-exec".to_string()]);
        let view = db.dashboard(50).unwrap();
        assert_eq!(view.len(), 1);
        assert_eq!(view[0].kind, "exec");
        // Not eligible for sweep-resume even while non-terminal (kind filter).
        assert!(db.active_run().unwrap().is_none());

        // A killed exec leaves a non-terminal row; abort_stale_execs finalizes it and frees its pod for
        // reaping, while a non-terminal sweep is left alone.
        let sweep = Run {
            id: "run-keep".into(),
            kind: "sweep".into(),
            git_ref: "x".into(),
            scenario: "decode-128".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "g".into(),
            image: "i".into(),
            status: "sweeping".into(),
            pod_id: None,
        };
        db.create_run(&sweep).unwrap();
        db.set_exec_owner("exec-123", &ExecOwner::current().unwrap())
            .unwrap();
        assert_eq!(
            db.abort_stale_execs_with(false, |_| db::OwnerLiveness::Dead)
                .unwrap()
                .aborted,
            1
        ); // only the exec row
        assert_eq!(db.live_pod_ids().unwrap().len(), 0); // exec pod no longer protected
        assert!(db.active_run().unwrap().is_some()); // the sweep is still resumable
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn keep_warm_exec_pod_is_adoptable_then_reaped_when_expired() {
        let tmp = std::env::temp_dir().join(format!("pbo-warm-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let db = Db::open(tmp.to_str().unwrap()).unwrap();
        let mk = |id: &str, image: &str| Run {
            id: id.into(),
            kind: "exec".into(),
            git_ref: "abc".into(),
            scenario: "--scatter-check".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA L40S".into(),
            image: image.into(),
            status: "running".into(),
            pod_id: None,
        };
        db.create_run_with_owner(
            &mk("exec-warm", "ghcr-img"),
            Some(&ExecOwner::current().unwrap()),
        )
        .unwrap();
        db.set_pod("exec-warm", "pod-warm", Some("1.2.3.4"), Some(40000))
            .unwrap();

        // Warm in-window: adoptable and protected from reap.
        db.set_warm("exec-warm", &iso_in_minutes(30)).unwrap();
        let w = db
            .find_warm_exec("ghcr-img")
            .unwrap()
            .expect("warm pod found");
        assert_eq!(w.pod_id, "pod-warm");
        assert_eq!((w.host.as_str(), w.port), ("1.2.3.4", 40000));
        assert_eq!(
            db.abort_stale_execs_with(false, |_| db::OwnerLiveness::Dead)
                .unwrap()
                .aborted,
            0,
            "in-window warm row is not stale"
        );
        assert_eq!(db.live_pod_ids().unwrap(), vec!["pod-warm".to_string()]);
        // A different image must not adopt this warm pod.
        assert!(db.find_warm_exec("other-img").unwrap().is_none());

        // Expired warm: abort_stale_execs finalizes it, freeing its pod for reaping.
        db.set_warm("exec-warm", &iso_in_minutes(0)).unwrap(); // warm_until == ~now, not strictly > now
        assert_eq!(
            db.abort_stale_execs_with(false, |_| db::OwnerLiveness::Dead)
                .unwrap()
                .aborted,
            1,
            "expired warm row is stale"
        );
        assert!(db.live_pod_ids().unwrap().is_empty());
        let _ = std::fs::remove_file(&tmp);
    }

    /// Card 437 fixture: a run whose pod is kept warm and still inside its window.
    fn card437_warm_row(tag: &str) -> (Db, std::path::PathBuf) {
        let tmp = std::env::temp_dir().join(format!("pbo-437-{tag}-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let db = Db::open(tmp.to_str().unwrap()).unwrap();
        let run = Run {
            id: "exec-437".into(),
            kind: "exec".into(),
            git_ref: "abc".into(),
            scenario: "packed mutation row".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA L40S".into(),
            image: "ghcr-img".into(),
            status: "running".into(),
            pod_id: None,
        };
        db.create_run_with_owner(&run, Some(&ExecOwner::current().unwrap()))
            .unwrap();
        db.set_pod("exec-437", "pod-437", Some("1.2.3.4"), Some(40000))
            .unwrap();
        db.set_warm("exec-437", &iso_in_minutes(30)).unwrap();
        (db, tmp)
    }

    /// Card 437 (1 of 2): a terminal `set_status` must not make a live keep-warm row unadoptable.
    /// `set_warm` stores warm-ness in `status`, and `find_warm_exec` filters `status='warm'`. Kept
    /// separate from half 2 so each claim can fail independently.
    #[test]
    fn card437_terminal_status_must_not_make_a_live_warm_row_unadoptable() {
        let (db, tmp) = card437_warm_row("adopt");
        assert!(
            db.find_warm_exec("ghcr-img").unwrap().is_some(),
            "precondition: a freshly warmed row is adoptable"
        );

        db.set_status("exec-437", "failed").unwrap();

        assert!(
            db.find_warm_exec("ghcr-img").unwrap().is_some(),
            "a terminal status write must not make a live warm row unadoptable - \
             find_warm_exec filters status='warm', so clobbering that column hides the pod"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// Card 437 (2 of 2): a terminal `set_status` must not strip reap protection from a live keep-warm
    /// row. `reap_protection` treats terminal statuses as reapable, so the next `exec`'s opening reap
    /// would terminate a pod still inside its warm window (reap runs before adoption).
    #[test]
    fn card437_terminal_status_must_not_strip_reap_protection_from_a_live_warm_row() {
        let (db, tmp) = card437_warm_row("reap");
        assert!(
            db.reap_protection()
                .unwrap()
                .pod_ids
                .contains(&"pod-437".to_string()),
            "precondition: a freshly warmed row's pod is reap-protected"
        );

        db.set_status("exec-437", "failed").unwrap();

        assert!(
            db.reap_protection()
                .unwrap()
                .pod_ids
                .contains(&"pod-437".to_string()),
            "a terminal status write must not strip reap protection from a live warm row - \
             the next exec's opening reap would terminate a pod still inside its warm window"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    /// Card 437: the guard covers only a live window; an expired warm row must accept a terminal status.
    #[test]
    fn card437_expired_warm_row_still_accepts_a_terminal_status() {
        let (db, tmp) = card437_warm_row("expired");
        db.set_warm("exec-437", &iso_in_minutes(0)).unwrap();
        db.set_status("exec-437", "failed").unwrap();
        assert_eq!(
            db.run_status("exec-437").unwrap().as_deref(),
            Some("failed"),
            "an EXPIRED warm row must still accept a terminal status"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn concurrent_reap_process_protects_live_exec_before_pod_id_callback() {
        let tmp = std::env::temp_dir().join(format!(
            "pbo-two-process-reap-{}-{}.db",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let _ = std::fs::remove_file(&tmp);
        let db = Db::open(tmp.to_str().unwrap()).unwrap();
        let exec = Run {
            id: "exec-1788530213-892720615-3077622-0".into(),
            kind: "exec".into(),
            git_ref: "abc".into(),
            scenario: "provisioning regression".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA L40S".into(),
            image: "img".into(),
            status: "new".into(),
            pod_id: None,
        };
        let owner = ExecOwner::current().unwrap();
        db.create_run_with_owner(&exec, Some(&owner)).unwrap();
        assert!(
            db.live_pod_ids().unwrap().is_empty(),
            "reproduce the interval before on_create records the pod id"
        );

        let out = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("reap_child_process_helper")
            .arg("--nocapture")
            .env("POOT_ORCH_REAP_CHILD_DB", tmp.to_str().unwrap())
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "child reap failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout = String::from_utf8_lossy(&out.stdout);
        let child_pid = stdout
            .lines()
            .find_map(|line| line.split_once("child pid: ").map(|(_, pid)| pid.trim()))
            .and_then(|pid| pid.parse::<u32>().ok())
            .expect("child helper must print its OS pid");
        assert_ne!(
            child_pid,
            std::process::id(),
            "the regression must run stale-exec finalization from a separate process"
        );
        assert_eq!(
            db.run_status("exec-1788530213-892720615-3077622-0")
                .unwrap()
                .as_deref(),
            Some("new"),
            "normal reap from another process must not abort a live exec owner"
        );
        assert!(
            stdout.contains("child targets: []"),
            "the separate process must protect the cloud-visible pod before its DB id callback:\n{stdout}"
        );
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn reap_child_process_helper() {
        let Some(path) = std::env::var("POOT_ORCH_REAP_CHILD_DB").ok() else {
            return;
        };
        let db = Db::open(&path).unwrap();
        let report = db.abort_stale_execs(false).unwrap();
        let prefix = db.pod_owner_prefix().unwrap();
        let cloud = vec![pod(
            "4dmn0aykzpftwj",
            &prefix.pod_name("exec-1788530213-892720615-3077622-0", 1),
            None,
            None,
        )];
        let protection = db.reap_protection().unwrap();
        let targets = reap_targets(&cloud, &prefix, &protection, false);
        println!("child pid: {}", std::process::id());
        println!(
            "child report: aborted={} kept_live={} kept_unknown={}",
            report.aborted, report.kept_live, report.kept_unknown
        );
        println!("child targets: {targets:?}");
    }

    #[test]
    fn dashboard_view_serializes_with_per_model_progress() {
        let tmp = std::env::temp_dir().join(format!("pbo-dash-{}.db", std::process::id()));
        let _ = std::fs::remove_file(&tmp);
        let db = Db::open(tmp.to_str().unwrap()).unwrap();
        let run = Run {
            id: "run-9".into(),
            kind: "sweep".into(),
            git_ref: "deadbeef".into(),
            scenario: "decode-128".into(),
            repeats: 1,
            models: vec!["qwen2.5-0.5b".into()],
            gpu_types: "NVIDIA GeForce RTX 3090".into(),
            image: "img:latest".into(),
            status: "sweeping".into(),
            pod_id: None,
        };
        db.create_run(&run).unwrap();
        db.set_pod("run-9", "pod-9", Some("9.9.9.9"), Some(8787))
            .unwrap();
        db.set_model_setup("run-9", "qwen2.5-0.5b", true).unwrap();

        let view = db.dashboard(50).unwrap();
        assert_eq!(view.len(), 1);
        let r = &view[0];
        assert_eq!(r.id, "run-9");
        assert_eq!(r.status, "sweeping");
        assert_eq!(r.pod_id.as_deref(), Some("pod-9"));
        assert_eq!(r.ssh_host.as_deref(), Some("9.9.9.9"));
        assert_eq!(r.ssh_port, Some(8787));
        assert_eq!(r.models.len(), 1);
        assert!(r.models[0].setup_done && !r.models[0].sweep_done);

        // /api/state serializes this view.
        let json = serde_json::to_string(&view).unwrap();
        assert!(json.contains("\"status\":\"sweeping\""));
        assert!(json.contains("\"ssh_host\":\"9.9.9.9\""));
        assert!(json.contains("\"ssh_port\":8787"));
        assert!(json.contains("\"setup_done\":true"));
        assert!(json.contains("\"model\":\"qwen2.5-0.5b\""));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn safe_log_name_blocks_traversal() {
        assert!(safe_log_name("run-123.log"));
        assert!(safe_log_name("image-slim.log"));
        assert!(!safe_log_name(""));
        assert!(!safe_log_name("../Cargo.toml"));
        assert!(!safe_log_name("a/b.log"));
        assert!(!safe_log_name(".."));
        assert!(!safe_log_name("x\\y")); // windows-style separator
    }

    #[test]
    fn classify_log_kind_splits_runs_and_images() {
        assert_eq!(classify_log_kind("image-slim.log"), LogKind::Image);
        assert_eq!(classify_log_kind("image-latest.log"), LogKind::Image);
        assert_eq!(classify_log_kind("run-1234.log"), LogKind::Run);
        assert_eq!(classify_log_kind("run-demo.log"), LogKind::Run);
    }

    #[test]
    fn classify_log_state_reads_markers_and_phrases() {
        // Mid-flight sweep: no terminal marker.
        let running = "[12:00:00] === sweep: qwen3-0.6b ===\n  qwen3-0.6b: [progress] tok/s=44.1\n";
        assert_eq!(classify_log_state(running), LogState::Running);
        // Clean exit, code embedded in a phrase.
        let ok = "  qwen3-0.6b sweep finished (SWEEP_EXIT=0)\n";
        assert_eq!(classify_log_state(ok), LogState::Done);
        // Failed exit.
        let bad = "  qwen3-0.6b: oom\nSWEEP_EXIT=137\n";
        assert_eq!(classify_log_state(bad), LogState::Failed);
        // Build completion phrase.
        let built =
            "[1/2] STEP 8/8 : CMD ...\n[12:40:00] build complete: ghcr.io/x/poot-bench:slim\n";
        assert_eq!(classify_log_state(built), LogState::Done);
        // An exit marker overrides an earlier phrase.
        let build_fail = "[12:00:00] build complete\nBUILD_EXIT=2\n";
        assert_eq!(classify_log_state(build_fail), LogState::Failed);
        // Failure phrase.
        assert_eq!(
            classify_log_state("run failed: pod gone\n"),
            LogState::Failed
        );
        assert_eq!(
            classify_log_state("image smoke failed (rc=1)\n"),
            LogState::Failed
        );
        // An empty log defaults to running.
        assert_eq!(classify_log_state(""), LogState::Running);
    }

    #[test]
    fn now_iso_is_well_formed() {
        let s = now_iso();
        assert_eq!(s.len(), 20, "{s}");
        assert!(s.ends_with('Z') && &s[4..5] == "-" && &s[10..11] == "T");
    }

    #[test]
    fn iso_from_secs_known_values() {
        // The civil-from-days conversion and zero-padding must reproduce known timestamps exactly.
        assert_eq!(iso_from_secs(0), "1970-01-01T00:00:00Z");
        assert_eq!(iso_from_secs(86_400), "1970-01-02T00:00:00Z");
        // 1_000_000_000 = 2001-09-09T01:46:40Z.
        assert_eq!(iso_from_secs(1_000_000_000), "2001-09-09T01:46:40Z");
        // Leap day: 2024-02-29T12:00:00Z = 1709208000.
        assert_eq!(iso_from_secs(1_709_208_000), "2024-02-29T12:00:00Z");
    }

    #[test]
    fn iso_from_secs_string_order_matches_time_order() {
        // The `--keep-warm` expiry compares `warm_until > now_iso()` as plain strings, so the ISO strings
        // must sort chronologically, including across month boundaries.
        let times = [
            0i64,
            59,
            60,
            86_399,
            86_400,
            1_000_000_000,
            1_709_207_999,
            1_709_208_000,
        ];
        for w in times.windows(2) {
            let (a, b) = (iso_from_secs(w[0]), iso_from_secs(w[1]));
            assert!(a < b, "expected {a} < {b} for {} < {}", w[0], w[1]);
        }
        // Month boundary (Jan 31 vs Feb 1, 2023).
        assert!(iso_from_secs(1_675_209_599) < iso_from_secs(1_675_209_600));
    }

    #[test]
    fn cuda_floor_to_allowed_filters_orders_and_guards() {
        // The default cu1281 image needs CUDA >= 12.8.
        assert_eq!(
            cuda_floor_to_allowed("12.8"),
            vec!["13.0", "12.9", "12.8"],
            "12.8 floor -> the >=12.8 hosts, newest-first"
        );
        // Top version -> itself; a floor above every valid version -> empty.
        assert_eq!(cuda_floor_to_allowed("13.0"), vec!["13.0"]);
        assert!(cuda_floor_to_allowed("14.0").is_empty());
        // A version between valid entries (12.10 > 12.9) still filters.
        assert_eq!(cuda_floor_to_allowed("12.10"), vec!["13.0"]);
        // Empty or unparseable floor -> no filter (empty list), not a panic or a full list.
        assert!(cuda_floor_to_allowed("").is_empty());
        assert!(cuda_floor_to_allowed("garbage").is_empty());
        assert!(
            cuda_floor_to_allowed("12").is_empty(),
            "no minor -> unparseable -> no filter"
        );
        // The oldest floor includes everything.
        assert_eq!(cuda_floor_to_allowed("11.8").len(), 12);
    }
}
