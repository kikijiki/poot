//! The typed serve configuration (POOT-590): every setting is a flag with a strict parser, and no
//! environment variable selects a mode or a limit.

use std::net::{IpAddr, Ipv4Addr};
use std::path::Path;

use clap::Parser;
use poot_executor::DetailedTiming;
use poot_llm::ChatTemplateOverride;
use poot_llm::driver::{BackendChoice, DraftPolicy};

use super::*;
use crate::admin::{LoraAdminDenial, LoraAdminPolicyMode};
use crate::config::{ReadTimeout, ServeConfig};
use crate::http::AuthorizationHeader;

fn parse(flags: &[&str]) -> Result<ServeConfig, String> {
    ServeConfig::try_parse_from(["poot-serve"].into_iter().chain(flags.iter().copied()))
        .map_err(|error| error.to_string())
}

fn config(flags: &[&str]) -> ServeConfig {
    parse(flags).unwrap_or_else(|error| panic!("{flags:?} must parse: {error}"))
}

/// A scratch file holding `contents`, removed on drop.
struct TempFile(std::path::PathBuf);

impl TempFile {
    fn new(label: &str, contents: &str) -> Self {
        let path = std::env::temp_dir().join(format!(
            "poot-serve-config-{label}-{}-{}",
            std::process::id(),
            unique_fixture_suffix()
        ));
        std::fs::write(&path, contents).expect("write the scratch file");
        Self(path)
    }

    fn path(&self) -> &str {
        self.0.to_str().expect("a UTF-8 temp path")
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// With no flag every setting has its documented default.
#[test]
fn defaults_are_the_documented_ones() {
    let config = config(&[]);
    assert_eq!(config.model, None);
    assert_eq!(config.addr, "127.0.0.1:8080");
    assert_eq!(config.backend, BackendChoice::Auto);
    assert_eq!(config.draft_lookup, None);
    assert_eq!(config.draft_model, None);
    assert_eq!(config.rate_limit_rpm, 0);
    assert_eq!(config.max_connections, 4096);
    assert_eq!(
        config.read_timeout_secs,
        ReadTimeout(Some(Duration::from_secs(30)))
    );
    assert_eq!(config.write_timeout_secs, Duration::from_secs(30));
    assert_eq!(
        config.drain_timeout_secs.duration(),
        Duration::from_secs(30)
    );
    assert!(config.lora_admin_key_file.is_none());
    assert_eq!(config.telemetry(), TelemetryConfig::CountersOnly);
    assert_eq!(config.load_options().chat_template, None);
}

/// Every flag takes effect; `0` keeps its documented meaning (no limit, no read bound) where one exists.
#[test]
fn flags_set_their_fields() {
    let config = config(&[
        "--backend",
        "vulkan",
        "--rate-limit-rpm",
        "60",
        "--max-connections",
        "0",
        "--read-timeout-secs",
        "0",
        "--write-timeout-secs",
        " 7 ",
        "--drain-timeout-secs",
        "86400",
        "--draft-lookup",
        "3:4",
        "model-dir",
        "0.0.0.0:9000",
    ]);
    assert_eq!(config.backend, BackendChoice::Vulkan);
    assert_eq!(config.rate_limit_rpm, 60);
    assert_eq!(config.max_connections, 0);
    assert_eq!(config.read_timeout_secs, ReadTimeout(None));
    assert_eq!(config.write_timeout_secs, Duration::from_secs(7));
    assert_eq!(
        config.drain_timeout_secs.duration(),
        Duration::from_secs(86_400)
    );
    let lookup = config.draft_lookup.expect("prompt lookup");
    assert_eq!((lookup.ngram.get(), lookup.k.get()), (3, 4));
    assert_eq!(config.model.as_deref(), Some("model-dir"));
    assert_eq!(config.addr, "0.0.0.0:9000");
}

/// The chat template flags become the load options' override: text, or a file, or none; both at once is
/// refused by the parser.
#[test]
fn the_chat_template_flags_become_the_load_override() {
    assert_eq!(
        config(&["--chat-template", "{{ x }}"])
            .load_options()
            .chat_template,
        Some(ChatTemplateOverride::Jinja("{{ x }}".into()))
    );
    assert_eq!(
        config(&["--chat-template-file", "t.jinja"])
            .load_options()
            .chat_template,
        Some(ChatTemplateOverride::File("t.jinja".into()))
    );
    assert!(
        parse(&[
            "--chat-template",
            "{{ x }}",
            "--chat-template-file",
            "t.jinja"
        ])
        .is_err()
    );
}

/// SC-002. An invalid value for any field is a startup error that names the flag and the value, never a
/// silent default. Mutation: make one parser fall back to its default on a bad value (for example
/// `parse_count` returning `Ok(0)` on an error); that row stops being an error and this goes red.
#[test]
fn an_invalid_value_for_any_field_is_a_startup_error() {
    let table: &[(&[&str], &str)] = &[
        (&["--backend", "cuda"], "unknown backend"),
        (&["--backend", "Rocm"], "unknown backend"),
        (&["--rate-limit-rpm", "many"], "must be a whole number"),
        (&["--rate-limit-rpm=-1"], "non-negative"),
        (&["--rate-limit-rpm", ""], "must be a whole number"),
        (&["--max-connections", "garbage"], "must be a whole number"),
        (&["--max-connections=-5"], "non-negative"),
        (
            &["--read-timeout-secs", "garbage"],
            "must be a whole number",
        ),
        (&["--read-timeout-secs=-1"], "non-negative"),
        (&["--write-timeout-secs", "0"], "greater than zero"),
        (&["--write-timeout-secs=-1"], "non-negative"),
        (
            &["--write-timeout-secs", "garbage"],
            "must be a whole number",
        ),
        (
            &["--write-timeout-secs", "18446744073709551616"],
            "too large",
        ),
        (&["--drain-timeout-secs", "0"], "greater than zero"),
        (&["--drain-timeout-secs", "86401"], "at most 86400"),
        (
            &["--drain-timeout-secs", "invalid"],
            "must be a whole number",
        ),
        (&["--draft-lookup", "3"], "expected NGRAM:K"),
        (&["--draft-lookup", "3:0"], "not a positive integer"),
        (&["--draft-lookup", "0:4"], "not a positive integer"),
        (&["--draft-lookup", "x:4"], "not a positive integer"),
        (&["--draft-lookup", "3:"], "not a positive integer"),
        (&["--draft-model", "no-colon"], "expected PATH:K"),
        (&["--draft-model", "m:0"], "not a positive integer"),
        (&["--draft-model", ":2"], "model path is empty"),
        (&["--draft-mtp", "0"], "not a positive integer"),
        (&["--draft-mtp", "x"], "not a positive integer"),
        (&["--draft-mtp"], "a value is required"),
        (
            &["--draft-mtp", "2", "--draft-lookup", "3:4"],
            "cannot be used",
        ),
        (
            &["--draft-mtp", "2", "--draft-model", "m:2"],
            "cannot be used",
        ),
        (
            &["--draft-lookup", "3:4", "--draft-model", "m:2"],
            "cannot be used",
        ),
        (
            &["--device-timing", "--timing-max-steps", "0"],
            "greater than zero",
        ),
        (
            &["--device-timing", "--timing-max-steps", "16385"],
            "at most 16384",
        ),
        (
            &["--device-timing", "--timing-max-dispatch-records", "0"],
            "greater than zero",
        ),
        (
            &["--device-timing", "--timing-max-dispatch-records", "262145"],
            "at most 262144",
        ),
        (
            &["--device-timing", "--timing-max-in-flight-queries", "0"],
            "greater than zero",
        ),
        (
            &["--device-timing", "--timing-max-in-flight-queries", "4097"],
            "at most 4096",
        ),
        (
            &["--device-timing", "--timing-sample-every", "0"],
            "greater than zero",
        ),
        (
            &["--device-timing", "--timing-sample-every", "garbage"],
            "must be a whole number",
        ),
        // A timing bound without the mode it bounds is a mistake, not an ignored flag.
        (&["--timing-max-steps", "8"], "--device-timing"),
        (&["--timing-sample-every", "2"], "--device-timing"),
        (
            &["--lora-admin-key-file", "/nonexistent/poot-admin-key"],
            "cannot read the key file",
        ),
        (&["--no-such-flag"], "unexpected argument"),
    ];
    for (flags, expected) in table {
        let error = parse(flags).expect_err(&format!("{flags:?} must be refused"));
        assert!(
            error.contains(expected),
            "{flags:?}: expected {expected:?} in {error}"
        );
    }

    let empty = TempFile::new("empty-key", "\n");
    let spaced = TempFile::new("spaced-key", "two words\n");
    for key_file in [&empty, &spaced] {
        let error = parse(&["--lora-admin-key-file", key_file.path()])
            .expect_err("an invalid key file is a startup error");
        assert!(error.contains("RFC 6750 b64token"), "{error}");
    }
}

/// SC-003. The legacy `--profile` is refused on every backend: serve has one timing configuration, the
/// typed one below, and no backend accepts the old boolean. Mutation: add a `--profile` flag to the
/// parser; every row goes red.
#[test]
fn legacy_profile_is_refused_on_every_backend() {
    for backend in ["wgpu", "rocm", "ptx", "vulkan", "auto"] {
        let error =
            parse(&["--backend", backend, "--profile"]).expect_err("--profile must be refused");
        assert!(
            error.contains("unexpected argument '--profile'"),
            "{backend}: {error}"
        );
    }
}

/// SC-003. The default configuration is counters only: no device timing, no detail, no timestamp
/// query. Detailed timing is an explicit flag and carries finite limits. Mutation: make
/// `telemetry()` return `Detailed` without the flag; the first assertion goes red. Drop a limit (set
/// `max_retained_steps` to `usize::MAX`); the bound assertions go red.
#[test]
fn detailed_timing_is_explicit_and_bounded() {
    assert_eq!(config(&[]).telemetry(), TelemetryConfig::CountersOnly);
    assert_eq!(
        config(&["--backend", "wgpu"]).telemetry(),
        TelemetryConfig::CountersOnly
    );

    let TelemetryConfig::Detailed(default) = config(&["--device-timing"]).telemetry() else {
        panic!("--device-timing selects detailed timing");
    };
    assert_eq!(default, DetailedTiming::default_diagnostic());
    assert!(default.max_retained_steps > 0 && default.max_retained_steps <= 1 << 14);
    assert!(default.max_retained_dispatch_records > 0);
    assert!(default.max_in_flight_queries > 0);

    let TelemetryConfig::Detailed(custom) = config(&[
        "--device-timing",
        "--timing-max-steps",
        "8",
        "--timing-max-dispatch-records",
        "32",
        "--timing-max-in-flight-queries",
        "2",
        "--timing-sample-every",
        "5",
    ])
    .telemetry() else {
        panic!("--device-timing selects detailed timing");
    };
    assert_eq!(custom.max_retained_steps, 8);
    assert_eq!(custom.max_retained_dispatch_records, 32);
    assert_eq!(custom.max_in_flight_queries, 2);
    assert_eq!(custom.sample_every, 5);
}

/// SC-005. `--lora-admin-key-file` reads the key from the named file (a trailing line break is
/// trimmed), the policy then requires exactly that key on both listener classes, and a wrong or missing
/// key is refused.
#[test]
fn the_admin_key_file_sets_the_bearer_key_and_rejects_a_wrong_one() {
    let key_file = TempFile::new("admin-key", "card590-admin-secret\n");
    let config = config(&["--lora-admin-key-file", key_file.path()]);
    for listener in [Ipv4Addr::LOCALHOST, Ipv4Addr::UNSPECIFIED] {
        let policy = config
            .lora_admin_policy(IpAddr::V4(listener))
            .expect("the policy builds");
        assert_eq!(policy.mode(), LoraAdminPolicyMode::BearerRequired);
        let bearer = |value: &str| AuthorizationHeader::Single(value.to_string());
        assert_eq!(
            policy.authorize(&bearer("Bearer card590-admin-secret")),
            Ok(())
        );
        assert_eq!(
            policy.authorize(&bearer("Bearer card590-admin-wrong")),
            Err(LoraAdminDenial::Authentication)
        );
        assert_eq!(
            policy.authorize(&AuthorizationHeader::Missing),
            Err(LoraAdminDenial::Authentication)
        );
    }
}

/// Run in a child so environment variables can be set without touching this test process. It parses the
/// flags it is given and prints what the production consumers hold: the transport values the accept loop
/// runs on (`Transport::of`, which `run` uses), the engine arguments (`engine_args`, which `run` uses),
/// the telemetry the executor is opened with and how the admin policy treats a bearer key.
#[test]
fn config_under_environment_child() {
    use crate::startup::{Transport, engine_args};

    let Ok(flags) = std::env::var("POOT_CONFIG_CHILD_FLAGS") else {
        return; // run by the parent, not as a test
    };
    let flags: Vec<&str> = flags.split_whitespace().collect();
    let config = config(&flags);
    let transport = Transport::of(&config);
    let args = engine_args(&config, DraftPolicy::Off);
    let policy = config
        .lora_admin_policy(IpAddr::V4(Ipv4Addr::UNSPECIFIED))
        .expect("a policy");
    let bearer = AuthorizationHeader::Single("Bearer env-admin-key".to_string());
    println!(
        "CONFIG {:?} | {:?} | {:?} | rpm={} max_connections={} read={:?} write={:?} drain={:?} | {:?} | \
         mode={:?} env_key={:?} missing={:?}",
        config,
        config.telemetry(),
        config.load_options().chat_template,
        transport.limiter.rpm,
        transport.max_connections,
        transport.read_timeout,
        transport.write_timeout,
        transport.drain_timeout,
        args,
        policy.mode(),
        policy.authorize(&bearer),
        policy.authorize(&AuthorizationHeader::Missing),
    );
}

/// What the child prints for `flags` with `env` set.
fn config_under(flags: &str, env: &[(&str, &str)]) -> String {
    let mut command = std::process::Command::new(std::env::current_exe().unwrap());
    command
        .args([
            "--exact",
            "tests::config::config_under_environment_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("POOT_CONFIG_CHILD_FLAGS", flags);
    for (name, value) in env {
        command.env(name, value);
    }
    let output = command.output().expect("run the config child");
    assert!(
        output.status.success(),
        "child failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    // libtest prints its own `test ... ` prefix on the same line, so take what follows the marker.
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let start = stdout
        .find("CONFIG ")
        .unwrap_or_else(|| panic!("no CONFIG line in {stdout:?}"));
    stdout[start..].lines().next().unwrap().to_string()
}

/// SC-001. Every variable that used to choose a mode or a limit now has no effect: what the production
/// consumers hold (configuration, rate limiter, connection cap, read/write/drain bounds, engine arguments,
/// admin policy behaviour) is byte-identical with it set, and the typed flag that replaced it changes it.
/// Mutation: re-add an environment read in a consumer (for example `POOT_RATE_LIMIT_RPM` in
/// `RateLimiter::new`); that variable changes `rpm=` in the child's output and its row goes red.
#[test]
fn former_mode_variables_have_no_effect_and_the_typed_flags_do() {
    let key_file = TempFile::new("env-key", "env-admin-key\n");
    // (the former variable, a value it used to honor, the flag that replaced it, the flag's value)
    let former: &[(&str, &str, &str, &str)] = &[
        ("POOT_SLOTS", "8", "", ""),
        ("POOT_CAP", "512", "", ""),
        ("POOT_KV_QUANT", "q8_0", "", ""),
        // The speculative-decoding variables are deleted too; the stale-name gate keeps their names out of
        // code, and `--draft-lookup` / `--draft-model` are the typed flags that replaced them.
        ("POOT_TP_GPUS", "2", "", ""),
        ("POOT_RATE_LIMIT_RPM", "60", "--rate-limit-rpm", "60"),
        ("POOT_MAX_CONNECTIONS", "7", "--max-connections", "7"),
        ("POOT_READ_TIMEOUT_SECS", "5", "--read-timeout-secs", "5"),
        ("POOT_WRITE_TIMEOUT_SECS", "5", "--write-timeout-secs", "5"),
        ("POOT_DRAIN_TIMEOUT_SECS", "5", "--drain-timeout-secs", "5"),
        (
            "POOT_LORA_ADMIN_KEY",
            "env-admin-key",
            "--lora-admin-key-file",
            key_file.path(),
        ),
        // Even the garbage values the old parsers swallowed or refused are inert now.
        ("POOT_WRITE_TIMEOUT_SECS", "0", "", ""),
        ("POOT_DRAIN_TIMEOUT_SECS", "garbage", "", ""),
        (
            "POOT_CARD327_TEST_ACCEPT_BARRIER_ADDR",
            "127.0.0.1:1",
            "",
            "",
        ),
    ];
    let baseline = config_under("", &[]);
    for &(name, value, flag, flag_value) in former {
        assert_eq!(
            config_under("", &[(name, value)]),
            baseline,
            "{name}={value} changed the configuration"
        );
        if !flag.is_empty() {
            let with_flag = config_under(&format!("{flag} {flag_value}"), &[]);
            assert_ne!(
                with_flag, baseline,
                "{flag} did not change the configuration"
            );
        }
    }
}

/// The key file path of [`the_admin_key_file_sets_the_bearer_key_and_rejects_a_wrong_one`] is read
/// when the flag parses, so a key file that is gone afterwards cannot change a parsed configuration.
#[test]
fn the_key_is_read_once_at_parse_time() {
    let key_file = TempFile::new("read-once", "read-once-key\n");
    let parsed = config(&["--lora-admin-key-file", key_file.path()]);
    assert!(Path::new(key_file.path()).exists());
    drop(key_file);
    let policy = parsed
        .lora_admin_policy(IpAddr::V4(Ipv4Addr::LOCALHOST))
        .unwrap();
    assert_eq!(policy.mode(), LoraAdminPolicyMode::BearerRequired);
}

/// The draft flags are exclusive at the parser, and the startup policy also refuses both: the state is an
/// error, not an unreachable branch. Mutation: pick the lookup policy when both are set; the row goes red.
#[test]
fn both_draft_flags_are_refused_by_the_parser_and_by_the_policy() {
    use crate::config::DraftModel;

    assert!(parse(&["--draft-lookup", "3:4", "--draft-model", "m:2"]).is_err());
    let mut both = config(&["--draft-lookup", "3:4"]);
    both.draft_model = Some(DraftModel {
        path: "m".to_string(),
        k: std::num::NonZeroUsize::new(2).unwrap(),
    });
    let error = crate::startup::draft_policy(&both).expect_err("both draft flags");
    assert!(error.to_string().contains("exclusive"), "{error}");
}

/// `--draft-mtp K` names the model's own MTP layer with a required K, and the policy it builds
/// carries the verify window as its hidden-state carry window. Without it there is no MTP policy.
/// Mutation: build `DraftPolicy::Off` for the flag; the first assertion is red.
#[test]
fn the_draft_mtp_flag_builds_the_mtp_policy_with_its_carry_window() {
    let policy = crate::startup::draft_policy(&config(&["--draft-mtp", "2"])).unwrap();
    let DraftPolicy::Mtp { k } = &policy else {
        panic!("the flag names the MTP policy: {policy:?}");
    };
    assert_eq!(k.get(), 2);
    assert_eq!(
        policy.carry_window().map(std::num::NonZeroUsize::get),
        Some(3)
    );
    assert!(matches!(
        crate::startup::draft_policy(&config(&[])).unwrap(),
        DraftPolicy::Off
    ));
}

/// The placement flag defaults to the one default device and parses a device group strictly: the kind,
/// then at least two distinct GPU ordinals, kept in the order given because the order is the rank order.
/// Mutation: accept a repeated ordinal or a single GPU for a group; those rows go red.
#[test]
fn the_placement_flag_parses_a_device_group_strictly() {
    use crate::config::PlacementSpec;

    assert_eq!(config(&[]).placement, PlacementSpec::Single);
    assert_eq!(
        config(&["--placement", "single"]).placement,
        PlacementSpec::Single
    );
    assert_eq!(
        config(&["--placement", "pipeline:0,1"]).placement,
        PlacementSpec::Pipeline {
            ordinals: vec![0, 1]
        }
    );
    assert_eq!(
        config(&["--placement", "tensor:3,1,2"]).placement,
        PlacementSpec::Tensor {
            ordinals: vec![3, 1, 2]
        }
    );
    for bad in [
        "pipeline",
        "pipeline:",
        "pipeline:0",
        "pipeline:0,0",
        "pipeline:0,",
        "pipeline:0,x",
        "pipeline:0,-1",
        "ring:0,1",
        "tensor:",
        "single:0",
        "",
    ] {
        assert!(
            parse(&["--placement", bad]).is_err(),
            "{bad:?} must be refused"
        );
    }
}

/// A device group records no device timing: asking for both is a startup error before anything loads.
#[test]
fn a_device_group_refuses_device_timing() {
    let config = config(&["--placement", "tensor:0,1", "--device-timing"]);
    let error = crate::startup::engine_args(&config, DraftPolicy::Off)
        .expect_err("a group with device timing");
    assert!(error.to_string().contains("--device-timing"), "{error}");
    let ok = self::config(&["--placement", "tensor:0,1"]);
    assert!(crate::startup::engine_args(&ok, DraftPolicy::Off).is_ok());
}
