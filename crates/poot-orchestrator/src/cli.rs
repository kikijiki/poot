//! Command-line schema.

use std::time::Duration;

use clap::{Args, Parser, Subcommand};

/// The most `--keep-warm` minutes `exec` accepts: a kept pod bills until its window ends.
pub(crate) const MAX_KEEP_WARM_MINUTES: u64 = 240;

/// The longest `--max-cmd` `exec` accepts.
pub(crate) const MAX_CMD_CEILING: Duration = Duration::from_secs(12 * 3600);

/// A CLI input error surfaced by clap's value parsers (typed, not `anyhow`).
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub(crate) enum CliError {
    #[error(
        "invalid data-center id {0:?}: must be non-empty and use only A-Z, 0-9, and '-' \
         (e.g. EU-RO-1)"
    )]
    DataCenter(String),
    #[error("invalid duration {0:?}: expected a whole number plus s, m or h (e.g. 90s, 30m, 2h)")]
    Duration(String),
    #[error("duration {0:?} must be longer than zero and at most 12h")]
    DurationRange(String),
}

/// Parse `--max-cmd`: a whole number of seconds, minutes or hours, above zero and at most
/// [`MAX_CMD_CEILING`].
pub(crate) fn parse_max_cmd(raw: &str) -> Result<Duration, CliError> {
    let raw = raw.trim();
    let split = raw
        .find(|c: char| !c.is_ascii_digit())
        .ok_or_else(|| CliError::Duration(raw.to_string()))?;
    let (digits, unit) = raw.split_at(split);
    let n: u64 = digits
        .parse()
        .map_err(|_| CliError::Duration(raw.to_string()))?;
    let secs = match unit {
        "s" => n,
        "m" => n.saturating_mul(60),
        "h" => n.saturating_mul(3600),
        _ => return Err(CliError::Duration(raw.to_string())),
    };
    let d = Duration::from_secs(secs);
    if d.is_zero() || d > MAX_CMD_CEILING {
        return Err(CliError::DurationRange(raw.to_string()));
    }
    Ok(d)
}

/// Parse one `--data-center` id: non-empty, uppercase letters, digits, and dashes only.
/// Accepts one value per occurrence; clap's `value_delimiter` also splits comma lists first.
pub(crate) fn parse_data_center_id(raw: &str) -> Result<String, CliError> {
    let id = raw.trim();
    if id.is_empty()
        || !id
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '-')
    {
        return Err(CliError::DataCenter(raw.to_string()));
    }
    Ok(id.to_string())
}

#[derive(Parser)]
#[command(
    name = "poot-orchestrator",
    version,
    about = "Resumable poot benchmark sweeps on RunPod"
)]
pub(crate) struct Cli {
    /// SQLite run-state DB (enables resume + reap). Default: $XDG_STATE_HOME/poot-orchestrator/state.db
    /// (else ~/.local/state/poot-orchestrator/state.db), shared by every git worktree.
    #[arg(long, global = true, env = "PBO_STATE_DB")]
    pub(crate) state_db: Option<String>,
    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand)]
// The enum is built once from argv, so the size gap is harmless.
#[allow(clippy::large_enum_variant)]
pub(crate) enum Command {
    /// Run (or resume) a sweep: build, provision, setup, sweep, gather, teardown.
    Run(RunArgs),
    /// Terminate orphaned pods (our name prefix) not tied to an in-progress run. Safe to run anytime.
    Reap(ReapArgs),
    /// Show recent runs and their phase.
    Status,
    /// Serve a read-only web dashboard (runs + phases + per-model progress) over the state DB.
    Serve(ServeArgs),
    /// Build (and optionally push) the pre-baked bench container image via podman, streaming progress.
    Image(ImageArgs),
    /// Tail a run/build log from the shared state dir's logs/ (no name lists them).
    Logs(LogsArgs),
    /// Provision a pod, scp a binary, run a command, tear down. For poot tests on real NVIDIA hardware,
    /// not the bench sweep.
    Exec(ExecArgs),
}

#[derive(Args)]
pub(crate) struct ExecArgs {
    /// the command to run on the pod (its output is printed when it exits).
    #[arg(long)]
    pub(crate) cmd: String,
    /// longest the setup and `--cmd` may run on the pod before ssh is killed and the exec fails with a
    /// timeout (`90s`, `30m`, `2h`; at most 12h). The pod is torn down afterwards.
    #[arg(long, default_value = "2h", value_parser = parse_max_cmd)]
    pub(crate) max_cmd: Duration,
    /// minutes after which the pod removes itself, whether or not this machine is still running.
    /// Default: `--max-cmd` plus `--keep-warm` plus a 30 minute margin for upload and setup.
    #[arg(long, value_name = "MINUTES", value_parser = clap::value_parser!(u64).range(1..))]
    pub(crate) pod_ttl: Option<u64>,
    /// a local binary to scp to /root/ before running (patchelf'd to the stock glibc loader).
    #[arg(long)]
    pub(crate) bin: Option<String>,
    /// extra files to scp before `--cmd`, each `local:remote` (repeatable). Copied verbatim, no patchelf.
    #[arg(long = "upload", value_name = "LOCAL:REMOTE")]
    pub(crate) uploads: Vec<String>,
    /// a setup command run on the pod before `--cmd` (e.g. an `hf download`).
    #[arg(long)]
    pub(crate) setup: Option<String>,
    /// GPU types to try in order.
    #[arg(
        long,
        default_value = "NVIDIA L40S,NVIDIA A100 80GB PCIe,NVIDIA RTX A5000,NVIDIA GeForce RTX 4090"
    )]
    pub(crate) gpu_types: String,
    /// GPUs per pod, for multi-GPU work (e.g. tensor-parallel: `--gpu-count 2`).
    #[arg(long, default_value_t = 1)]
    pub(crate) gpu_count: u32,
    /// RunPod data-center ids to pin the pod to (`dataCenterIds`), e.g. EU-RO-1. Repeatable or
    /// comma-separated. Empty: no pin (RunPod picks any data center with stock).
    #[arg(
        long = "data-center",
        value_name = "ID",
        value_delimiter = ',',
        value_parser = parse_data_center_id
    )]
    pub(crate) data_center: Vec<String>,
    /// pod image, resolved to a digest before the pod is created; the pod runs that digest. The default is
    /// the poot bench image: ubuntu24.04 (glibc 2.39) so a nix-built poot binary runs without ABI
    /// patching, and it carries llc and libvulkan1, which the stock runpod/pytorch image does not.
    #[arg(long, default_value = "ghcr.io/kikijiki/poot-bench:latest")]
    pub(crate) image: String,
    #[arg(long, default_value = "SECURE")]
    pub(crate) cloud: String,
    /// Minimum host CUDA version; expands to RunPod `allowedCudaVersions` (all valid versions >= this).
    /// The default `cu1281` image needs >= 12.8; older-driver hosts fail the nvidia-container-cli
    /// `cuda>=12.8` hook. Empty disables the filter.
    #[arg(long, default_value = "12.8")]
    pub(crate) min_cuda_version: Option<String>,
    #[arg(long, default_value_t = 30)]
    pub(crate) container_disk_gb: u32,
    /// Create-time container env `KEY=VAL` (repeatable), merged into the RunPod create `env` with
    /// `PUBLIC_KEY`. A `NVIDIA_DRIVER_CAPABILITIES` key overrides the default, e.g.
    /// `--env NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics`
    #[arg(long = "env", value_name = "KEY=VAL")]
    pub(crate) env: Vec<String>,
    #[arg(long, default_value = "~/.ssh/id_ed25519")]
    pub(crate) ssh_key: String,
    #[arg(long, default_value_t = 3)]
    pub(crate) provision_attempts: u32,
    #[arg(long, default_value_t = 600)]
    pub(crate) provision_timeout_s: u64,
    #[arg(long, default_value_t = 600)]
    pub(crate) ssh_timeout_s: u64,
    /// keep the pod alive after the run (debug). It is still reaped on the next `run`/`reap`, and removes
    /// itself through the pod-side TTL (`--max-cmd` plus 30 minutes unless `--pod-ttl` says otherwise).
    #[arg(long)]
    pub(crate) keep: bool,
    /// keep the pod warm for this many minutes so the next `exec` with the same image adopts it (skipping
    /// provision and the image pull). Reaped once the window passes. Supersedes `--keep`. At most 240.
    #[arg(
        long,
        value_name = "MINUTES",
        value_parser = clap::value_parser!(u64).range(1..=MAX_KEEP_WARM_MINUTES)
    )]
    pub(crate) keep_warm: Option<u64>,
}

#[derive(Args)]
pub(crate) struct ImageArgs {
    /// Dockerfile relative to benchmarks/ (the build context).
    #[arg(long, default_value = "docker/Dockerfile")]
    pub(crate) dockerfile: String,
    /// image tag to build, e.g. latest or slim.
    #[arg(long, default_value = "latest")]
    pub(crate) tag: String,
    /// GHCR owner (the image is ghcr.io/<owner>/poot-bench:<tag>).
    #[arg(long, default_value = "kikijiki", env = "GHCR_OWNER")]
    pub(crate) owner: String,
    /// push to GHCR after a successful build (needs `podman login ghcr.io`).
    #[arg(long)]
    pub(crate) push: bool,
    /// skip the build and only push an already-built local image (with --push).
    #[arg(long)]
    pub(crate) no_build: bool,
}

#[derive(Args)]
pub(crate) struct LogsArgs {
    /// log file name under the shared state dir's logs/ (e.g. run-123.log, image-slim.log). Omit to list.
    pub(crate) name: Option<String>,
    /// lines to show from the end.
    #[arg(long, default_value_t = 200)]
    pub(crate) tail: usize,
    /// keep printing new lines as they are appended.
    #[arg(long, short)]
    pub(crate) follow: bool,
}

#[derive(Args)]
pub(crate) struct ServeArgs {
    /// address to bind the dashboard. Localhost-only by default.
    #[arg(long, default_value = "127.0.0.1:8787")]
    pub(crate) addr: String,
}

#[derive(Args)]
pub(crate) struct RunArgs {
    /// git ref to benchmark (records the sha; a non-HEAD ref builds from a clean worktree).
    #[arg(long, default_value = "HEAD")]
    pub(crate) git_ref: String,
    /// comma-separated model ids (default: all poot-supported in manifest.toml).
    #[arg(long)]
    pub(crate) models: Option<String>,
    #[arg(long, default_value = "decode-curve")]
    pub(crate) scenario: String,
    /// comma-separated RunPod GPU type ids to try in order. All are sm_86 (the bench image's CUDA arch)
    /// with 24 GB; RTX 3090 is primary for comparability, A5000 the SECURE-cloud fallback.
    #[arg(long, default_value = "NVIDIA GeForce RTX 3090,NVIDIA RTX A5000")]
    pub(crate) gpu_types: String,
    /// SECURE by default: COMMUNITY hosts time out mid-pull on the large image.
    #[arg(long, default_value = "SECURE")]
    pub(crate) cloud: String,
    #[arg(long, default_value = "ghcr.io/kikijiki/poot-bench:latest")]
    pub(crate) image: String,
    /// Minimum host CUDA version; expands to RunPod `allowedCudaVersions` (all valid versions >= this).
    /// The bench image is built on CUDA 12.6.3. Empty disables the filter.
    #[arg(long, default_value = "12.6")]
    pub(crate) min_cuda_version: Option<String>,
    #[arg(long, default_value_t = 40)]
    pub(crate) container_disk_gb: u32,
    /// RunPod network volume to mount at /workspace as the model-weight cache; setup fetches only
    /// missing models from /workspace/models/<id>. Populate it with `just volume-sync`.
    #[arg(long)]
    pub(crate) network_volume_id: Option<String>,
    /// Datacenter the volume lives in (e.g. US-IL-1); the pod is pinned here.
    #[arg(long)]
    pub(crate) data_center: Option<String>,
    /// SSH private key (its .pub is injected as the pod PUBLIC_KEY).
    #[arg(long, default_value = "~/.ssh/id_ed25519")]
    pub(crate) ssh_key: String,
    #[arg(long, default_value_t = 4)]
    pub(crate) provision_attempts: u32,
    /// seconds to wait for a pod's SSH endpoint. The poot-bench image is ~34 GB, so a cold pull takes minutes.
    #[arg(long, default_value_t = 600)]
    pub(crate) provision_timeout_s: u64,
    #[arg(long, default_value_t = 300)]
    pub(crate) ssh_timeout_s: u64,
    /// seconds to wait for one repeat of a model's sweep; the wait for the model is this times `--repeats`.
    #[arg(long, default_value_t = 4 * 3600)]
    pub(crate) sweep_timeout_s: u64,
    /// runs of the whole matrix per model, one results directory each. 1 records a snapshot; a
    /// `bench compare` baseline needs 5.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub(crate) repeats: u32,
    /// do NOT terminate the pod at the end (debug). It will still be reaped on the next run. It removes
    /// itself through the pod-side TTL (every model's repeats at `--sweep-timeout-s`, plus 30 minutes).
    #[arg(long)]
    pub(crate) keep_pod: bool,
    /// skip building the poot runner; use this prebuilt binary.
    #[arg(long)]
    pub(crate) poot_bin: Option<String>,
    /// git add + commit the new snapshot(s) when done.
    #[arg(long)]
    pub(crate) commit: bool,
    /// build + validate config, but do not touch RunPod.
    #[arg(long)]
    pub(crate) dry_run: bool,
}

impl RunArgs {
    /// Model weights dir on the pod: the network volume when attached, else the container disk.
    pub(crate) fn models_root(&self) -> &'static str {
        if self.network_volume_id.is_some() {
            "/workspace/models"
        } else {
            "/root/models"
        }
    }
}

#[derive(Args)]
pub(crate) struct ReapArgs {
    /// terminate ALL pods with the name prefix, even ones tied to an in-progress run (force cleanup).
    #[arg(long)]
    pub(crate) all: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    /// Valid ids pass; invalid ids surface the typed `CliError::DataCenter`.
    #[test]
    fn parse_data_center_id_accepts_and_rejects() {
        assert_eq!(parse_data_center_id("EU-RO-1").unwrap(), "EU-RO-1");
        assert_eq!(parse_data_center_id("US-CA-2").unwrap(), "US-CA-2");
        assert_eq!(parse_data_center_id(" A1 ").unwrap(), "A1");

        // Loose format: non-empty, only A-Z, 0-9, and '-'. Anything else is a typed error.
        for bad in [
            "", "   ", "eu-ro-1", "EU RO 1", "EU_RO_1", "EU-RO-1!", "EU;RO",
        ] {
            let err = parse_data_center_id(bad).unwrap_err();
            assert!(
                matches!(err, CliError::DataCenter(_)),
                "expected typed DataCenter error for {bad:?}, got {err:?}"
            );
            assert!(
                err.to_string().contains("data-center id"),
                "error message should name the flag: {err}"
            );
        }
    }

    /// `exec --data-center` is repeatable and accepts comma lists; absent means empty (no pin).
    #[test]
    fn exec_data_center_flag_is_repeatable_and_comma_list() {
        let absent = Cli::try_parse_from(["poot-orchestrator", "exec", "--cmd", "true"]).unwrap();
        match &absent.command {
            Command::Exec(a) => assert!(a.data_center.is_empty()),
            _ => panic!("expected exec"),
        }

        let multi = Cli::try_parse_from([
            "poot-orchestrator",
            "exec",
            "--cmd",
            "true",
            "--data-center",
            "EU-RO-1",
            "--data-center",
            "US-CA-2,AP-IN-1",
        ])
        .unwrap();
        match &multi.command {
            Command::Exec(a) => assert_eq!(a.data_center, ["EU-RO-1", "US-CA-2", "AP-IN-1"]),
            _ => panic!("expected exec"),
        }

        // Invalid id fails at parse time with the typed error message.
        let bad = Cli::try_parse_from([
            "poot-orchestrator",
            "exec",
            "--cmd",
            "true",
            "--data-center",
            "lower case",
        ]);
        let Err(err) = bad else {
            panic!("invalid data-center must fail parse");
        };
        assert!(
            err.to_string().contains("data-center id"),
            "parse error should carry the typed message: {err}"
        );
    }

    /// SC-003: `--keep-warm` above the cap is refused at parse time; the cap itself is accepted.
    /// Removing the range from the value parser makes 100000 parse and this red.
    #[test]
    fn exec_keep_warm_is_capped() {
        let over = Cli::try_parse_from([
            "poot-orchestrator",
            "exec",
            "--cmd",
            "true",
            "--keep-warm",
            "100000",
        ]);
        let Err(err) = over else {
            panic!("--keep-warm 100000 must be refused");
        };
        assert!(err.to_string().contains("100000"), "{err}");

        let at_cap = Cli::try_parse_from([
            "poot-orchestrator",
            "exec",
            "--cmd",
            "true",
            "--keep-warm",
            &MAX_KEEP_WARM_MINUTES.to_string(),
        ])
        .unwrap();
        match at_cap.command {
            Command::Exec(a) => assert_eq!(a.keep_warm, Some(MAX_KEEP_WARM_MINUTES)),
            _ => panic!("expected exec"),
        }
    }

    /// `--max-cmd` is mandatory in effect (2h default), parses s/m/h and refuses zero, junk and values
    /// above the ceiling.
    #[test]
    fn exec_max_cmd_parses_defaults_and_refuses() {
        assert_eq!(parse_max_cmd("2s").unwrap(), Duration::from_secs(2));
        assert_eq!(parse_max_cmd("30m").unwrap(), Duration::from_secs(1800));
        assert_eq!(parse_max_cmd("12h").unwrap(), MAX_CMD_CEILING);
        for bad in ["", "2", "s", "2d", "1.5h", "-1s", "2 s x"] {
            assert!(
                matches!(parse_max_cmd(bad), Err(CliError::Duration(_))),
                "{bad:?}"
            );
        }
        for bad in ["0s", "13h", "99999999999999999h"] {
            assert!(
                matches!(
                    parse_max_cmd(bad),
                    Err(CliError::DurationRange(_)) | Err(CliError::Duration(_))
                ),
                "{bad:?}"
            );
        }
        let Command::Exec(a) = Cli::try_parse_from(["poot-orchestrator", "exec", "--cmd", "x"])
            .unwrap()
            .command
        else {
            panic!("expected exec");
        };
        assert_eq!(a.max_cmd, Duration::from_secs(2 * 3600));
        assert!(
            Cli::try_parse_from([
                "poot-orchestrator",
                "exec",
                "--cmd",
                "x",
                "--runpod-api-key",
                "k"
            ])
            .is_err()
        );
    }
}
