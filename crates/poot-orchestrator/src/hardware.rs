//! The hardware a result set was measured on, and the guard that keeps `--extend-set` on it. TTFT and the
//! per-point times depend on the host CPU's launch overhead, so repeats from two host CPUs are two populations:
//! pooled into one set they make a two-mode cell that no repeat count narrows. Sets of different hosts are
//! different cells.

use std::fmt;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};

use crate::gather;
use crate::heal::Unrecoverable;
use crate::manifest::Model;
use crate::ssh::Remote;

/// How long the hardware probe may take on the pod.
const PROBE_BUDGET: Duration = Duration::from_secs(60);

/// Prints the GPU name and driver (as `nvidia-smi` reports them, which is what the harness records in
/// `env.json`) on the first line and the host CPU model on the second.
const PROBE_COMMAND: &str = "nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -n 1; \
                             grep -m1 'model name' /proc/cpuinfo | cut -d: -f2";

/// What `env.json` records about the machine a set ran on, and what is compared.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Hardware {
    pub gpu_name: String,
    pub gpu_driver: String,
    pub cpu: String,
}

impl fmt::Display for Hardware {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} (driver {}) on {}",
            self.gpu_name, self.gpu_driver, self.cpu
        )
    }
}

impl Hardware {
    /// The hardware a set's `env.json` names. Every field must be there: a set that does not say where it ran
    /// cannot be extended safely.
    pub(crate) fn of_env(text: &str) -> Result<Self> {
        let env: serde_json::Value = serde_json::from_str(text).context("env.json is not JSON")?;
        let field = |name: &str| {
            env.get(name)
                .and_then(|v| v.as_str())
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .ok_or_else(|| anyhow!("env.json has no {name}"))
        };
        Ok(Self {
            gpu_name: field("gpu_name")?,
            gpu_driver: field("gpu_driver")?,
            cpu: field("cpu")?,
        })
    }

    /// The hardware of the machine behind `remote`.
    pub(crate) fn probe<R: Remote + ?Sized>(remote: &R) -> Result<Self> {
        let run = remote
            .ssh_until(PROBE_COMMAND, Instant::now() + PROBE_BUDGET)
            .context("remote hardware probe")?;
        Self::of_probe(&run.stdout)
    }

    fn of_probe(stdout: &str) -> Result<Self> {
        let mut lines = stdout.lines();
        let smi = lines.next().unwrap_or_default();
        let (gpu_name, gpu_driver) = smi
            .split_once(',')
            .ok_or_else(|| anyhow!("nvidia-smi printed no `name, driver` line: {stdout:?}"))?;
        let cpu = lines.next().unwrap_or_default().trim();
        if cpu.is_empty() {
            return Err(anyhow!("/proc/cpuinfo printed no model name: {stdout:?}"));
        }
        Ok(Self {
            gpu_name: gpu_name.trim().to_string(),
            gpu_driver: gpu_driver.trim().to_string(),
            cpu: cpu.to_string(),
        })
    }

    /// What differs from `other`, by field name.
    fn differences(&self, other: &Self) -> Vec<&'static str> {
        [
            ("GPU", self.gpu_name != other.gpu_name),
            ("driver", self.gpu_driver != other.gpu_driver),
            ("host CPU", self.cpu != other.cpu),
        ]
        .into_iter()
        .filter_map(|(name, differs)| differs.then_some(name))
        .collect()
    }
}

/// Every gathered repeat of `models` under the set id `set`, with the hardware its `env.json` names.
fn extended_sets(results: &Path, set: &str, models: &[Model]) -> Result<Vec<(String, Hardware)>> {
    let mut found = Vec::new();
    for model in models {
        // A run of one repeat names its set without the `-r<N>` suffix.
        let bare = format!("{set}-{}", model.id);
        let names = std::iter::once(bare.clone())
            .chain((1..=gather::MAX_EXTENDED_REPEATS).map(|n| format!("{bare}-r{n}")));
        for name in names {
            let Some(dir) = gather::local_set(results, &name) else {
                continue;
            };
            let text = std::fs::read_to_string(results.join(&dir).join("env.json"))
                .with_context(|| format!("read {dir}/env.json"))?;
            let hardware = Hardware::of_env(&text)
                .with_context(|| format!("the set {dir} cannot be compared"))?;
            found.push((dir, hardware));
        }
    }
    Ok(found)
}

/// Refuse an extension that would pool repeats from different hardware: the pod behind `remote` must have the
/// GPU, driver and host CPU of every set `--extend-set {set}` adds to. The error is [`Unrecoverable`], so the run
/// ends and its pod is terminated.
pub(crate) fn require_same_hardware<R: Remote + ?Sized>(
    remote: &R,
    results: &Path,
    set: &str,
    models: &[Model],
) -> Result<()> {
    let sets = extended_sets(results, set, models)?;
    let pod = Hardware::probe(remote)?;
    for (dir, hardware) in &sets {
        let differs = pod.differences(hardware);
        if !differs.is_empty() {
            return Err(Unrecoverable(format!(
                "--extend-set {set}: this pod's {} differs from {dir} ({hardware}); the pod is {pod}. \
                 Sets of different hosts are different cells: run a new set on this pod, or extend on the pod \
                 the set ran on (--reuse-pod).",
                differs.join(" and "),
            ))
            .into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ssh::{DeadlineError, Run};

    /// A machine that answers the hardware probe with canned output.
    struct FakeHost(&'static str);

    impl Remote for FakeHost {
        fn wait_ssh(&self, _timeout: Duration) -> Result<()> {
            Ok(())
        }

        fn scp_up(&self, _local: &str, _remote: &str) -> Result<()> {
            Ok(())
        }

        fn ssh_until(&self, cmd: &str, _deadline: Instant) -> Result<Run, DeadlineError> {
            assert_eq!(cmd, PROBE_COMMAND);
            Ok(Run {
                code: 0,
                stdout: self.0.to_string(),
                stderr: String::new(),
            })
        }
    }

    const EPYC_7663: &str =
        "NVIDIA GeForce RTX 3090, 580.95.05\n AMD EPYC 7663 56-Core Processor\n";
    const EPYC_7452: &str =
        "NVIDIA GeForce RTX 3090, 580.95.05\n AMD EPYC 7452 32-Core Processor\n";
    const SET: &str = "run-1792676008-4242";

    fn model(id: &str) -> Model {
        let manifest = format!(
            "[[model]]\nid = \"{id}\"\nhf_repo = \"org/{id}\"\nframeworks.poot = {{ precision = \"bf16\" }}\n"
        );
        crate::manifest::parse_manifest_models(&manifest)
            .unwrap()
            .remove(0)
    }

    /// A results directory holding `repeats` repeats of the set `SET`-`model`, measured on `cpu`.
    fn results_with(tag: &str, model: &str, repeats: u32, cpu: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("poot-hardware-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        for n in 1..=repeats {
            let name = gather::canonical_dir(
                &format!("{SET}-{model}-r{n}"),
                Some("NVIDIA GeForce RTX 3090"),
            )
            .unwrap();
            let set = dir.join(name);
            std::fs::create_dir_all(&set).unwrap();
            std::fs::write(
                set.join("env.json"),
                format!(
                    r#"{{"gpu_name": "NVIDIA GeForce RTX 3090", "gpu_driver": "580.95.05", "cpu": " {cpu}"}}"#
                ),
            )
            .unwrap();
            for file in ["results.jsonl", "report.md"] {
                std::fs::write(set.join(file), "{}\n").unwrap();
            }
        }
        dir
    }

    #[test]
    fn the_probe_reads_the_gpu_driver_and_cpu_the_way_the_harness_records_them() {
        let got = Hardware::probe(&FakeHost(EPYC_7663)).unwrap();
        assert_eq!(got.gpu_name, "NVIDIA GeForce RTX 3090");
        assert_eq!(got.gpu_driver, "580.95.05");
        assert_eq!(got.cpu, "AMD EPYC 7663 56-Core Processor");
        assert!(Hardware::of_probe("no nvidia-smi output").is_err());
    }

    /// The extension's pod has the sets' GPU, driver and CPU.
    #[test]
    fn an_extension_on_the_same_hardware_is_allowed() {
        let results = results_with("same", "qwen3-4b", 2, "AMD EPYC 7663 56-Core Processor");
        require_same_hardware(&FakeHost(EPYC_7663), &results, SET, &[model("qwen3-4b")]).unwrap();
    }

    /// The incident: five repeats on an EPYC 7663 host, the extension on an EPYC 7452 one. Red when the check
    /// is skipped.
    #[test]
    fn an_extension_on_another_host_cpu_is_refused_and_says_why() {
        let results = results_with("cpu", "qwen3-4b", 2, "AMD EPYC 7663 56-Core Processor");
        let err = require_same_hardware(&FakeHost(EPYC_7452), &results, SET, &[model("qwen3-4b")])
            .unwrap_err();
        assert!(
            err.downcast_ref::<Unrecoverable>().is_some(),
            "the run must end, and with it the pod: {err:#}"
        );
        let text = format!("{err:#}");
        assert!(text.contains("host CPU"), "{text}");
        assert!(
            text.contains("EPYC 7452") && text.contains("EPYC 7663"),
            "{text}"
        );
        assert!(text.contains("--reuse-pod"), "{text}");
    }

    #[test]
    fn a_different_driver_is_refused_too() {
        let results = results_with("driver", "qwen3-4b", 1, "AMD EPYC 7663 56-Core Processor");
        let err = require_same_hardware(
            &FakeHost("NVIDIA GeForce RTX 3090, 570.1.2\n AMD EPYC 7663 56-Core Processor\n"),
            &results,
            SET,
            &[model("qwen3-4b")],
        )
        .unwrap_err();
        assert!(format!("{err:#}").contains("driver"), "{err:#}");
    }

    /// A set whose `env.json` does not say where it ran cannot be shown to match.
    #[test]
    fn a_set_that_does_not_name_its_hardware_cannot_be_extended() {
        let results = results_with("noenv", "qwen3-4b", 1, "AMD EPYC 7663 56-Core Processor");
        let env = std::fs::read_dir(&results)
            .unwrap()
            .next()
            .unwrap()
            .unwrap()
            .path()
            .join("env.json");
        std::fs::write(env, r#"{"gpu_name": "NVIDIA GeForce RTX 3090"}"#).unwrap();
        let err = require_same_hardware(&FakeHost(EPYC_7663), &results, SET, &[model("qwen3-4b")])
            .unwrap_err();
        assert!(format!("{err:#}").contains("cannot be compared"), "{err:#}");
    }
}
