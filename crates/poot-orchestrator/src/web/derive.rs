//! What the dashboard derives from the state DB, the run logs and RunPod's own pod list, as pure functions over
//! their inputs so each can be tested without a server: the capacity wait of a run, what a pod has cost, whether
//! RunPod still knows a pod the DB calls live, and the spend/failure totals of the overview.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;

use crate::clock::{now_secs, secs_from_iso};
use crate::db::Pod;

const DAY_SECS: i64 = 24 * 3600;
/// How long RunPod may take to list a pod it just created: a pod created less than this before the list was read
/// is not "gone" for being missing from it.
const LISTING_GRACE_SECS: i64 = 60;
/// How often the background check lists the pods RunPod knows.
const VERIFY_EVERY: Duration = Duration::from_secs(30);

/// A run's wait for a GPU, as the run log last reported it (`no capacity for any of ..., waiting 71s (pass 4,
/// waiting 1 min so far)`; a miss on the price cap says `no price cap (--max-price-per-hour 0.5) for ...`).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct CapacityWait {
    pub pass: u32,
    pub waited_min: u64,
    /// The pause the run took after that pass, seconds.
    pub retry_secs: u64,
    /// The `--max-price-per-hour` the last miss was against, when it was one.
    pub price_cap: Option<f64>,
}

/// The number that follows `key` in `line`, up to the first non-digit.
fn number_after(line: &str, key: &str) -> Option<u64> {
    let rest = &line[line.rfind(key)? + key.len()..];
    let digits: String = rest.chars().take_while(char::is_ascii_digit).collect();
    digits.parse().ok()
}

/// The last capacity-wait line of a run log, parsed.
pub(super) fn capacity_wait(log: &str) -> Option<CapacityWait> {
    let line = log.lines().rev().find(|l| {
        l.contains(" so far)") && (l.contains("] no capacity for") || l.contains("] no price cap"))
    })?;
    let price_cap = line.find("--max-price-per-hour ").and_then(|at| {
        let rest = &line[at + "--max-price-per-hour ".len()..];
        rest[..rest.find(')')?].parse().ok()
    });
    // `<head>, waiting 71s (pass 4, waiting 1 min so far)`: the pause is in the head, the pass and the minutes after it.
    let (head, tail) = line.split_once(" (pass ")?;
    if !tail.contains(" min so far)") {
        return None;
    }
    Some(CapacityWait {
        pass: tail.split(',').next()?.parse().ok()?,
        waited_min: number_after(tail, ", waiting ")?,
        retry_secs: number_after(head, ", waiting ")?,
        price_cap,
    })
}

/// The log's last line, without its `[hh:mm:ss] ` stamp: why a run that ended, ended.
pub(super) fn last_line(log: &str) -> Option<String> {
    let line = log.lines().rev().map(str::trim).find(|l| !l.is_empty())?;
    let line = match line.strip_prefix('[').and_then(|r| r.split_once("] ")) {
        Some((_, rest)) => rest,
        None => line,
    };
    Some(line.chars().take(240).collect())
}

/// How many times the run replaced its pod (a lost pod or a host fault), counted from its log.
pub(super) fn replacements(log: &str) -> usize {
    log.lines()
        .filter(|l| l.contains(", provisioning replacement, "))
        .count()
}

/// Seconds the run still needs, extrapolated from the sets it has gathered of the sets it will gather; `None`
/// before the first set (no rate yet) and once all are in.
pub(super) fn eta_secs(elapsed_secs: i64, done: usize, total: usize) -> Option<u64> {
    if done == 0 || done >= total || elapsed_secs <= 0 {
        return None;
    }
    let remaining = (total - done) as u64;
    Some(elapsed_secs as u64 * remaining / done as u64)
}

/// The pods RunPod listed at one moment (id -> its `desiredStatus`), or why it could not be asked.
#[derive(Debug, Clone)]
pub(super) struct RunpodView {
    pub checked_at: i64,
    pub pods: Result<HashMap<String, String>, String>,
}

/// Whether RunPod agrees with the DB about a pod.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub(super) enum Verified {
    /// The DB says the pod is terminated: nothing to check.
    Terminated,
    /// RunPod lists the pod; `runpod_status` is its `desiredStatus` there.
    Confirmed {
        runpod_status: String,
        checked_secs_ago: i64,
    },
    /// RunPod does not list a pod the DB calls live: it was removed (or reaped) without the DB hearing of it.
    Gone { checked_secs_ago: i64 },
    /// RunPod could not be asked (no key, no network, an error): the DB's word is all there is.
    Unverified { reason: String },
}

pub(super) fn verify(pod: &Pod, view: Option<&RunpodView>, now: i64) -> Verified {
    if pod.status == "terminated" {
        return Verified::Terminated;
    }
    let Some(view) = view else {
        return Verified::Unverified {
            reason: "RunPod has not been asked yet".into(),
        };
    };
    let checked_secs_ago = (now - view.checked_at).max(0);
    match &view.pods {
        Err(reason) => Verified::Unverified {
            reason: reason.clone(),
        },
        Ok(listed) => match listed.get(&pod.id) {
            Some(runpod_status) => Verified::Confirmed {
                runpod_status: runpod_status.clone(),
                checked_secs_ago,
            },
            None if created_within_grace(pod, view.checked_at) => Verified::Unverified {
                reason: "created moments before RunPod's list was read".into(),
            },
            None => Verified::Gone { checked_secs_ago },
        },
    }
}

/// Was the pod created less than [`LISTING_GRACE_SECS`] before `checked_at` (or after it)?
fn created_within_grace(pod: &Pod, checked_at: i64) -> bool {
    secs_from_iso(&pod.created_at).is_some_and(|created| created > checked_at - LISTING_GRACE_SECS)
}

/// The span a pod has been billable: from its creation until it was terminated, or RunPod stopped listing it
/// (the DB's last update then), or now.
fn lifetime(pod: &Pod, verified: &Verified, now: i64) -> Option<(i64, i64)> {
    let start = secs_from_iso(&pod.created_at)?;
    let ended = pod.status == "terminated" || matches!(verified, Verified::Gone { .. });
    let end = if ended {
        secs_from_iso(&pod.updated_at)?
    } else {
        now
    };
    Some((start, end.max(start)))
}

/// What a pod has run for and cost, estimated as its hourly price times its lifetime (there is no billing API).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub(super) struct Meter {
    pub uptime_secs: i64,
    pub accrued_usd: Option<f64>,
}

pub(super) fn meter(pod: &Pod, verified: &Verified, now: i64) -> Option<Meter> {
    let (start, end) = lifetime(pod, verified, now)?;
    Some(Meter {
        uptime_secs: end - start,
        accrued_usd: pod
            .cost_per_hr
            .map(|per_hr| per_hr * (end - start) as f64 / 3600.0),
    })
}

/// A pod that is billing right now, as far as anyone can tell: not terminated and not missing from RunPod.
pub(super) fn is_billing(pod: &Pod, verified: &Verified) -> bool {
    pod.status != "terminated" && !matches!(verified, Verified::Gone { .. })
}

/// The estimated spend over the last 24 hours: each pod's price over the part of its lifetime inside the window.
pub(super) fn spend_24h(pods: &[(&Pod, Verified)], now: i64) -> f64 {
    pods.iter()
        .filter_map(|(pod, verified)| {
            let (start, end) = lifetime(pod, verified, now)?;
            let overlap = (end - start.max(now - DAY_SECS)).max(0);
            Some(pod.cost_per_hr? * overlap as f64 / 3600.0)
        })
        .sum()
}

/// The estimated spend per hour right now: the price of every pod that is billing.
pub(super) fn burn_per_hr(pods: &[(&Pod, Verified)]) -> f64 {
    pods.iter()
        .filter(|(pod, verified)| is_billing(pod, verified))
        .filter_map(|(pod, _)| pod.cost_per_hr)
        // Not `.sum()`: the sum of no prices is -0.0, which the overview would print as "-0".
        .fold(0.0, |total, price| total + price)
}

/// What the overview reads of a run log.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct LogFacts {
    pub replacements: usize,
    pub last_line: Option<String>,
}

/// Where run logs are read from: the state directory in production, memory in tests.
pub(super) trait LogSource {
    /// `(size, mtime)` of the log, which changes whenever the run writes to it; `None` for no such log.
    fn stamp(&self, name: &str) -> Option<(u64, u64)>;
    fn read(&self, name: &str) -> String;
}

/// The facts of each log, kept until the log changes: the overview is polled every few seconds, and a log of a run
/// that has ended never changes again, so it is read once.
#[derive(Default)]
pub(super) struct LogCache(HashMap<String, ((u64, u64), LogFacts)>);

impl LogCache {
    pub(super) fn facts(&mut self, source: &dyn LogSource, name: &str) -> LogFacts {
        let Some(stamp) = source.stamp(name) else {
            return LogFacts {
                replacements: 0,
                last_line: None,
            };
        };
        if let Some((cached, facts)) = self.0.get(name)
            && *cached == stamp
        {
            return facts.clone();
        }
        let log = source.read(name);
        let facts = LogFacts {
            replacements: replacements(&log),
            last_line: last_line(&log),
        };
        self.0.insert(name.to_string(), (stamp, facts.clone()));
        facts
    }
}

pub(super) type SharedView = Arc<Mutex<Option<RunpodView>>>;

/// Ask RunPod for its pod list every [`VERIFY_EVERY`] on a background thread, so a request never waits on the
/// network. Read-only: the check lists pods and nothing else.
pub(super) fn spawn_verifier() -> SharedView {
    let shared: SharedView = Arc::default();
    let out = Arc::clone(&shared);
    std::thread::spawn(move || {
        loop {
            let pods = crate::workflow::runpod_api_key()
                .and_then(crate::runpod::RunPod::new)
                .and_then(|rp| rp.list_pods())
                .map(|listed| {
                    listed
                        .into_iter()
                        .map(|p| (p.id, p.desired_status))
                        .collect()
                })
                .map_err(|e| format!("{e:#}"));
            if let Ok(mut slot) = out.lock() {
                *slot = Some(RunpodView {
                    checked_at: now_secs(),
                    pods,
                });
            }
            std::thread::sleep(VERIFY_EVERY);
        }
    });
    shared
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pod(id: &str, status: &str, per_hr: Option<f64>, created: &str, updated: &str) -> Pod {
        Pod {
            id: id.into(),
            gpu_type: "NVIDIA GeForce RTX 3090".into(),
            cloud: "COMMUNITY".into(),
            cost_per_hr: per_hr,
            image: "img".into(),
            status: status.into(),
            ssh_host: None,
            ssh_port: None,
            current_run: None,
            created_at: created.into(),
            updated_at: updated.into(),
        }
    }

    fn t(iso: &str) -> i64 {
        secs_from_iso(iso).unwrap()
    }

    const WAIT: &str = "[23:15:09] no capacity for any of NVIDIA GeForce RTX 3090, waiting 71s (pass 4, waiting 1 min so far)\n";

    /// The last wait line gives the pass, the minutes waited and the pause; the first pass is not the answer.
    /// Mutation: take the first matching line instead of the last and the pass reads 1.
    #[test]
    fn a_capacity_wait_is_read_from_the_last_wait_line() {
        let log = format!(
            "[23:13:25] no capacity for any of X, waiting 21s (pass 1, waiting 0 min so far)\n{WAIT}[23:16:21] provision pass 5, try 5: X\n"
        );
        assert_eq!(
            capacity_wait(&log),
            Some(CapacityWait {
                pass: 4,
                waited_min: 1,
                retry_secs: 71,
                price_cap: None
            })
        );
    }

    /// A miss on the price cap carries the cap.
    #[test]
    fn a_price_cap_miss_names_the_cap() {
        let log = "[10:00:00] no price cap (--max-price-per-hour 0.5) for any of NVIDIA L40S, waiting 30s (pass 2, waiting 7 min so far)\n";
        let wait = capacity_wait(log).unwrap();
        assert_eq!(
            (wait.pass, wait.waited_min, wait.retry_secs, wait.price_cap),
            (2, 7, 30, Some(0.5))
        );
    }

    #[test]
    fn a_log_without_a_wait_has_none() {
        assert_eq!(capacity_wait("[10:00:00] pod abc created\n"), None);
        assert_eq!(capacity_wait(""), None);
    }

    #[test]
    fn the_last_line_loses_its_stamp() {
        assert_eq!(
            last_line("[1] a\n[10:00:01] build failed: boom\n\n").as_deref(),
            Some("build failed: boom")
        );
        assert_eq!(last_line("\n  \n"), None);
    }

    #[test]
    fn replacements_are_counted_from_the_log() {
        let log = "[1] pod a has no working GPU, provisioning replacement, 3 of 15 repeats already gathered\n[2] x\n[3] pod b lost, provisioning replacement, 0 of 15 repeats already gathered\n";
        assert_eq!(replacements(log), 2);
    }

    #[test]
    fn eta_extrapolates_from_the_sets_gathered() {
        assert_eq!(eta_secs(600, 2, 8), Some(1800));
        assert_eq!(eta_secs(600, 0, 8), None);
        assert_eq!(eta_secs(600, 8, 8), None);
    }

    fn view(checked_at: i64, pods: Result<&[(&str, &str)], &str>) -> RunpodView {
        RunpodView {
            checked_at,
            pods: pods
                .map(|p| {
                    p.iter()
                        .map(|(id, s)| ((*id).into(), (*s).into()))
                        .collect()
                })
                .map_err(String::from),
        }
    }

    /// A pod the DB calls live is confirmed only when RunPod lists it; one RunPod does not list is gone, and a
    /// failed call is unverified, never gone. Mutation: treat a failed list as an empty list and a live pod reads gone.
    #[test]
    fn a_live_pod_is_checked_against_runpods_list() {
        let live = pod(
            "p1",
            "busy",
            Some(1.0),
            "2026-10-23T10:00:00Z",
            "2026-10-23T10:00:00Z",
        );
        let now = t("2026-10-23T12:00:00Z");
        let ok = view(now - 20, Ok(&[("p1", "RUNNING")]));
        assert_eq!(
            verify(&live, Some(&ok), now),
            Verified::Confirmed {
                runpod_status: "RUNNING".into(),
                checked_secs_ago: 20
            }
        );
        let other = view(now - 20, Ok(&[("p2", "RUNNING")]));
        assert_eq!(
            verify(&live, Some(&other), now),
            Verified::Gone {
                checked_secs_ago: 20
            }
        );
        let failed = view(now - 20, Err("RUNPOD_API_KEY is not set"));
        assert_eq!(
            verify(&live, Some(&failed), now),
            Verified::Unverified {
                reason: "RUNPOD_API_KEY is not set".into()
            }
        );
        assert!(matches!(
            verify(&live, None, now),
            Verified::Unverified { .. }
        ));
        let done = pod(
            "p1",
            "terminated",
            None,
            "2026-10-23T10:00:00Z",
            "2026-10-23T11:00:00Z",
        );
        assert_eq!(verify(&done, Some(&other), now), Verified::Terminated);
    }

    /// A pod created moments before RunPod's list was read is not "gone" for missing from it: it is unverified, so
    /// it still counts as billing. One that was created well before is gone. Mutation: drop the grace and the young
    /// pod reads gone.
    #[test]
    fn a_pod_too_young_to_be_listed_is_not_gone() {
        let now = t("2026-10-23T12:00:00Z");
        let empty = view(now, Ok(&[]));
        let young = pod(
            "y",
            "provisioning",
            Some(1.0),
            "2026-10-23T11:59:30Z",
            "2026-10-23T11:59:30Z",
        );
        let verified = verify(&young, Some(&empty), now);
        assert!(
            matches!(verified, Verified::Unverified { .. }),
            "{verified:?}"
        );
        assert!(is_billing(&young, &verified));
        let created_after = pod(
            "z",
            "provisioning",
            Some(1.0),
            "2026-10-23T12:00:10Z",
            "2026-10-23T12:00:10Z",
        );
        assert!(matches!(
            verify(&created_after, Some(&empty), now),
            Verified::Unverified { .. }
        ));
        let old = pod(
            "o",
            "provisioning",
            Some(1.0),
            "2026-10-23T11:50:00Z",
            "2026-10-23T11:50:00Z",
        );
        assert!(matches!(
            verify(&old, Some(&empty), now),
            Verified::Gone { .. }
        ));
    }

    /// A pod bills until it was terminated or RunPod stopped listing it, not until now. Mutation: end every pod
    /// at `now` and the stale pod keeps accruing.
    #[test]
    fn a_pod_stops_accruing_when_it_ends_or_goes_missing() {
        let now = t("2026-10-23T12:00:00Z");
        let live = pod(
            "a",
            "busy",
            Some(2.0),
            "2026-10-23T10:00:00Z",
            "2026-10-23T10:05:00Z",
        );
        let confirmed = Verified::Confirmed {
            runpod_status: "RUNNING".into(),
            checked_secs_ago: 5,
        };
        let m = meter(&live, &confirmed, now).unwrap();
        assert_eq!((m.uptime_secs, m.accrued_usd), (7200, Some(4.0)));

        let stale = pod(
            "b",
            "provisioning",
            Some(2.0),
            "2026-10-23T10:00:00Z",
            "2026-10-23T10:30:00Z",
        );
        let m = meter(
            &stale,
            &Verified::Gone {
                checked_secs_ago: 5,
            },
            now,
        )
        .unwrap();
        assert_eq!((m.uptime_secs, m.accrued_usd), (1800, Some(1.0)));

        let ended = pod(
            "c",
            "terminated",
            Some(2.0),
            "2026-10-23T10:00:00Z",
            "2026-10-23T11:00:00Z",
        );
        let m = meter(&ended, &Verified::Terminated, now).unwrap();
        assert_eq!(m.uptime_secs, 3600);

        let unpriced = pod(
            "d",
            "busy",
            None,
            "2026-10-23T11:00:00Z",
            "2026-10-23T11:00:00Z",
        );
        assert_eq!(meter(&unpriced, &confirmed, now).unwrap().accrued_usd, None);
    }

    /// Spend counts only the part of a pod's life inside the last 24 h; burn counts only pods that bill.
    #[test]
    fn spend_and_burn_total_the_pods() {
        let now = t("2026-10-24T12:00:00Z");
        // Created 36 h ago, terminated 12 h ago: 12 h of its 24 h window... overlap is [now-24h, now-12h].
        let old = pod(
            "old",
            "terminated",
            Some(1.0),
            "2026-10-23T00:00:00Z",
            "2026-10-24T00:00:00Z",
        );
        let live = pod(
            "live",
            "busy",
            Some(2.0),
            "2026-10-24T10:00:00Z",
            "2026-10-24T10:00:00Z",
        );
        let gone = pod(
            "gone",
            "provisioning",
            Some(5.0),
            "2026-10-24T09:00:00Z",
            "2026-10-24T09:00:00Z",
        );
        let confirmed = Verified::Confirmed {
            runpod_status: "RUNNING".into(),
            checked_secs_ago: 1,
        };
        let pods = [
            (&old, Verified::Terminated),
            (&live, confirmed),
            (
                &gone,
                Verified::Gone {
                    checked_secs_ago: 1,
                },
            ),
        ];
        // old: 12 h x 1.0 inside the window; live: 2 h x 2.0; gone: stopped at its update, 0 h.
        assert!((spend_24h(&pods, now) - 16.0).abs() < 1e-9);
        assert!((burn_per_hr(&pods) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn the_burn_rate_with_nothing_billing_is_positive_zero() {
        let burn = burn_per_hr(&[]);
        assert_eq!(burn, 0.0);
        assert!(
            burn.is_sign_positive(),
            "burn_usd_per_hr must serialize as 0.0, not -0.0"
        );
        assert_eq!(serde_json::to_string(&burn).unwrap(), "0.0");
    }
}
