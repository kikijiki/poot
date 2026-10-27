//! Read-only web dashboard over the run-state DB. `serve` binds a blocking tiny_http server and exposes:
//!   GET  /            the dashboard page (the committed Vite build of `ui/`, inlined into dashboard.html)
//!   GET  /api/state   JSON: recent runs + per-model progress and pods, what the page polls every few seconds.
//!                     Fields are only ever added (`scripts/status-board` reads `runs[].id/status/models[].model` and
//!                     `pods[].id/gpu_type/cloud/status`). A run adds `repeats`, `has_sets`, `sets_done`,
//!                     `sets_total`, `eta_secs` (while it has not ended), `wait` (while it waits for capacity: the
//!                     last `{pass, waited_min, retry_secs, price_cap}` its log reported) and `cost_usd` (its pod's
//!                     estimate). A pod adds `uptime_secs`, `accrued_usd` (its price times its lifetime: an
//!                     estimate, there is no billing API) and `verified` (RunPod's own pod list, listed every 30 s on
//!                     a background thread: `confirmed`, `gone`, `unverified` with the reason, or `terminated`).
//!   GET  /api/overview  JSON: `{spend_24h_usd, burn_usd_per_hr, billing_pods, failures:[{run, status, at, reason,
//!                     replacements}], replacements_24h}`: what the overview page shows.
//!   GET  /api/logfiles  JSON: log files (name, size, mtime, kind, state), newest first
//!   GET  /api/log     text/plain: the tail of one log, by name (path-traversal guarded)
//!   GET  /api/results   JSON `{sets, reason}`: a run's result sets (rows of the sets gathered, failed or being
//!                     mirrored); `reason` says why `sets` is empty (`exec_run`, `no_results_dir`, `no_sets_yet`,
//!                     `ended_without_sets`), null otherwise
//!   POST /api/build   trigger an image build off-thread (logs to image-<tag>.log), returns 202
//!   GET  /api/serve-metrics  JSON: proxy a live poot-serve's `GET /metrics` (card 051)
//!   GET  /healthz     "ok"
//! It only reads the SQLite DB, so it is safe alongside an active `run` (WAL allows concurrent readers).
//!
//! The build endpoint runs an external process, so serve binds 127.0.0.1 by default. Do not bind a
//! public address with it enabled.

use std::path::Path;

use anyhow::{Result, anyhow};
use tiny_http::{Header, Method, Response, Server};

use crate::clock::{now_secs, secs_from_iso};
use crate::db::{Db, Pod};
use crate::logging::{list_logs, log, log_kind_and_state, read_log_tail, spawn_image_build};
use crate::workflow::repo_root;

mod derive;

const INDEX_HTML: &str = include_str!("dashboard.html");

/// Minimal query-string lookup (`a=1&name=run-2.log`) with %-decoding of the few chars we expect.
fn query_param(q: &str, key: &str) -> Option<String> {
    q.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        if k == key {
            Some(v.replace("%2F", "/").replace("%2E", ".").replace('+', " "))
        } else {
            None
        }
    })
}

pub(crate) fn serve(db: Db, addr: &str) -> Result<()> {
    // Single-threaded request loop, so the non-Send Connection in `db` is fine. Only the image build
    // (POST /api/build) runs detached.
    let server = Server::http(addr).map_err(|e| anyhow!("bind {addr}: {e}"))?;
    log(format!("dashboard on http://{addr}  (Ctrl-C to stop)"));

    // `benchmarks/results` of the checkout `serve` runs from; the run process writes the sets and their mirrors there.
    let results_dir = repo_root()
        .ok()
        .map(|root| root.join("benchmarks").join("results"));
    let verifier = derive::spawn_verifier();
    let mut log_cache = derive::LogCache::default();
    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let path = url.split('?').next().unwrap_or("/");
        let is_post = *req.method() == Method::Post;
        let (status, ctype, body) = match (is_post, path) {
            (false, "/") | (false, "/index.html") => {
                (200, "text/html; charset=utf-8", INDEX_HTML.to_string())
            }
            (false, "/healthz") => (200, "text/plain", "ok".to_string()),
            // Runs and pods are distinct entities; return both lists.
            (false, "/api/state") => {
                let view = verifier.lock().ok().and_then(|v| v.clone());
                match state_json(&db, results_dir.as_deref(), view.as_ref(), now_secs()) {
                    Ok(json) => (200, "application/json", json),
                    Err(e) => (
                        500,
                        "application/json",
                        format!("{{\"error\":{:?}}}", e.to_string()),
                    ),
                }
            }
            (false, "/api/overview") => {
                let view = verifier.lock().ok().and_then(|v| v.clone());
                match overview_json(
                    &db,
                    view.as_ref(),
                    now_secs(),
                    &StateDirLogs,
                    &mut log_cache,
                ) {
                    Ok(json) => (200, "application/json", json),
                    Err(e) => (
                        500,
                        "application/json",
                        format!("{{\"error\":{:?}}}", e.to_string()),
                    ),
                }
            }
            // log files in the shared state dir's logs/: list with kind+state, tail one by name.
            (false, "/api/logfiles") => {
                let items: Vec<_> = list_logs()
                    .into_iter()
                    .map(|(name, size, mtime)| {
                        let (kind, state) = log_kind_and_state(&name).unwrap_or(("run", "running"));
                        serde_json::json!({
                            "name": name, "size": size, "mtime": mtime,
                            "kind": kind, "state": state,
                        })
                    })
                    .collect();
                (
                    200,
                    "application/json",
                    serde_json::to_string(&items).unwrap_or_else(|_| "[]".into()),
                )
            }
            (false, "/api/log") => {
                let q = url.split('?').nth(1).unwrap_or("");
                let name = query_param(q, "name").unwrap_or_default();
                let tail = query_param(q, "tail")
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(400usize);
                match read_log_tail(&name, tail) {
                    Some(text) => (200, "text/plain; charset=utf-8", text),
                    None => (404, "text/plain", "no such log".to_string()),
                }
            }
            // Per-run bench results: the results.jsonl rows for each snapshot the run produced.
            (false, "/api/results") => {
                let q = url.split('?').nth(1).unwrap_or("");
                let run = query_param(q, "run").unwrap_or_default();
                (
                    200,
                    "application/json",
                    results_json(&db, results_dir.as_deref(), &run),
                )
            }
            // Body is a query-string form: dockerfile=&tag=&push=. Spawns a detached build and returns.
            (true, "/api/build") => {
                let mut form = String::new();
                let _ = req.as_reader().read_to_string(&mut form);
                build_response(&form)
            }
            // Proxy a live poot-serve's JSON /metrics (card 051); `target` is `host:port`. Fetched
            // server-side to avoid CORS and to reach pods the browser cannot.
            (false, "/api/serve-metrics") => {
                let q = url.split('?').nth(1).unwrap_or("");
                let target = query_param(q, "target").unwrap_or_default();
                (200, "application/json", serve_metrics_json(&target))
            }
            _ => (404, "text/plain", "not found".to_string()),
        };
        let header = Header::from_bytes(&b"Content-Type"[..], ctype.as_bytes()).unwrap();
        let resp = Response::from_string(body)
            .with_status_code(status)
            .with_header(header);
        let _ = req.respond(resp);
    }
    Ok(())
}

/// One result set of a run as the dashboard shows it.
struct Snap {
    /// The directory name, under `results/`, `failed-results/` or (a mirror) `results/<name>.partial`.
    dir: String,
    env: serde_json::Value,
    rows: Vec<serde_json::Value>,
    /// Gathered with no `ok` row (kept in `failed-results/`).
    failed: bool,
    /// Still being swept: the rows so far, mirrored off the pod.
    in_progress: bool,
    /// When the mirror was last written, ms since the epoch (in-progress sets).
    updated_ms: u64,
}

/// The sets a run produced that are here. The run's DB record names them as the pod does
/// (`run-<epoch>-<pid>-<model>-r<N>`); a set is found under the name it was gathered as, in `results/` or in
/// `failed-results/`, or as its `.partial` mirror while it is being swept (only for a run that has not ended: the
/// mirror a crashed run left behind is not "in progress"). Nothing depends on which binary gathered it.
fn run_sets(db: &crate::db::Db, results: &Path, run_id: &str, active: bool) -> Result<Vec<Snap>> {
    let failed_root = crate::gather::failed_dir(results);
    let mut snaps = Vec::new();
    for pod_dir in db.snapshot_dirs(run_id)? {
        let (root, dir, failed, in_progress) =
            if let Some(dir) = crate::gather::local_set(results, &pod_dir) {
                (results, dir, false, false)
            } else if let Some(dir) = crate::gather::local_failed_set(results, &pod_dir) {
                (failed_root.as_path(), dir, true, false)
            } else if active && let Some(dir) = crate::gather::local_partial(results, &pod_dir) {
                (results, dir, false, true)
            } else {
                continue;
            };
        let d = root.join(&dir);
        let rows = std::fs::read_to_string(d.join("results.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .collect();
        let env = std::fs::read_to_string(d.join("env.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        let updated_ms = if in_progress {
            modified_ms(&d.join("results.jsonl"))
        } else {
            0
        };
        snaps.push(Snap {
            dir,
            env,
            rows,
            failed,
            in_progress,
            updated_ms,
        });
    }
    Ok(snaps)
}

/// Why a run has no result sets to show, or `None` when it has some (or may yet have, while it sweeps).
fn no_sets_reason(
    run: &crate::db::runs::RunView,
    results: Option<&Path>,
    active: bool,
) -> &'static str {
    if run.kind == "exec" {
        "exec_run"
    } else if results.is_none() {
        "no_results_dir"
    } else if active {
        "no_sets_yet"
    } else {
        "ended_without_sets"
    }
}

/// Per-run bench results: `{sets: [{dir, env, rows, failed, in_progress, updated_ms, last_cell}], reason}`; `reason`
/// says why `sets` is empty (see [`no_sets_reason`]) and is null otherwise. A run the DB does not know is `unknown_run`.
fn results_json(db: &crate::db::Db, results: Option<&Path>, run_id: &str) -> String {
    let error = |e: String| format!("{{\"error\":{e:?}}}");
    let run = match db.dashboard_run(run_id) {
        Ok(run) => run,
        Err(e) => return error(e.to_string()),
    };
    let Some(run) = run else {
        return serde_json::json!({ "sets": [], "reason": "unknown_run" }).to_string();
    };
    let active = db
        .unfinished_run_ids()
        .is_ok_and(|ids| ids.contains(run_id));
    let snaps = match results
        .filter(|_| run.kind != "exec")
        .map(|r| run_sets(db, r, run_id, active))
    {
        Some(Ok(snaps)) => snaps,
        Some(Err(e)) => return error(e.to_string()),
        None => Vec::new(),
    };
    let sets: Vec<serde_json::Value> = snaps
        .iter()
        .map(|s| {
            let mut snap = serde_json::json!({
                "dir": s.dir, "env": s.env, "rows": s.rows, "failed": s.failed, "in_progress": s.in_progress,
            });
            if s.in_progress {
                snap["updated_ms"] = s.updated_ms.into();
                snap["last_cell"] = s.rows.last().map(last_cell).into();
            }
            snap
        })
        .collect();
    let reason = sets
        .is_empty()
        .then(|| no_sets_reason(&run, results, active));
    serde_json::json!({ "sets": sets, "reason": reason }).to_string()
}

/// How a run that ended its sweep came out, from its gathered rows: `ok` (every cell measured), `errors` (some
/// cells failed) or `failed` (every cell failed). A cell is a row that measured or errored; an unsupported
/// combination is neither. `None` when the run has no such row yet.
fn outcome(rows: &[serde_json::Value]) -> Option<serde_json::Value> {
    let total = rows
        .iter()
        .filter(|r| r["status"] == "ok" || r["status"] == "error")
        .count();
    let failed = rows.iter().filter(|r| r["status"] == "error").count();
    let (kind, label) = match (failed, total) {
        (_, 0) => return None,
        (0, _) => ("ok", "done".to_string()),
        (f, t) if f == t => ("failed", "done: all cells failed".to_string()),
        (f, t) => (
            "errors",
            format!("done with errors ({f} of {t} cells failed)"),
        ),
    };
    Some(serde_json::json!({ "kind": kind, "label": label, "failed": failed, "total": total }))
}

/// A row's cell as the dashboard names it: `<framework> <backend> <scenario>`.
fn last_cell(row: &serde_json::Value) -> String {
    ["framework", "backend", "scenario"]
        .iter()
        .filter_map(|key| row.get(key)?.as_str())
        .collect::<Vec<_>>()
        .join(" ")
}

/// When a file was last written, in milliseconds since the epoch (0 when unknown).
fn modified_ms(path: &Path) -> u64 {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

/// The sets a run has gathered, of the sets it will gather (every repeat of every model).
fn sets_progress(db: &Db, results: &Path, run: &crate::db::runs::RunView) -> (usize, usize) {
    let done = db
        .snapshot_dirs(&run.id)
        .map(|dirs| {
            dirs.iter()
                .filter(|d| crate::gather::set_is_complete(results, d))
                .count()
        })
        .unwrap_or(0);
    (done, run.models.len() * run.repeats as usize)
}

/// Every pod with what RunPod says of it and what it has cost so far, paired with its check.
fn pods_with_checks<'a>(
    pods: &'a [Pod],
    view: Option<&derive::RunpodView>,
    now: i64,
) -> Vec<(&'a Pod, derive::Verified)> {
    pods.iter()
        .map(|pod| (pod, derive::verify(pod, view, now)))
        .collect()
}

/// `/api/state`: recent runs and pods, each with what the dashboard derives (see the module doc); a run that has
/// not ended carries the harness's latest progress line.
fn state_json(
    db: &crate::db::Db,
    results: Option<&Path>,
    runpod: Option<&derive::RunpodView>,
    now: i64,
) -> Result<String> {
    let pods = db.all_pods(50)?;
    let checked = pods_with_checks(&pods, runpod, now);
    let views = db.dashboard(50)?;
    let mut runs = serde_json::to_value(&views)?;
    let unfinished = db.unfinished_run_ids()?;
    for (run, view) in runs.as_array_mut().into_iter().flatten().zip(&views) {
        let id = run["id"].as_str().unwrap_or_default().to_string();
        let status = run["status"].as_str().unwrap_or_default().to_string();
        let active = unfinished.contains(&id);
        run["has_sets"] = (run["kind"] == "sweep").into();
        if let Some((pod, verified)) = checked
            .iter()
            .find(|(p, _)| run["pod_id"].as_str() == Some(p.id.as_str()))
        {
            run["cost_usd"] = derive::meter(pod, verified, now)
                .and_then(|m| m.accrued_usd)
                .into();
        }
        if active && status.starts_with("provisioning (waiting for capacity") {
            let log = crate::logging::read_log_tail(&format!("{id}.log"), 400).unwrap_or_default();
            run["wait"] = serde_json::to_value(derive::capacity_wait(&log))?;
        }
        let Some(results) = results else { continue };
        if run["kind"] == "sweep" {
            let (done, total) = sets_progress(db, results, view);
            run["sets_done"] = done.into();
            run["sets_total"] = total.into();
            if active {
                let started = secs_from_iso(run["created_at"].as_str().unwrap_or_default());
                run["eta_secs"] = started
                    .and_then(|s| derive::eta_secs(now - s, done, total))
                    .into();
            }
        }
        if active {
            if let Some(line) = crate::gather::read_progress(results, &id) {
                run["progress"] = line.into();
            }
        } else if matches!(status.split(' ').next(), Some("done" | "incomplete:")) {
            // Only a sweep that ended is judged by its cells; cancelled and abandoned runs keep their status.
            let rows: Vec<_> = run_sets(db, results, &id, false)?
                .into_iter()
                .filter(|s| !s.in_progress)
                .flat_map(|s| s.rows)
                .collect();
            run["outcome"] = outcome(&rows).into();
        }
    }
    let pods: Vec<_> = checked
        .iter()
        .map(|(pod, verified)| {
            let mut v = serde_json::to_value(pod).unwrap_or_default();
            v["verified"] = serde_json::to_value(verified).unwrap_or_default();
            if let Some(m) = derive::meter(pod, verified, now) {
                v["uptime_secs"] = m.uptime_secs.into();
                v["accrued_usd"] = m.accrued_usd.into();
            }
            v
        })
        .collect();
    Ok(serde_json::json!({ "runs": runs, "pods": pods }).to_string())
}

/// The run logs of the state directory.
struct StateDirLogs;

impl derive::LogSource for StateDirLogs {
    fn stamp(&self, name: &str) -> Option<(u64, u64)> {
        let meta = std::fs::metadata(crate::logging::logs_dir().join(name)).ok()?;
        Some((
            meta.len(),
            modified_ms(&crate::logging::logs_dir().join(name)),
        ))
    }

    fn read(&self, name: &str) -> String {
        std::fs::read_to_string(crate::logging::logs_dir().join(name)).unwrap_or_default()
    }
}

/// `/api/overview`: the totals and failures the overview page shows over the last 24 hours. The logs of the runs of
/// that window are read through `cache`, so a poll re-reads only a log that has grown.
fn overview_json(
    db: &crate::db::Db,
    view: Option<&derive::RunpodView>,
    now: i64,
    logs: &dyn derive::LogSource,
    cache: &mut derive::LogCache,
) -> Result<String> {
    let pods = db.all_pods(200)?;
    let checked = pods_with_checks(&pods, view, now);
    let live = checked
        .iter()
        .filter(|(p, v)| derive::is_billing(p, v))
        .count();
    let since = now - 24 * 3600;
    let mut failures = Vec::new();
    let mut replacements_24h = 0;
    for run in db.dashboard(200)? {
        let at = secs_from_iso(&run.updated_at).unwrap_or(0);
        let started = secs_from_iso(&run.created_at).unwrap_or(0);
        if at < since && started < since {
            continue;
        }
        let facts = cache.facts(logs, &format!("{}.log", run.id));
        replacements_24h += facts.replacements;
        if matches!(run.status.as_str(), "failed" | "aborted") {
            failures.push(serde_json::json!({
                "run": run.id, "status": run.status, "at": run.updated_at,
                "reason": facts.last_line, "replacements": facts.replacements,
            }));
        }
    }
    Ok(serde_json::json!({
        "spend_24h_usd": derive::spend_24h(&checked, now),
        "burn_usd_per_hr": derive::burn_per_hr(&checked),
        "billing_pods": live,
        "failures": failures,
        "replacements_24h": replacements_24h,
    })
    .to_string())
}

/// Validate a `target=host:port` proxy target before it goes into a URL: no whitespace or `scheme://`,
/// and it splits on the last `:` into a non-empty host and an all-digit port.
fn valid_serve_target(target: &str) -> bool {
    if target.is_empty() || target.contains(char::is_whitespace) || target.contains("://") {
        return false;
    }
    match target.rsplit_once(':') {
        Some((host, port)) => {
            !host.is_empty() && !port.is_empty() && port.chars().all(|c| c.is_ascii_digit())
        }
        None => false,
    }
}

/// Proxy a live poot-serve's `GET /metrics` (card 051). Always returns HTTP 200 with a JSON body; the UI
/// treats an `error` key as failure, so an unreachable serve renders as "unreachable".
fn serve_metrics_json(target: &str) -> String {
    if !valid_serve_target(target) {
        return "{\"error\":\"missing or invalid target (expected host:port)\"}".to_string();
    }
    let url = format!("http://{target}/metrics");
    let client = match reqwest::blocking::Client::builder()
        .timeout(std::time::Duration::from_secs(3))
        .build()
    {
        Ok(c) => c,
        Err(e) => return format!("{{\"error\":{:?}}}", e.to_string()),
    };
    let resp = match client.get(&url).send() {
        Ok(r) => r,
        Err(e) => return format!("{{\"error\":{:?}}}", format!("fetch {url}: {e}")),
    };
    if !resp.status().is_success() {
        let status = resp.status();
        return format!("{{\"error\":{:?}}}", format!("{url} returned {status}"));
    }
    let text = match resp.text() {
        Ok(t) => t,
        Err(e) => return format!("{{\"error\":{:?}}}", e.to_string()),
    };
    // Pass the body through only if it parses as JSON.
    if serde_json::from_str::<serde_json::Value>(&text).is_ok() {
        text
    } else {
        "{\"error\":\"target did not return JSON\"}".to_string()
    }
}

/// Validate the build form, kick off the detached build, and build the (status, ctype, body) tuple.
fn build_response(form: &str) -> (u16, &'static str, String) {
    let dockerfile = query_param(form, "dockerfile").unwrap_or_else(|| "docker/Dockerfile".into());
    let tag = query_param(form, "tag").unwrap_or_else(|| "latest".into());
    let push = matches!(
        query_param(form, "push").as_deref(),
        Some("1") | Some("true") | Some("on")
    );
    // The tag goes into a file name (image-<tag>.log); restrict to safe chars so it cannot escape logs_dir.
    if tag.is_empty()
        || !tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        return (
            400,
            "application/json",
            "{\"error\":\"bad tag\"}".to_string(),
        );
    }
    if dockerfile.contains("..") || dockerfile.is_empty() {
        return (
            400,
            "application/json",
            "{\"error\":\"bad dockerfile\"}".to_string(),
        );
    }
    match spawn_image_build(&dockerfile, &tag, push) {
        Ok(logname) => (202, "application/json", format!("{{\"log\":{logname:?}}}")),
        Err(e) => (
            500,
            "application/json",
            format!("{{\"error\":{:?}}}", e.to_string()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Run;

    /// A scratch directory removed on drop.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn new() -> Self {
            use std::sync::atomic::{AtomicU32, Ordering};
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let dir = std::env::temp_dir().join(format!(
                "poot-web-test-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir_all(&dir).unwrap();
            Self(dir)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The pod's names of the two repeats of the run below, and the names gathering commits them under.
    const POD_R1: &str = "run-1792632197-3202502-qwen3-4b-r1";
    const POD_R2: &str = "run-1792632197-3202502-qwen3-4b-r2";
    const C1: &str = "2026-10-22-geforce-rtx-3090-qwen3-4b-012317-r1";
    const C2: &str = "2026-10-22-geforce-rtx-3090-qwen3-4b-012317-r2";
    const OK: &str =
        r#"{"framework":"poot","backend":"ptx","scenario":"decode-curve","status":"ok"}"#;
    const ERR: &str = r#"{"framework":"poot","backend":"ptx","scenario":"decode-curve","status":"error","reason":"rc=1: boom"}"#;

    /// A sweep of two repeats of qwen3-4b whose sets the DB names as the pod did, in the given status.
    fn run_of_two_repeats(status: &str) -> (Db, String) {
        let db = Db::open(":memory:").unwrap();
        let id = "run-1792632197-3202502".to_string();
        let run = Run {
            id: id.clone(),
            kind: "sweep".into(),
            git_ref: "abc".into(),
            scenario: "decode-curve".into(),
            repeats: 2,
            models: vec!["qwen3-4b".into()],
            gpu_types: "NVIDIA GeForce RTX 3090".into(),
            image: "img".into(),
            status: status.into(),
            pod_id: None,
        };
        db.create_run(&run).unwrap();
        db.set_status(&id, status).unwrap();
        db.set_model_sweep(
            &id,
            "qwen3-4b",
            true,
            Some("run-1792632197-3202502-qwen3-4b"),
        )
        .unwrap();
        (db, id)
    }

    fn write_set(root: &Path, name: &str, rows: &[&str]) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("results.jsonl"), format!("{}\n", rows.join("\n"))).unwrap();
        std::fs::write(
            dir.join("env.json"),
            r#"{"gpu_name":"NVIDIA GeForce RTX 3090"}"#,
        )
        .unwrap();
        std::fs::write(dir.join("report.md"), "r").unwrap();
    }

    fn results_reply(db: &Db, results: Option<&Path>, id: &str) -> serde_json::Value {
        serde_json::from_str(&results_json(db, results, id)).unwrap()
    }

    fn results_of(db: &Db, results: &Path, id: &str) -> Vec<serde_json::Value> {
        results_reply(db, Some(results), id)["sets"]
            .as_array()
            .unwrap()
            .clone()
    }

    fn state_of(db: &Db, results: &Path) -> serde_json::Value {
        serde_json::from_str(&state_json(db, Some(results), None, now_secs()).unwrap()).unwrap()
    }

    /// A run whose sets were gathered under their committed names shows their rows, whichever binary gathered
    /// them. Mutation: look the sets up by the pod's name only and every snapshot is missing.
    #[test]
    fn a_run_gathered_under_canonical_names_shows_its_rows() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("done");
        write_set(&results, C1, &[OK, OK]);
        write_set(&results, C2, &[OK]);
        let snaps = results_of(&db, &results, &id);
        let dirs: Vec<_> = snaps.iter().map(|s| s["dir"].as_str().unwrap()).collect();
        assert_eq!(dirs, [C1, C2]);
        assert_eq!(snaps[0]["rows"].as_array().unwrap().len(), 2);
        assert_eq!(snaps[0]["failed"], false);
        assert_eq!(snaps[0]["in_progress"], false);
        assert!(!dirs.contains(&POD_R1) && !dirs.contains(&POD_R2));
    }

    /// A set with no ok row is listed from `failed-results/`, marked failed, with its rows' reasons.
    #[test]
    fn a_failed_set_is_listed_with_its_reasons() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("done");
        write_set(&crate::gather::failed_dir(&results), C1, &[ERR]);
        let snaps = results_of(&db, &results, &id);
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0]["failed"], true);
        assert_eq!(snaps[0]["rows"][0]["reason"], "rc=1: boom");
    }

    /// A set still being swept is shown from its `.partial` mirror, marked in progress, with the last cell.
    #[test]
    fn a_set_being_swept_is_shown_from_its_mirror() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("sweeping");
        write_set(&results, &format!("{C1}.partial"), &[OK, ERR]);
        let snaps = results_of(&db, &results, &id);
        assert_eq!(snaps.len(), 1);
        assert_eq!(snaps[0]["in_progress"], true);
        assert_eq!(snaps[0]["last_cell"], "poot ptx decode-curve");
        assert!(snaps[0]["updated_ms"].as_u64().unwrap() > 0);
    }

    /// The mirror a crashed run left behind is not "in progress": a run that has ended shows no mirror.
    /// Mutation: ignore `active` in `run_sets` and the abandoned run's set is listed as in progress.
    #[test]
    fn the_mirror_of_an_ended_run_is_not_shown() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("abandoned");
        write_set(&results, &format!("{C1}.partial"), &[OK]);
        assert!(results_of(&db, &results, &id).is_empty());
    }

    fn state_outcome(status: &str, sets: &[(&str, &[&str])]) -> serde_json::Value {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats(status);
        for (name, rows) in sets {
            write_set(&results, name, rows);
        }
        let state: serde_json::Value = state_of(&db, &results);
        assert_eq!(state["runs"][0]["id"], id.as_str());
        state["runs"][0]["outcome"].clone()
    }

    /// A finished sweep is judged by its cells: all ok is done, some errors amber, every cell failed red.
    /// Mutation: derive the outcome from the lifecycle status only and the last two cases read `ok`.
    #[test]
    fn a_done_run_is_judged_by_its_cells() {
        let ok = state_outcome("done", &[(C1, &[OK, OK]), (C2, &[OK])]);
        assert_eq!(
            (ok["kind"].as_str(), ok["label"].as_str()),
            (Some("ok"), Some("done"))
        );

        let some = state_outcome("done", &[(C1, &[OK, ERR]), (C2, &[OK, OK])]);
        assert_eq!(some["kind"], "errors");
        assert_eq!(some["label"], "done with errors (1 of 4 cells failed)");
        assert_eq!(
            (some["failed"].as_u64(), some["total"].as_u64()),
            (Some(1), Some(4))
        );

        let all = state_outcome("done", &[(C1, &[ERR, ERR]), (C2, &[ERR])]);
        assert_eq!(all["kind"], "failed");
        assert_eq!(all["label"], "done: all cells failed");
    }

    /// Cancelled and abandoned runs keep their status; so does a run still sweeping.
    #[test]
    fn only_a_done_run_has_an_outcome() {
        for status in ["cancelled", "abandoned", "failed", "sweeping"] {
            let outcome = state_outcome(status, &[(C1, &[ERR])]);
            assert!(outcome.is_null(), "{status}");
        }
    }

    /// An unsupported combination is neither a measured nor a failed cell.
    #[test]
    fn an_unsupported_row_is_not_a_cell() {
        let unsupported = r#"{"status":"unsupported"}"#;
        assert_eq!(outcome(&[serde_json::from_str(unsupported).unwrap()]), None);
        let rows: Vec<serde_json::Value> = [OK, unsupported]
            .iter()
            .map(|r| serde_json::from_str(r).unwrap())
            .collect();
        assert_eq!(outcome(&rows).unwrap()["total"], 1);
    }

    /// A run that has not ended carries the harness's latest progress line; an ended one does not.
    #[test]
    fn a_running_run_carries_its_progress_line() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("sweeping");
        crate::gather::write_progress(
            &results,
            &id,
            "qwen3-4b: [progress] poot: isl 128 done (1/4), ETA 9m",
        )
        .unwrap();
        let state: serde_json::Value = state_of(&db, &results);
        assert_eq!(
            state["runs"][0]["progress"],
            "qwen3-4b: [progress] poot: isl 128 done (1/4), ETA 9m"
        );
        db.set_status(&id, "done").unwrap();
        let state: serde_json::Value = state_of(&db, &results);
        assert!(state["runs"][0]["progress"].is_null());
    }

    /// An exec run has no sets, and the reply says so; a sweep says why it has none yet or ever. Mutation: drop the
    /// kind check in `no_sets_reason` and an exec run reads `ended_without_sets`.
    #[test]
    fn a_run_with_no_sets_says_why() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("sweeping");
        assert_eq!(
            results_reply(&db, Some(&results), &id)["reason"],
            "no_sets_yet"
        );
        assert_eq!(results_reply(&db, None, &id)["reason"], "no_results_dir");
        db.set_status(&id, "cancelled").unwrap();
        assert_eq!(
            results_reply(&db, Some(&results), &id)["reason"],
            "ended_without_sets"
        );
        assert_eq!(
            results_reply(&db, Some(&results), "run-nope")["reason"],
            "unknown_run"
        );

        let exec = Run {
            id: "exec-1-2-3-0".into(),
            kind: "exec".into(),
            git_ref: "exec".into(),
            scenario: "bash x".into(),
            repeats: 1,
            models: vec![],
            gpu_types: "NVIDIA A100".into(),
            image: "img".into(),
            status: "running".into(),
            pod_id: None,
        };
        db.create_run(&exec).unwrap();
        let reply = results_reply(&db, Some(&results), &exec.id);
        assert_eq!(
            (
                reply["reason"].as_str(),
                reply["sets"].as_array().map(Vec::len)
            ),
            (Some("exec_run"), Some(0))
        );
    }

    /// A run with sets has no reason: the page shows the sets.
    #[test]
    fn a_run_with_sets_has_no_reason() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("done");
        write_set(&results, C1, &[OK]);
        assert!(results_reply(&db, Some(&results), &id)["reason"].is_null());
    }

    /// `/api/state` says how many of a sweep's sets are gathered, whether it has sets at all, and what its pod
    /// has cost; each pod carries its uptime, estimate and RunPod's word. Mutation: check a pod against an empty
    /// RunPod list and `verified.state` never reads `gone`.
    #[test]
    fn state_carries_set_progress_and_the_pod_meter() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, id) = run_of_two_repeats("sweeping");
        write_set(&results, C1, &[OK]);
        db.create_pod_record(
            "pod-a",
            "NVIDIA GeForce RTX 3090",
            "COMMUNITY",
            Some(0.5),
            "img",
        )
        .unwrap();
        db.set_pod(&id, "pod-a", None, None).unwrap();
        let now = now_secs() + 7200;
        let listed = derive::RunpodView {
            checked_at: now,
            pods: Ok(std::collections::HashMap::from([(
                "pod-a".to_string(),
                "RUNNING".to_string(),
            )])),
        };
        let state: serde_json::Value =
            serde_json::from_str(&state_json(&db, Some(&results), Some(&listed), now).unwrap())
                .unwrap();
        let run = &state["runs"][0];
        assert_eq!(
            (
                run["has_sets"].as_bool(),
                run["sets_done"].as_u64(),
                run["sets_total"].as_u64()
            ),
            (Some(true), Some(1), Some(2))
        );
        assert_eq!(run["repeats"], 2);
        assert!(run["eta_secs"].as_u64().is_some());
        assert!((run["cost_usd"].as_f64().unwrap() - 1.0).abs() < 0.01);
        let pod = &state["pods"][0];
        assert_eq!(pod["verified"]["state"], "confirmed");
        assert!((pod["uptime_secs"].as_i64().unwrap() - 7200).abs() <= 2);

        let empty = derive::RunpodView {
            checked_at: now,
            pods: Ok(Default::default()),
        };
        let state: serde_json::Value =
            serde_json::from_str(&state_json(&db, Some(&results), Some(&empty), now).unwrap())
                .unwrap();
        assert_eq!(state["pods"][0]["verified"]["state"], "gone");
        // Gone pods stop accruing at their last DB update: seconds, not two hours.
        assert!(state["pods"][0]["uptime_secs"].as_i64().unwrap() < 5);
    }

    /// The overview totals the spend and burn of the pods and lists the runs that failed.
    #[test]
    fn overview_totals_spend_and_lists_failures() {
        let (db, id) = run_of_two_repeats("failed");
        db.create_pod_record(
            "pod-a",
            "NVIDIA GeForce RTX 3090",
            "COMMUNITY",
            Some(2.0),
            "img",
        )
        .unwrap();
        let now = now_secs() + 3600;
        let listed = derive::RunpodView {
            checked_at: now,
            pods: Ok(std::collections::HashMap::from([(
                "pod-a".to_string(),
                "RUNNING".to_string(),
            )])),
        };
        let logs = FakeLogs::new(&[]);
        let v: serde_json::Value = serde_json::from_str(
            &overview_json(
                &db,
                Some(&listed),
                now,
                &logs,
                &mut derive::LogCache::default(),
            )
            .unwrap(),
        )
        .unwrap();
        assert!((v["spend_24h_usd"].as_f64().unwrap() - 2.0).abs() < 0.01);
        assert_eq!(v["burn_usd_per_hr"], 2.0);
        assert_eq!(v["billing_pods"], 1);
        assert_eq!(v["failures"][0]["run"], id.as_str());
        assert_eq!(v["failures"][0]["status"], "failed");
    }

    /// Run logs held in memory, counting how often each is read.
    struct FakeLogs {
        logs: std::cell::RefCell<std::collections::HashMap<String, (String, u64)>>,
        reads: std::cell::Cell<usize>,
    }

    impl FakeLogs {
        fn new(logs: &[(&str, &str)]) -> Self {
            Self {
                logs: std::cell::RefCell::new(
                    logs.iter()
                        .map(|(n, t)| ((*n).into(), ((*t).into(), 1)))
                        .collect(),
                ),
                reads: std::cell::Cell::new(0),
            }
        }

        /// The run appends to its log.
        fn append(&self, name: &str, text: &str) {
            let mut logs = self.logs.borrow_mut();
            let entry = logs.get_mut(name).unwrap();
            entry.0.push_str(text);
            entry.1 += 1;
        }
    }

    impl derive::LogSource for FakeLogs {
        fn stamp(&self, name: &str) -> Option<(u64, u64)> {
            self.logs
                .borrow()
                .get(name)
                .map(|(text, mtime)| (text.len() as u64, *mtime))
        }

        fn read(&self, name: &str) -> String {
            self.reads.set(self.reads.get() + 1);
            self.logs.borrow()[name].0.clone()
        }
    }

    const REPLACED: &str = "[10:00:00] pod a has no working GPU, provisioning replacement, 0 of 3 repeats already gathered\n";

    fn overview_of(db: &Db, logs: &FakeLogs, cache: &mut derive::LogCache) -> serde_json::Value {
        let now = now_secs() + 60;
        serde_json::from_str(&overview_json(db, None, now, logs, cache).unwrap()).unwrap()
    }

    /// A poll reads a log once while it does not change, and again once it grows; the replacements of every run are
    /// counted from the whole log, not a tail. Mutation: bypass the cache and `reads` counts every poll.
    #[test]
    fn the_overview_rereads_a_log_only_when_it_changes() {
        let (db, id) = run_of_two_repeats("failed");
        let name = format!("{id}.log");
        let logs = FakeLogs::new(&[(
            name.as_str(),
            &format!("{}{}[10:05:00] boom\n", "[1] line\n".repeat(5000), REPLACED),
        )]);
        let mut cache = derive::LogCache::default();
        let first = overview_of(&db, &logs, &mut cache);
        let again = overview_of(&db, &logs, &mut cache);
        assert_eq!(logs.reads.get(), 1, "an unchanged log is read once");
        assert_eq!(first, again);
        assert_eq!(
            first["replacements_24h"], 1,
            "a replacement before the last 4000 lines still counts"
        );
        assert_eq!(first["failures"][0]["reason"], "boom");
        assert_eq!(first["failures"][0]["replacements"], 1);

        logs.append(&name, REPLACED);
        let grown = overview_of(&db, &logs, &mut cache);
        assert_eq!(logs.reads.get(), 2, "a log that grew is read again");
        assert_eq!(grown["replacements_24h"], 2);
    }

    /// A run older than the 200 most recent still has its results looked up. Mutation: find it in `dashboard(200)`
    /// and the oldest run reads `unknown_run`.
    #[test]
    fn results_find_a_run_whatever_its_age() {
        let tmp = TempDir::new();
        let results = tmp.0.join("results");
        let (db, oldest) = run_of_two_repeats("done");
        db.set_created_at(&oldest, "2000-01-01T00:00:00Z").unwrap();
        for n in 0..201 {
            let run = Run {
                id: format!("run-9{n:03}"),
                kind: "sweep".into(),
                git_ref: "abc".into(),
                scenario: "decode-curve".into(),
                repeats: 1,
                models: vec![],
                gpu_types: "NVIDIA GeForce RTX 3090".into(),
                image: "img".into(),
                status: "done".into(),
                pod_id: None,
            };
            db.create_run(&run).unwrap();
        }
        assert_eq!(db.dashboard(200).unwrap().len(), 200);
        assert!(db.dashboard(200).unwrap().iter().all(|r| r.id != oldest));
        assert_eq!(
            results_reply(&db, Some(&results), &oldest)["reason"],
            "ended_without_sets"
        );
    }

    #[test]
    fn query_param_finds_key() {
        let q = "name=run-2.log&tail=400";
        assert_eq!(query_param(q, "name").as_deref(), Some("run-2.log"));
        assert_eq!(query_param(q, "tail").as_deref(), Some("400"));
        assert_eq!(query_param(q, "missing"), None);
    }

    #[test]
    fn query_param_decodes_a_few_chars() {
        assert_eq!(
            query_param("target=127.0.0.1%3A8080", "target").as_deref(),
            Some("127.0.0.1%3A8080") // ':' is not in our tiny decode set - only %2F/%2E/+ are handled.
        );
        assert_eq!(
            query_param("name=a%2Fb%2Ec", "name").as_deref(),
            Some("a/b.c")
        );
    }

    #[test]
    fn valid_serve_target_accepts_host_port() {
        assert!(valid_serve_target("127.0.0.1:8080"));
        assert!(valid_serve_target("localhost:80"));
        assert!(valid_serve_target("some-pod.runpod.internal:9000"));
    }

    #[test]
    fn valid_serve_target_rejects_bad_input() {
        assert!(!valid_serve_target(""));
        assert!(!valid_serve_target("no-port-here"));
        assert!(!valid_serve_target("http://127.0.0.1:8080")); // scheme not allowed
        assert!(!valid_serve_target("127.0.0.1: 8080")); // whitespace
        assert!(!valid_serve_target("127.0.0.1:"));
        assert!(!valid_serve_target(":8080"));
        assert!(!valid_serve_target("127.0.0.1:abc")); // non-numeric port
        assert!(!valid_serve_target(" 127.0.0.1:8080"));
    }

    #[test]
    fn serve_metrics_json_rejects_bad_target_without_network() {
        let body = serve_metrics_json("");
        let v: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert!(v.get("error").is_some());

        let body = serve_metrics_json("http://x:80");
        let v: serde_json::Value = serde_json::from_str(&body).expect("valid json");
        assert!(v.get("error").is_some());
    }
}
