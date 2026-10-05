//! Read-only web dashboard over the run-state DB. `serve` binds a blocking tiny_http server and exposes:
//!   GET  /            the dashboard page (the committed Vite build of `ui/`, inlined into dashboard.html)
//!   GET  /api/state   JSON: recent runs + per-model progress, what the page polls every few seconds
//!   GET  /api/logfiles  JSON: log files (name, size, mtime, kind, state), newest first
//!   GET  /api/log     text/plain: the tail of one log, by name (path-traversal guarded)
//!   POST /api/build   trigger an image build off-thread (logs to image-<tag>.log), returns 202
//!   GET  /api/serve-metrics  JSON: proxy a live poot-serve's `GET /metrics` (card 051)
//!   GET  /healthz     "ok"
//! It only reads the SQLite DB, so it is safe alongside an active `run` (WAL allows concurrent readers).
//!
//! The build endpoint runs an external process, so serve binds 127.0.0.1 by default. Do not bind a
//! public address with it enabled.

use anyhow::{Result, anyhow};
use tiny_http::{Header, Method, Response, Server};

use crate::db::Db;
use crate::logging::{list_logs, log, log_kind_and_state, read_log_tail, spawn_image_build};
use crate::workflow::repo_root;

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

pub fn serve(db: Db, addr: &str) -> Result<()> {
    // Single-threaded request loop, so the non-Send Connection in `db` is fine. Only the image build
    // (POST /api/build) runs detached.
    let server = Server::http(addr).map_err(|e| anyhow!("bind {addr}: {e}"))?;
    log(format!("dashboard on http://{addr}  (Ctrl-C to stop)"));

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
                let built = (|| -> anyhow::Result<String> {
                    let runs = db.dashboard(50)?;
                    let pods = db.all_pods(50)?;
                    Ok(serde_json::json!({ "runs": runs, "pods": pods }).to_string())
                })();
                match built {
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
                (200, "application/json", results_json(&db, &run))
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

/// Per-run bench results: for each snapshot dir the run produced (`pod_run_id`), read `results.jsonl`
/// rows and `env.json` from `benchmarks/results/`. Returns `[{dir, env, rows:[...]}]` as JSON.
fn results_json(db: &crate::db::Db, run_id: &str) -> String {
    let dirs = match db.snapshot_dirs(run_id) {
        Ok(d) => d,
        Err(e) => return format!("{{\"error\":{:?}}}", e.to_string()),
    };
    let base = match repo_root() {
        Ok(root) => root.join("benchmarks").join("results"),
        Err(e) => return format!("{{\"error\":{:?}}}", e.to_string()),
    };
    let mut snaps = Vec::new();
    for dir in dirs {
        // Guard against path escape before touching the fs.
        if dir.contains('/') || dir.contains("..") {
            continue;
        }
        let d = base.join(&dir);
        let rows: Vec<serde_json::Value> = std::fs::read_to_string(d.join("results.jsonl"))
            .unwrap_or_default()
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
            .collect();
        let env: serde_json::Value = std::fs::read_to_string(d.join("env.json"))
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or(serde_json::Value::Null);
        snaps.push(serde_json::json!({ "dir": dir, "env": env, "rows": rows }));
    }
    serde_json::to_string(&snaps).unwrap_or_else(|_| "[]".into())
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
