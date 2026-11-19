//! Minimal blocking Runpod REST v2 client (`https://api.runpod.io/v2`, Bearer auth): create / get / list / delete a
//! pod and read the GPU price catalog.
//!
//! Runpod retired REST v1 (`rest.runpod.io`) on 2026-11-15; the host no longer resolves. v2 is the documented
//! surface the hosted Runpod MCP server drives, and its price catalog replaces the GraphQL one, so this module
//! speaks one API. Every wire shape (request body, pod, list page, catalog, problem document) is declared and
//! parsed here and nowhere else.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

use crate::db::PodOwnerPrefix;

const BASE: &str = "https://api.runpod.io/v2";

pub(crate) struct RunPod {
    client: reqwest::blocking::Client,
    key: String,
    /// REST v2 origin including the `/v2` prefix. Production always uses [`BASE`]; host tests may point at a
    /// loopback fixture with [`RunPod::with_base`] to observe real request paths.
    base: String,
}

/// A pod's lifecycle state, as Runpod's `status` reports the actual state (not a desired one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub(crate) enum PodStatus {
    /// Being allocated.
    Provisioning,
    /// The container is starting (image pull included).
    Starting,
    /// The container is healthy.
    Running,
    /// The container exited: the pod is stopped.
    Exited,
    /// The container is in an unrecoverable error state; waiting never fixes it.
    Error,
    /// Permanently deleted.
    Terminated,
    /// A status this client does not know yet.
    #[serde(other)]
    Unknown,
}

impl std::fmt::Display for PodStatus {
    /// The status as Runpod spells it on the wire.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            PodStatus::Provisioning => "PROVISIONING",
            PodStatus::Starting => "STARTING",
            PodStatus::Running => "RUNNING",
            PodStatus::Exited => "EXITED",
            PodStatus::Error => "ERROR",
            PodStatus::Terminated => "TERMINATED",
            PodStatus::Unknown => "UNKNOWN",
        })
    }
}

/// One way into a pod over ssh.
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct SshEndpoint {
    pub host: String,
    pub port: u16,
}

/// How to ssh into a pod. Only `direct` (the pod's own sshd over its published `22/tcp`) carries scp and port
/// forwarding; Runpod's proxy is an interactive shell only, so it is not read.
#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct PodSsh {
    /// `None` until the pod is `RUNNING` with a public port for `22/tcp`, and always `None` in a list
    /// response: read the pod by id for the live mapping.
    #[serde(default)]
    pub direct: Option<SshEndpoint>,
}

/// A pod as the v2 API returns it, from a create, a read by id or a list.
#[derive(Debug, Deserialize)]
pub(crate) struct Pod {
    pub id: String,
    #[serde(default)]
    pub name: String,
    pub status: PodStatus,
    #[serde(default)]
    pub ssh: PodSsh,
    /// The CUDA version the pod's host reports; `None` while unknown.
    #[serde(rename = "cudaVersion", default)]
    pub cuda_version: Option<String>,
    /// What the pod bills per hour (USD). Present on every response, including a create.
    #[serde(rename = "cost", default)]
    pub cost_per_hr: Option<f64>,
}

impl Pod {
    /// The CUDA version the pod reports. `None` while the API has not assigned a host yet (or does not report
    /// one), which is not "below any floor".
    pub(crate) fn reported_cuda(&self) -> Option<String> {
        self.cuda_version.clone().filter(|v| !v.trim().is_empty())
    }

    /// (host, port) of the direct ssh endpoint, or None until the pod is up.
    pub(crate) fn ssh_endpoint(&self) -> Option<(String, u16)> {
        let direct = self.ssh.direct.as_ref()?;
        if direct.host.is_empty() {
            return None;
        }
        Some((direct.host.clone(), direct.port))
    }
}

/// Runpod refused a create call with this HTTP status and problem text.
#[derive(Debug, thiserror::Error)]
#[error("create_pod failed ({status}): {body}")]
pub(crate) struct CreateError {
    pub(crate) status: u16,
    pub(crate) body: String,
}

impl CreateError {
    /// Whether waiting could fix this refusal: a server error, a rate limit, or the 400 Runpod answers when no
    /// machine matches (a 400 carries no machine-readable code, only that text). Any other 4xx (a rule
    /// violation, an unknown GPU type, auth, balance) never fixes itself.
    pub(crate) fn is_capacity(&self) -> bool {
        const NO_CAPACITY: [&str; 2] = [
            "instances available with the requested specifications",
            "does not have the resources to deploy your pod",
        ];
        let body = self.body.to_lowercase();
        self.status >= 500
            || self.status == 429
            || NO_CAPACITY.iter().any(|message| body.contains(message))
    }
}

/// Runpod answered 404: the pod does not exist (any other `get_pod` failure says nothing about the pod).
#[derive(Debug, thiserror::Error)]
#[error("pod {0} does not exist (404)")]
pub(crate) struct PodNotFound(pub String);

/// The RunPod cloud one create request rents from. The API accepts exactly these two values for `cloud`; any
/// other string is a 422.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CloudType {
    /// Runpod-operated datacenters.
    Secure,
    /// Third-party hosts.
    Community,
}

impl std::fmt::Display for CloudType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            CloudType::Secure => "SECURE",
            CloudType::Community => "COMMUNITY",
        })
    }
}

/// Fields for a create request.
pub struct CreateSpec<'a> {
    /// The state database's pod prefix. The pod's cloud name is built from it, so `reap` finds every pod
    /// this create call makes.
    pub owner: &'a PodOwnerPrefix,
    /// The orchestrator run this pod serves; part of the pod's cloud name.
    pub run_id: &'a str,
    /// The provisioning attempt of `run_id` this pod is.
    pub attempt: u32,
    /// The container image the pod boots.
    pub image: &'a str,
    /// The Runpod GPU type id to rent. One type per request: the caller retries across types.
    pub gpu_type_id: &'a str,
    /// Which Runpod cloud to rent from.
    pub cloud_type: CloudType,
    /// How many GPUs the pod gets.
    pub gpu_count: u32,
    /// Size of the pod's container disk, in GB.
    pub container_disk_gb: u32,
    /// Ports the pod exposes, in Runpod's `port/proto` form, e.g. `22/tcp`. v2 has no default.
    pub ports: &'a [&'a str],
    /// Environment variables set in the container.
    pub env: &'a HashMap<String, String>,
    /// Network volume to attach at /workspace (the model-weight cache). Pins the pod to the volume's
    /// datacenter, so `data_center_ids` should match it.
    pub network_volume_id: Option<&'a str>,
    /// Data centers to pin the pod to (`dataCenterIds`). Empty means no pin (field omitted).
    pub data_center_ids: &'a [&'a str],
}

/// ONE table of GPU types this orchestrator provisions: (underscore spelling a `--gpu-types` token may use,
/// the catalog id the v2 `gpu.id` field takes). Lookup accepts either spelling. Tokens not in the table pass
/// through unchanged (already a catalog id, or a type this table has not listed yet).
const GPU_TYPE_NAMES: &[(&str, &str)] = &[
    ("AMD_Instinct_MI300X_OAM", "AMD Instinct MI300X OAM"),
    ("NVIDIA_L40S", "NVIDIA L40S"),
    ("NVIDIA_A100_80GB_PCIe", "NVIDIA A100 80GB PCIe"),
    ("NVIDIA_RTX_A5000", "NVIDIA RTX A5000"),
    ("NVIDIA_GeForce_RTX_4090", "NVIDIA GeForce RTX 4090"),
    ("NVIDIA_GeForce_RTX_3090", "NVIDIA GeForce RTX 3090"),
    ("NVIDIA_H100_80GB_HBM3", "NVIDIA H100 80GB HBM3"),
    ("NVIDIA_A40", "NVIDIA A40"),
];

/// Normalize a `--gpu-types` token to the catalog id `gpu.id` takes.
/// Accepts the underscore spelling or the catalog id; unknown tokens pass through unchanged.
pub(crate) fn normalize_gpu_type_id(token: &str) -> &str {
    GPU_TYPE_NAMES
        .iter()
        .find(|(underscored, id)| *underscored == token || *id == token)
        .map(|(_, id)| *id)
        .unwrap_or(token)
}

/// Build the `POST /v2/pods` JSON body for [`RunPod::create_pod`].
///
/// v2 nests the compute (`gpu: {id, count}`, one GPU type only), takes `env` as a JSON object, expresses
/// storage only through `mounts` (no `mounts` means no persistent storage) and rejects unknown fields with a 422.
/// Separate function so tests can assert the wire shape without a live API.
pub(crate) fn create_pod_body(spec: &CreateSpec<'_>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "name": spec.owner.pod_name(spec.run_id, spec.attempt),
        "image": spec.image,
        "gpu": { "id": normalize_gpu_type_id(spec.gpu_type_id), "count": spec.gpu_count },
        "cloud": spec.cloud_type,
        "disk": spec.container_disk_gb,
        "ports": spec.ports,
        "env": spec.env,
    });
    // Model-cache volume at /workspace; placement is pinned to its datacenter.
    if let Some(vol) = spec.network_volume_id {
        body["mounts"] = serde_json::json!({
            "network": [{ "volumeId": vol, "path": "/workspace" }],
        });
    }
    if !spec.data_center_ids.is_empty() {
        body["dataCenterIds"] = serde_json::json!(spec.data_center_ids);
    }
    body
}

/// One GPU type of Runpod's catalog with its hourly price per GPU in each cloud. A cloud that does not offer the
/// type has `false` in its flag (its price field can still carry a number).
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct GpuPrice {
    pub id: String,
    /// The short display name, e.g. `RTX 4090`.
    pub name: String,
    pub secure: bool,
    pub community: bool,
    price: GpuPrices,
}

#[derive(Debug, Clone, Deserialize)]
struct GpuPrices {
    #[serde(default)]
    secure: Option<f64>,
    #[serde(default)]
    community: Option<f64>,
}

/// What the catalog says one GPU type costs in one cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum CatalogPrice {
    /// The catalog does not list this GPU type at all: it says nothing about price.
    Unlisted,
    /// The catalog lists the type but the cloud does not offer it.
    NotOffered,
    /// USD per hour for the whole pod (per-GPU price times the GPU count).
    PerHour(f64),
}

/// Runpod's GPU catalog, as far as pricing goes.
#[derive(Debug, Clone, Default)]
pub(crate) struct GpuCatalog {
    pub(crate) gpus: Vec<GpuPrice>,
}

impl GpuCatalog {
    /// The hourly price of a pod of `gpu_count` `gpu` GPUs in `cloud`. `gpu` is a `--gpu-types` token: a catalog
    /// id, its underscore spelling or the short display name.
    pub(crate) fn price(&self, gpu: &str, cloud: CloudType, gpu_count: u32) -> CatalogPrice {
        let id = normalize_gpu_type_id(gpu);
        let Some(entry) = self.gpus.iter().find(|g| g.id == id || g.name == gpu) else {
            return CatalogPrice::Unlisted;
        };
        let (offered, per_gpu) = match cloud {
            CloudType::Community => (entry.community, entry.price.community),
            CloudType::Secure => (entry.secure, entry.price.secure),
        };
        match per_gpu.filter(|price| offered && *price > 0.0) {
            Some(price) => CatalogPrice::PerHour(price * f64::from(gpu_count)),
            None => CatalogPrice::NotOffered,
        }
    }
}

/// One page of `GET /v2/pods`: the wrapped pods and the cursor of the next page.
#[derive(Debug, Deserialize)]
struct PodPage {
    pods: Vec<Pod>,
    pagination: Pagination,
}

#[derive(Debug, Deserialize)]
struct Pagination {
    #[serde(rename = "hasNextPage")]
    has_next_page: bool,
    #[serde(rename = "nextCursor")]
    next_cursor: Option<String>,
}

/// An RFC 9457 problem document: what v2 answers every error with.
#[derive(Deserialize)]
struct Problem {
    title: String,
    detail: String,
    #[serde(default)]
    errors: Vec<String>,
}

/// The text of a failed response: the problem document's `title: detail` (with its validation `errors`), or the
/// raw body when it is not a problem document.
fn problem_text(body: &str) -> String {
    match serde_json::from_str::<Problem>(body) {
        Ok(problem) if problem.errors.is_empty() => {
            format!("{}: {}", problem.title, problem.detail)
        }
        Ok(problem) => format!(
            "{}: {} [{}]",
            problem.title,
            problem.detail,
            problem.errors.join("; ")
        ),
        Err(_) => body.to_string(),
    }
}

/// Parse the `GET /v2/catalog/gpus` answer.
pub(crate) fn parse_gpu_catalog(text: &str) -> Result<GpuCatalog> {
    #[derive(Deserialize)]
    struct Answer {
        gpus: Vec<GpuPrice>,
    }
    serde_json::from_str::<Answer>(text)
        .map(|answer| GpuCatalog { gpus: answer.gpus })
        .with_context(|| format!("parse gpu catalog: {text}"))
}

/// Parse one `GET /v2/pods` page.
fn parse_pod_page(text: &str) -> Result<PodPage> {
    serde_json::from_str(text).with_context(|| format!("parse list_pods: {text}"))
}

impl RunPod {
    pub(crate) fn new(key: String) -> Result<RunPod> {
        Self::with_base(key, BASE)
    }

    /// Build a client against an alternate v2 origin (scheme, host and the `/v2` prefix). Host tests use this
    /// to serve loopback responses; production callers use [`RunPod::new`].
    pub(crate) fn with_base(key: String, base: impl Into<String>) -> Result<RunPod> {
        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()?;
        Ok(RunPod {
            client,
            key,
            base: base.into(),
        })
    }

    fn auth(&self, rb: reqwest::blocking::RequestBuilder) -> reqwest::blocking::RequestBuilder {
        rb.bearer_auth(&self.key)
    }

    pub(crate) fn create_pod(&self, spec: &CreateSpec) -> Result<Pod> {
        let body = create_pod_body(spec);
        let resp = self
            .auth(self.client.post(format!("{}/pods", self.base)))
            .json(&body)
            .send()
            .context("create_pod request")?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(CreateError {
                status: status.as_u16(),
                body: problem_text(&text),
            }
            .into());
        }
        serde_json::from_str::<Pod>(&text)
            .with_context(|| format!("parse create_pod response: {text}"))
    }

    /// The GPU catalog with each cloud's hourly price per GPU.
    pub(crate) fn gpu_catalog(&self) -> Result<GpuCatalog> {
        let resp = self
            .auth(self.client.get(format!("{}/catalog/gpus", self.base)))
            .send()
            .context("gpu catalog request")?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!(
                "gpu catalog failed ({status}): {}",
                problem_text(&text)
            ));
        }
        parse_gpu_catalog(&text)
    }

    /// Read one pod by id: the only read that carries the live ssh mapping.
    pub(crate) fn get_pod(&self, id: &str) -> Result<Pod> {
        let resp = self
            .auth(self.client.get(format!("{}/pods/{id}", self.base)))
            .send()
            .context("get_pod request")?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if status == reqwest::StatusCode::NOT_FOUND {
            return Err(PodNotFound(id.to_string()).into());
        }
        if !status.is_success() {
            return Err(anyhow!(
                "get_pod {id} failed ({status}): {}",
                problem_text(&text)
            ));
        }
        serde_json::from_str::<Pod>(&text).with_context(|| format!("parse get_pod: {text}"))
    }

    /// Every pod of the account, following the pagination cursor to the last page. List entries carry no ssh
    /// mapping; callers needing one read the pod by id.
    pub(crate) fn list_pods(&self) -> Result<Vec<Pod>> {
        let mut pods = Vec::new();
        let mut cursor: Option<String> = None;
        loop {
            let mut request = self.auth(self.client.get(format!("{}/pods", self.base)));
            if let Some(cursor) = &cursor {
                request = request.query(&[("cursor", cursor)]);
            }
            let resp = request.send().context("list_pods request")?;
            let status = resp.status();
            let text = resp.text().unwrap_or_default();
            if !status.is_success() {
                return Err(anyhow!(
                    "list_pods failed ({status}): {}",
                    problem_text(&text)
                ));
            }
            let page = parse_pod_page(&text)?;
            pods.extend(page.pods);
            match (page.pagination.has_next_page, page.pagination.next_cursor) {
                (false, _) => return Ok(pods),
                (true, Some(next)) => cursor = Some(next),
                (true, None) => {
                    return Err(anyhow!("list_pods: a next page without a cursor: {text}"));
                }
            }
        }
    }

    /// Permanently delete a pod. A 404 counts as already gone.
    pub(crate) fn delete_pod(&self, id: &str) -> Result<()> {
        let resp = self
            .auth(self.client.delete(format!("{}/pods/{id}", self.base)))
            .send()
            .context("delete_pod request")?;
        let status = resp.status();
        if delete_status_is_gone(status.as_u16()) {
            return Ok(());
        }
        Err(anyhow!(
            "delete_pod {id} failed ({status}): {}",
            problem_text(&resp.text().unwrap_or_default())
        ))
    }
}

/// Classify a `DELETE /v2/pods/{id}` status code: 2xx (204) and 404 both mean the pod is gone
/// (FR-015 exact-id teardown treats an already-missing pod as success). Every other
/// status is a failure the caller must see.
pub(crate) fn delete_status_is_gone(status: u16) -> bool {
    (200..300).contains(&status) || status == 404
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    fn owner() -> PodOwnerPrefix {
        Db::open(":memory:").unwrap().pod_owner_prefix().unwrap()
    }

    fn spec<'a>(
        owner: &'a PodOwnerPrefix,
        env: &'a HashMap<String, String>,
        gpu: &'a str,
    ) -> CreateSpec<'a> {
        CreateSpec {
            owner,
            run_id: "run-1",
            attempt: 2,
            image: "ghcr.io/kikijiki/poot-bench:latest",
            gpu_type_id: gpu,
            cloud_type: CloudType::Secure,
            gpu_count: 1,
            container_disk_gb: 40,
            ports: &["22/tcp"],
            env,
            network_volume_id: None,
            data_center_ids: &[],
        }
    }

    /// A loopback Runpod: answers each request with the next canned `(status, body)` and reports every request
    /// as `"<request line>\n<body>"` (`<METHOD> <path+query> HTTP/1.1`), so a test sees the real path.
    fn serve(responses: Vec<(u16, &'static str)>) -> (String, mpsc::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback fixture");
        let base = format!("http://{}/v2", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            for (status, body) in responses {
                let (mut stream, _) = listener.accept().expect("accept");
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                let (head_len, content_length) = loop {
                    let n = stream.read(&mut chunk).expect("read request");
                    buf.extend_from_slice(&chunk[..n]);
                    let text = String::from_utf8_lossy(&buf).into_owned();
                    if let Some(end) = text.find("\r\n\r\n") {
                        let length = text
                            .lines()
                            .find_map(|l| {
                                let (k, v) = l.split_once(':')?;
                                k.eq_ignore_ascii_case("content-length")
                                    .then(|| v.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        break (end + 4, length);
                    }
                    assert!(n > 0, "the client closed before sending a request head");
                };
                while buf.len() < head_len + content_length {
                    let n = stream.read(&mut chunk).expect("read body");
                    assert!(n > 0, "the client closed mid-body");
                    buf.extend_from_slice(&chunk[..n]);
                }
                let text = String::from_utf8_lossy(&buf).into_owned();
                let line = text.lines().next().unwrap_or_default().to_string();
                tx.send(format!("{line}\n{}", &text[head_len..])).unwrap();
                let reply = format!(
                    "HTTP/1.1 {status} X\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                stream.write_all(reply.as_bytes()).expect("reply");
            }
        });
        (base, rx)
    }

    const RUNNING_POD: &str = r#"{"id":"abc123","name":"poot-bench-run-1","status":"RUNNING","cost":0.89,
        "cudaVersion":"12.8","ssh":{"proxy":{"host":"ssh.runpod.io","port":22,"username":"t","command":"ssh t@ssh.runpod.io"},
        "direct":{"host":"1.2.3.4","port":40001,"username":"root","command":"ssh -p 40001 root@1.2.3.4"}},
        "runtime":{"uptime":5,"ports":[{"private":22,"public":40001,"type":"tcp","ip":"1.2.3.4"}]}}"#;

    /// The v2 catalog answer parses; a cloud's flag, not its price field, says whether it is offered.
    #[test]
    fn the_gpu_catalog_parses_and_reads_a_false_flag_or_zero_price_as_not_offered() {
        let catalog = parse_gpu_catalog(
            r#"{"gpus":[
                {"id":"NVIDIA GeForce RTX 3090","name":"RTX 3090","secure":true,"community":true,"price":{"community":0.22,"secure":0.5}},
                {"id":"NVIDIA H100 80GB HBM3","name":"H100 SXM","secure":true,"community":false,"price":{"community":1.5,"secure":2.69}},
                {"id":"NVIDIA RTX 3080","name":"RTX 3080","secure":false,"community":true,"price":{"community":0.17,"secure":0}},
                {"id":"AMD Instinct MI300X OAM","name":"MI300X","secure":true,"community":false,"price":{"community":0.5,"secure":1.99}}]}"#,
        )
        .unwrap();
        assert_eq!(
            catalog.price("NVIDIA GeForce RTX 3090", CloudType::Community, 1),
            CatalogPrice::PerHour(0.22)
        );
        assert_eq!(
            catalog.price("NVIDIA_GeForce_RTX_3090", CloudType::Secure, 2),
            CatalogPrice::PerHour(1.0),
            "the underscore spelling and the GPU count both price"
        );
        assert_eq!(
            catalog.price("NVIDIA H100 80GB HBM3", CloudType::Community, 1),
            CatalogPrice::NotOffered,
            "a community price of 1.5 does not make the type offered there"
        );
        assert_eq!(
            catalog.price("NVIDIA RTX 3080", CloudType::Secure, 1),
            CatalogPrice::NotOffered
        );
        assert_eq!(
            catalog.price("MI300X", CloudType::Secure, 1),
            CatalogPrice::PerHour(1.99),
            "the short display name finds the entry"
        );
        assert_eq!(
            catalog.price("Some Future GPU", CloudType::Secure, 1),
            CatalogPrice::Unlisted
        );
        assert!(
            parse_gpu_catalog(r#"{"detail":"x","status":401,"title":"Unauthorized"}"#).is_err()
        );
    }

    /// The price is `cost` (hourly USD) on a create response and on a read.
    #[test]
    fn a_pod_price_is_cost() {
        let pod: Pod =
            serde_json::from_str(r#"{"id":"a","status":"PROVISIONING","cost":0.22}"#).unwrap();
        assert_eq!(pod.cost_per_hr, Some(0.22));
        assert_eq!(pod.status, PodStatus::Provisioning);
    }

    /// A running pod deserializes from the v2 shape and exposes its direct ssh endpoint; the proxy is not an
    /// endpoint (it carries no scp).
    #[test]
    fn pod_deserializes_a_running_pod_with_direct_ssh() {
        let pod: Pod = serde_json::from_str(RUNNING_POD).unwrap();
        assert_eq!(pod.id, "abc123");
        assert_eq!(pod.name, "poot-bench-run-1");
        assert_eq!(pod.status, PodStatus::Running);
        assert_eq!(pod.reported_cuda().as_deref(), Some("12.8"));
        assert_eq!(pod.ssh_endpoint(), Some(("1.2.3.4".to_string(), 40001)));
    }

    /// A pod is not ssh-reachable until `ssh.direct` is set; an unknown status does not fail the parse.
    #[test]
    fn pod_endpoint_pending_states() {
        let pending: Pod = serde_json::from_str(
            r#"{"id":"p","status":"STARTING","ssh":{"proxy":null,"direct":null},"cudaVersion":null}"#,
        )
        .unwrap();
        assert_eq!(pending.status, PodStatus::Starting);
        assert_eq!(pending.ssh_endpoint(), None);
        assert_eq!(pending.reported_cuda(), None);

        let no_ssh: Pod = serde_json::from_str(r#"{"id":"p","status":"EXITED"}"#).unwrap();
        assert_eq!(no_ssh.ssh_endpoint(), None);

        let blank_cuda: Pod =
            serde_json::from_str(r#"{"id":"p","status":"RUNNING","cudaVersion":" "}"#).unwrap();
        assert_eq!(blank_cuda.reported_cuda(), None);

        let future: Pod = serde_json::from_str(r#"{"id":"p","status":"HIBERNATING"}"#).unwrap();
        assert_eq!(future.status, PodStatus::Unknown);

        assert!(
            serde_json::from_str::<Pod>(r#"{"id":"p","desiredStatus":"RUNNING"}"#).is_err(),
            "the v1 `desiredStatus` is not a status"
        );
    }

    /// A list page is the wrapped `{"pods":[...],"pagination":{...}}`; the v1 bare array and malformed JSON are
    /// errors, not a silent empty list.
    #[test]
    fn parse_pod_page_reads_the_wrapper_and_refuses_the_v1_array() {
        let page = parse_pod_page(
            r#"{"pods":[{"id":"a","name":"n1","status":"RUNNING"},{"id":"b","status":"EXITED"}],
                "pagination":{"hasNextPage":true,"nextCursor":"c1"}}"#,
        )
        .unwrap();
        let ids: Vec<&str> = page.pods.iter().map(|p| p.id.as_str()).collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert!(page.pagination.has_next_page);
        assert_eq!(page.pagination.next_cursor.as_deref(), Some("c1"));

        let empty =
            parse_pod_page(r#"{"pods":[],"pagination":{"hasNextPage":false,"nextCursor":null}}"#)
                .unwrap();
        assert!(empty.pods.is_empty());

        assert!(parse_pod_page(r#"[{"id":"a"}]"#).is_err());
        assert!(parse_pod_page("not json").is_err());
    }

    /// The v2 create body: nested one-type `gpu`, `cloud`, `disk`, `env` as an object, no v1 field.
    #[test]
    fn create_pod_body_is_the_nested_v2_shape() {
        let mut env = HashMap::new();
        env.insert(
            "NVIDIA_DRIVER_CAPABILITIES".to_string(),
            "compute,utility,graphics".to_string(),
        );
        env.insert("PUBLIC_KEY".to_string(), "ssh-ed25519 AAAA".to_string());
        let owner = owner();
        let body = create_pod_body(&spec(&owner, &env, "NVIDIA RTX A5000"));
        assert_eq!(
            body["name"],
            owner.pod_name("run-1", 2),
            "the wire name is built from the owner prefix"
        );
        assert!(
            owner.owns(body["name"].as_str().unwrap()),
            "reap can select the pod by that prefix"
        );
        assert_eq!(body["image"], "ghcr.io/kikijiki/poot-bench:latest");
        assert_eq!(
            body["gpu"],
            serde_json::json!({"id": "NVIDIA RTX A5000", "count": 1}),
            "one GPU type, nested"
        );
        assert_eq!(body["cloud"], "SECURE");
        assert_eq!(body["disk"], 40);
        assert_eq!(body["ports"], serde_json::json!(["22/tcp"]));
        assert_eq!(body["env"]["PUBLIC_KEY"], "ssh-ed25519 AAAA");
        assert_eq!(
            body["env"]["NVIDIA_DRIVER_CAPABILITIES"],
            "compute,utility,graphics"
        );
        assert!(
            body["env"].is_object(),
            "env is an object, not [{{key,value}}]"
        );
        // v2 answers 422 to any field it does not know: none of the v1 names may be sent.
        for gone in [
            "imageName",
            "gpuTypeIds",
            "gpuCount",
            "cloudType",
            "containerDiskInGb",
            "supportPublicIp",
            "networkVolumeId",
            "volumeMountPath",
        ] {
            assert!(body.get(gone).is_none(), "v1 field {gone} in {body}");
        }
        assert!(body.get("mounts").is_none(), "no volume means no mounts");
        assert!(body.get("dataCenterIds").is_none());
    }

    /// A network volume is a `mounts.network` entry with its path; data centers serialize in order, and are
    /// omitted when empty.
    #[test]
    fn create_pod_body_mounts_the_volume_and_pins_data_centers() {
        let env = HashMap::new();
        let owner = owner();
        let mut with = spec(&owner, &env, "NVIDIA GeForce RTX 3090");
        with.network_volume_id = Some("vol-1");
        with.data_center_ids = &["EU-RO-1", "US-CA-2", "AP-IN-1"];
        with.cloud_type = CloudType::Community;
        let body = create_pod_body(&with);
        assert_eq!(
            body["mounts"],
            serde_json::json!({"network": [{"volumeId": "vol-1", "path": "/workspace"}]})
        );
        assert_eq!(
            body["dataCenterIds"],
            serde_json::json!(["EU-RO-1", "US-CA-2", "AP-IN-1"])
        );
        assert_eq!(body["cloud"], "COMMUNITY");
        assert!(body.get("allowedCudaVersions").is_none());
        assert!(body["gpu"].get("allowedCudaVersions").is_none());

        let none = create_pod_body(&spec(&owner, &env, "NVIDIA GeForce RTX 3090"));
        assert!(none.get("dataCenterIds").is_none());
        let one_dc = {
            let mut s = spec(&owner, &env, "NVIDIA GeForce RTX 3090");
            s.data_center_ids = &["EU-RO-1"];
            create_pod_body(&s)
        };
        assert_eq!(one_dc["dataCenterIds"], serde_json::json!(["EU-RO-1"]));
    }

    /// Both spellings of a GPU type normalize to the catalog id `gpu.id` takes; unknown tokens pass through.
    #[test]
    fn normalize_gpu_type_id_both_spellings() {
        assert_eq!(
            normalize_gpu_type_id("AMD_Instinct_MI300X_OAM"),
            "AMD Instinct MI300X OAM"
        );
        assert_eq!(
            normalize_gpu_type_id("AMD Instinct MI300X OAM"),
            "AMD Instinct MI300X OAM"
        );
        assert_eq!(normalize_gpu_type_id("Some Future GPU"), "Some Future GPU");
        assert_eq!(
            normalize_gpu_type_id("NVIDIA GeForce RTX 3090"),
            "NVIDIA GeForce RTX 3090"
        );
        let env = HashMap::new();
        let owner = owner();
        let body = create_pod_body(&spec(&owner, &env, "AMD_Instinct_MI300X_OAM"));
        assert_eq!(body["gpu"]["id"], "AMD Instinct MI300X OAM");
    }

    /// An error is a problem document: its title, detail and validation errors are the text; a body that is
    /// not one is kept raw.
    #[test]
    fn a_problem_document_is_the_error_text() {
        assert_eq!(
            problem_text(r#"{"title":"Bad Request","status":400,"detail":"no machine"}"#),
            "Bad Request: no machine"
        );
        assert_eq!(
            problem_text(
                r#"{"title":"Unprocessable Entity","status":422,"detail":"Request validation failed.",
                    "errors":["$: additional properties 'bogus' not allowed"]}"#
            ),
            "Unprocessable Entity: Request validation failed. [$: additional properties 'bogus' not allowed]"
        );
        assert_eq!(
            problem_text("<html>bad gateway</html>"),
            "<html>bad gateway</html>"
        );
    }

    /// Only a server error, a rate limit or Runpod's no-machine 400 can be waited out; a rule violation (also a
    /// 400), an unknown GPU type (422), a refused balance (402) and a bad key never fix themselves.
    #[test]
    fn only_no_capacity_answers_are_capacity() {
        let err = |status, body: &str| CreateError {
            status,
            body: body.to_string(),
        };
        assert!(err(
            400,
            "Bad Request: There are no longer any instances available with the requested specifications. Please refresh and try again."
        )
        .is_capacity());
        assert!(err(500, "Internal Server Error: boom").is_capacity());
        assert!(err(429, "Too Many Requests: slow down").is_capacity());
        assert!(
            !err(
                400,
                "Bad Request: mounts.persistent and mounts.network are exclusive"
            )
            .is_capacity()
        );
        assert!(!err(422, "Unprocessable Entity: Unknown GPU type: x").is_capacity());
        assert!(!err(402, "Payment Required: insufficient balance").is_capacity());
        assert!(!err(401, "Unauthorized: bad key").is_capacity());
    }

    /// `create_pod` POSTs `/v2/pods` with the nested body and parses the 201 pod, price included.
    #[test]
    fn create_pod_posts_the_v2_body_to_pods() {
        let (base, requests) = serve(vec![(
            201,
            r#"{"id":"new1","name":"n","status":"PROVISIONING","cost":0.89}"#,
        )]);
        let rp = RunPod::with_base("k".into(), base).unwrap();
        let env = HashMap::new();
        let owner = owner();
        let pod = rp
            .create_pod(&spec(&owner, &env, "NVIDIA GeForce RTX 4090"))
            .unwrap();
        assert_eq!((pod.id.as_str(), pod.cost_per_hr), ("new1", Some(0.89)));
        let request = requests.recv().unwrap();
        let (line, body) = request.split_once('\n').unwrap();
        assert_eq!(line, "POST /v2/pods HTTP/1.1");
        let sent: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(sent["gpu"]["id"], "NVIDIA GeForce RTX 4090");
        assert_eq!(sent["image"], "ghcr.io/kikijiki/poot-bench:latest");
    }

    /// A refused create carries the HTTP status and the problem text, and the capacity 400 is recognized.
    #[test]
    fn create_pod_refusal_is_a_create_error_with_the_problem_text() {
        let (base, _requests) = serve(vec![(
            400,
            r#"{"detail":"There are no longer any instances available with the requested specifications. Please refresh and try again.","status":400,"title":"Bad Request"}"#,
        )]);
        let rp = RunPod::with_base("k".into(), base).unwrap();
        let env = HashMap::new();
        let owner = owner();
        let err = rp
            .create_pod(&spec(&owner, &env, "NVIDIA GeForce RTX 4090"))
            .unwrap_err();
        let refusal = err.downcast_ref::<CreateError>().expect("a CreateError");
        assert_eq!(refusal.status, 400);
        assert!(
            refusal
                .body
                .starts_with("Bad Request: There are no longer any instances")
        );
        assert!(refusal.is_capacity());
    }

    /// `get_pod` GETs `/v2/pods/{id}`, parses the direct ssh endpoint, and answers 404 with `PodNotFound`.
    #[test]
    fn get_pod_reads_the_pod_by_id_and_maps_404() {
        let (base, requests) = serve(vec![
            (200, RUNNING_POD),
            (
                404,
                r#"{"detail":"pod not found","status":404,"title":"Not Found"}"#,
            ),
        ]);
        let rp = RunPod::with_base("k".into(), base).unwrap();
        let pod = rp.get_pod("abc123").unwrap();
        assert_eq!(pod.ssh_endpoint(), Some(("1.2.3.4".to_string(), 40001)));
        assert_eq!(
            requests.recv().unwrap().lines().next().unwrap(),
            "GET /v2/pods/abc123 HTTP/1.1"
        );
        let err = rp.get_pod("gone").unwrap_err();
        assert!(err.downcast_ref::<PodNotFound>().is_some(), "{err:#}");
        assert_eq!(
            requests.recv().unwrap().lines().next().unwrap(),
            "GET /v2/pods/gone HTTP/1.1"
        );
    }

    /// `list_pods` follows `pagination.nextCursor` until `hasNextPage` is false and returns every page's pods;
    /// the second request carries the first page's cursor.
    #[test]
    fn list_pods_follows_the_pagination_cursor() {
        let (base, requests) = serve(vec![
            (
                200,
                r#"{"pods":[{"id":"a","status":"RUNNING"}],"pagination":{"hasNextPage":true,"nextCursor":"c1"}}"#,
            ),
            (
                200,
                r#"{"pods":[{"id":"b","status":"EXITED"}],"pagination":{"hasNextPage":false,"nextCursor":null}}"#,
            ),
        ]);
        let rp = RunPod::with_base("k".into(), base).unwrap();
        let ids: Vec<String> = rp.list_pods().unwrap().into_iter().map(|p| p.id).collect();
        assert_eq!(ids, vec!["a", "b"]);
        assert_eq!(
            requests.recv().unwrap().lines().next().unwrap(),
            "GET /v2/pods HTTP/1.1"
        );
        assert_eq!(
            requests.recv().unwrap().lines().next().unwrap(),
            "GET /v2/pods?cursor=c1 HTTP/1.1"
        );
    }

    /// A next page without a cursor is an error, never a silently short list (a reap or a teardown check would
    /// miss pods).
    #[test]
    fn list_pods_refuses_a_next_page_without_a_cursor() {
        let (base, _requests) = serve(vec![(
            200,
            r#"{"pods":[],"pagination":{"hasNextPage":true,"nextCursor":null}}"#,
        )]);
        let rp = RunPod::with_base("k".into(), base).unwrap();
        assert!(rp.list_pods().is_err());
    }

    /// `delete_pod` DELETEs `/v2/pods/{id}`: 204 and 404 are gone, a 409 is a failure with its problem text.
    #[test]
    fn delete_pod_deletes_by_id_and_reads_204_and_404_as_gone() {
        let (base, requests) = serve(vec![
            (204, ""),
            (
                404,
                r#"{"detail":"pod not found","status":404,"title":"Not Found"}"#,
            ),
            (
                409,
                r#"{"detail":"pod belongs to a cluster","status":409,"title":"Conflict"}"#,
            ),
        ]);
        let rp = RunPod::with_base("k".into(), base).unwrap();
        rp.delete_pod("p1").unwrap();
        rp.delete_pod("p2").unwrap();
        let err = rp.delete_pod("p3").unwrap_err();
        assert!(
            format!("{err:#}").contains("Conflict: pod belongs to a cluster"),
            "{err:#}"
        );
        let lines: Vec<String> = (0..3)
            .map(|_| requests.recv().unwrap().lines().next().unwrap().to_string())
            .collect();
        assert_eq!(
            lines,
            [
                "DELETE /v2/pods/p1 HTTP/1.1",
                "DELETE /v2/pods/p2 HTTP/1.1",
                "DELETE /v2/pods/p3 HTTP/1.1"
            ]
        );
    }

    /// `gpu_catalog` GETs `/v2/catalog/gpus` (the unfiltered list, which carries prices).
    #[test]
    fn gpu_catalog_reads_the_v2_catalog_path() {
        let (base, requests) = serve(vec![(
            200,
            r#"{"gpus":[{"id":"NVIDIA A40","name":"A40","secure":true,"community":true,"price":{"community":0.2,"secure":0.4}}]}"#,
        )]);
        let rp = RunPod::with_base("k".into(), base).unwrap();
        let catalog = rp.gpu_catalog().unwrap();
        assert_eq!(
            catalog.price("NVIDIA A40", CloudType::Secure, 1),
            CatalogPrice::PerHour(0.4)
        );
        assert_eq!(
            requests.recv().unwrap().lines().next().unwrap(),
            "GET /v2/catalog/gpus HTTP/1.1"
        );
    }
}
