//! Minimal blocking RunPod REST v1 client (`https://rest.runpod.io/v1`, Bearer auth): create / get /
//! list / delete a pod.

use std::collections::HashMap;

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;

use crate::db::PodOwnerPrefix;

const BASE: &str = "https://rest.runpod.io/v1";

pub struct RunPod {
    client: reqwest::blocking::Client,
    key: String,
    /// REST v1 origin. Production always uses [`BASE`]; host tests may point at a
    /// loopback fixture with [`RunPod::with_base`] to observe real request paths.
    base: String,
}

/// A pod as the REST API returns it. `public_ip` and `port_mappings` populate only once the pod has
/// booted, so callers poll `get_pod` until `ssh_endpoint` is Some.
#[derive(Debug, Deserialize, Default)]
#[serde(default)]
pub struct Pod {
    pub id: String,
    pub name: String,
    #[serde(rename = "desiredStatus")]
    pub desired_status: String,
    #[serde(rename = "publicIp")]
    pub public_ip: Option<String>,
    #[serde(rename = "portMappings")]
    pub port_mappings: Option<HashMap<String, u32>>,
    /// The assigned machine. Empty object `{}` until a host picks up the pod: empty `machine` and empty
    /// `publicIp` with `desiredStatus=RUNNING` means rented but not started (cold image pull or a flaky
    /// community machine).
    #[serde(default)]
    pub machine: serde_json::Value,
}

impl Pod {
    /// True once a host machine has been assigned (the `machine` object is non-empty).
    pub fn machine_ready(&self) -> bool {
        self.machine
            .as_object()
            .map(|m| !m.is_empty())
            .unwrap_or(false)
    }

    /// (public_ip, public_port) for the exposed 22/tcp port, or None until the pod is up.
    pub fn ssh_endpoint(&self) -> Option<(String, u16)> {
        let ip = self.public_ip.as_ref()?;
        if ip.is_empty() {
            return None;
        }
        let port = self.port_mappings.as_ref()?.get("22").copied()?;
        Some((ip.clone(), port as u16))
    }
}

/// Fields for a create request.
pub struct CreateSpec<'a> {
    /// The state database's pod prefix. The pod's cloud name is built from it, so `reap` finds every pod
    /// this create call makes.
    pub owner: &'a PodOwnerPrefix,
    pub run_id: &'a str,
    /// The provisioning attempt of `run_id` this pod is.
    pub attempt: u32,
    pub image: &'a str,
    pub gpu_type_id: &'a str,
    pub cloud_type: &'a str, // COMMUNITY | SECURE | ALL
    pub gpu_count: u32,
    pub container_disk_gb: u32,
    pub ports: &'a [&'a str], // e.g. ["22/tcp"]
    pub env: &'a HashMap<String, String>,
    /// Network volume to attach at /workspace (the model-weight cache). Pins the pod to the volume's
    /// datacenter, so `data_center_ids` should match it.
    pub network_volume_id: Option<&'a str>,
    /// Data centers to pin the pod to (`dataCenterIds`). Empty means no pin (field omitted).
    pub data_center_ids: &'a [&'a str],
    /// Host CUDA versions to place on (`allowedCudaVersions`, e.g. `["12.8","12.9","13.0"]`); empty means
    /// no filter. Must cover the image's requirement (the default `cu1281` image needs >= 12.8), else an
    /// older-driver host fails the nvidia-container-cli hook (`unsatisfied condition: cuda>=12.8`) with an
    /// OCI start error or a pod that never gets a machine. Build from a floor with `cuda_floor_to_allowed`.
    pub allowed_cuda_versions: &'a [String],
}

/// ONE table of RunPod GPU types this orchestrator provisions: (catalog id spelling, REST display
/// name). REST v1 `gpuTypeIds` is an enum that REQUIRES the display name; the catalog id form
/// (`AMD_Instinct_MI300X_OAM`) is rejected with HTTP 400 "gpuTypeIds enum requires display name".
/// Lookup accepts either spelling. Tokens not in the table pass through unchanged (already a display
/// name, or a type this table has not listed yet).
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

/// Normalize a `--gpu-types` token to the display name the REST `gpuTypeIds` enum accepts.
/// Accepts the catalog id spelling or the display name; unknown tokens pass through unchanged.
pub fn normalize_gpu_type_id(token: &str) -> &str {
    GPU_TYPE_NAMES
        .iter()
        .find(|(id, display)| *id == token || *display == token)
        .map(|(_, display)| *display)
        .unwrap_or(token)
}

/// Build the REST v1 `POST /pods` JSON body for [`RunPod::create_pod`].
///
/// REST v1 takes `env` as a JSON object (`{ "KEY": "value" }`), not a GraphQL-style `[{key,value}]`
/// array. Separate function so tests can assert the wire shape without a live API.
pub fn create_pod_body(spec: &CreateSpec<'_>) -> serde_json::Value {
    let mut body = serde_json::json!({
        "name": spec.owner.pod_name(spec.run_id, spec.attempt),
        "imageName": spec.image,
        "gpuTypeIds": [normalize_gpu_type_id(spec.gpu_type_id)],
        "cloudType": spec.cloud_type,
        "gpuCount": spec.gpu_count,
        "containerDiskInGb": spec.container_disk_gb,
        "ports": spec.ports,
        "env": spec.env,
        "supportPublicIp": true,
    });
    // Model-cache volume at /workspace; placement is pinned to its datacenter.
    if let Some(vol) = spec.network_volume_id {
        body["networkVolumeId"] = serde_json::json!(vol);
        body["volumeMountPath"] = serde_json::json!("/workspace");
    }
    if !spec.data_center_ids.is_empty() {
        body["dataCenterIds"] = serde_json::json!(spec.data_center_ids);
    }
    if !spec.allowed_cuda_versions.is_empty() {
        body["allowedCudaVersions"] = serde_json::json!(spec.allowed_cuda_versions);
    }
    body
}

/// Parse a `list_pods` body: a bare `[...]` array (what `GET /pods` returns) or a `{ "pods": [...] }`
/// wrapper.
pub fn parse_pod_list(text: &str) -> Result<Vec<Pod>> {
    if let Ok(v) = serde_json::from_str::<Vec<Pod>>(text) {
        return Ok(v);
    }
    #[derive(Deserialize)]
    struct Wrap {
        pods: Vec<Pod>,
    }
    serde_json::from_str::<Wrap>(text)
        .map(|w| w.pods)
        .with_context(|| format!("parse list_pods: {text}"))
}

impl RunPod {
    pub fn new(key: String) -> Result<RunPod> {
        Self::with_base(key, BASE)
    }

    /// Build a client against an alternate REST origin. Host tests use this to serve
    /// loopback responses; production callers use [`RunPod::new`].
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

    pub fn create_pod(&self, spec: &CreateSpec) -> Result<Pod> {
        let body = create_pod_body(spec);
        let resp = self
            .auth(self.client.post(format!("{}/pods", self.base)))
            .json(&body)
            .send()
            .context("create_pod request")?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("create_pod failed ({status}): {text}"));
        }
        serde_json::from_str::<Pod>(&text)
            .with_context(|| format!("parse create_pod response: {text}"))
    }

    pub fn get_pod(&self, id: &str) -> Result<Pod> {
        let resp = self
            .auth(self.client.get(format!("{}/pods/{id}", self.base)))
            .send()
            .context("get_pod request")?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("get_pod {id} failed ({status}): {text}"));
        }
        serde_json::from_str::<Pod>(&text).with_context(|| format!("parse get_pod: {text}"))
    }

    pub fn list_pods(&self) -> Result<Vec<Pod>> {
        let resp = self
            .auth(self.client.get(format!("{}/pods", self.base)))
            .send()
            .context("list_pods request")?;
        let status = resp.status();
        let text = resp.text().unwrap_or_default();
        if !status.is_success() {
            return Err(anyhow!("list_pods failed ({status}): {text}"));
        }
        parse_pod_list(&text)
    }

    /// Permanently delete a pod. A 404 counts as already gone.
    pub fn delete_pod(&self, id: &str) -> Result<()> {
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
            resp.text().unwrap_or_default()
        ))
    }
}

/// Classify a `DELETE /pods/{id}` status code: 2xx and 404 both mean the pod is gone
/// (FR-015 exact-id teardown treats an already-missing pod as success). Every other
/// status is a failure the caller must see.
pub fn delete_status_is_gone(status: u16) -> bool {
    (200..300).contains(&status) || status == 404
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Db;

    fn owner() -> PodOwnerPrefix {
        Db::open(":memory:").unwrap().pod_owner_prefix().unwrap()
    }

    /// A running pod deserializes, and `ssh_endpoint`/`machine_ready` reflect a booted pod.
    #[test]
    fn pod_deserializes_a_running_pod_with_ssh() {
        let json = r#"{
            "id": "abc123",
            "name": "poot-bench-run-1",
            "desiredStatus": "RUNNING",
            "publicIp": "1.2.3.4",
            "portMappings": {"22": 40001, "8000": 40002},
            "machine": {"gpuTypeId": "NVIDIA A40"}
        }"#;
        let pod: Pod = serde_json::from_str(json).unwrap();
        assert_eq!(pod.id, "abc123");
        assert_eq!(pod.name, "poot-bench-run-1");
        assert_eq!(pod.desired_status, "RUNNING");
        assert!(
            pod.machine_ready(),
            "a non-empty machine object means a host is assigned"
        );
        assert_eq!(pod.ssh_endpoint(), Some(("1.2.3.4".to_string(), 40001)));
    }

    /// A pod is not SSH-reachable until it has a public IP and the 22/tcp mapping; `machine_ready` is
    /// false while the machine object is empty or absent.
    #[test]
    fn pod_endpoint_and_machine_ready_pending_states() {
        // rented, no host yet: empty machine, no IP.
        let pending: Pod =
            serde_json::from_str(r#"{"id":"p","desiredStatus":"RUNNING","machine":{}}"#).unwrap();
        assert!(!pending.machine_ready());
        assert_eq!(pending.ssh_endpoint(), None);

        // machine absent entirely (null) -> not ready.
        let no_machine: Pod = serde_json::from_str(r#"{"id":"p"}"#).unwrap();
        assert!(!no_machine.machine_ready());

        // empty public IP string counts as "not up".
        let empty_ip: Pod =
            serde_json::from_str(r#"{"id":"p","publicIp":"","portMappings":{"22":40001}}"#)
                .unwrap();
        assert_eq!(empty_ip.ssh_endpoint(), None);

        // IP present but no 22/tcp mapping yet -> not reachable.
        let no_ssh_port: Pod = serde_json::from_str(
            r#"{"id":"p","publicIp":"1.2.3.4","portMappings":{"8000":40002}}"#,
        )
        .unwrap();
        assert_eq!(no_ssh_port.ssh_endpoint(), None);
    }

    /// `parse_pod_list` accepts a bare array and a `{"pods":[...]}` wrapper, and errors on malformed JSON.
    #[test]
    fn parse_pod_list_accepts_both_shapes() {
        let bare = r#"[{"id":"a","name":"n1"},{"id":"b","name":"n2"}]"#;
        let ids: Vec<String> = parse_pod_list(bare)
            .unwrap()
            .into_iter()
            .map(|p| p.id)
            .collect();
        assert_eq!(ids, vec!["a", "b"]);

        let wrapped = r#"{"pods":[{"id":"c"}]}"#;
        assert_eq!(parse_pod_list(wrapped).unwrap()[0].id, "c");

        // an empty account is a bare empty array, not an error.
        assert!(parse_pod_list("[]").unwrap().is_empty());

        // malformed JSON is an error, not a silent empty list.
        assert!(parse_pod_list("not json").is_err());
    }

    /// `env` must serialize as a JSON object, not a GraphQL-style `[{key,value}]` array.
    #[test]
    fn create_pod_body_env_is_rest_object_with_driver_caps() {
        let mut env = HashMap::new();
        env.insert(
            "NVIDIA_DRIVER_CAPABILITIES".to_string(),
            "compute,utility,graphics".to_string(),
        );
        env.insert("PUBLIC_KEY".to_string(), "ssh-ed25519 AAAA".to_string());
        let owner = owner();
        let spec = CreateSpec {
            owner: &owner,
            run_id: "run-1",
            attempt: 2,
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
        assert_eq!(
            body["name"],
            owner.pod_name("run-1", 2),
            "the wire name is built from the owner prefix"
        );
        assert!(
            owner.owns(body["name"].as_str().unwrap()),
            "reap can select the pod by that prefix"
        );
        assert_eq!(body["imageName"], "ghcr.io/kikijiki/poot-bench:latest");
        assert_eq!(body["gpuTypeIds"], serde_json::json!(["NVIDIA RTX A5000"]));
        assert_eq!(body["cloudType"], "SECURE");
        assert_eq!(body["gpuCount"], 1);
        assert_eq!(body["containerDiskInGb"], 40);
        assert_eq!(body["ports"], serde_json::json!(["22/tcp"]));
        assert_eq!(body["supportPublicIp"], true);
        assert!(body.get("networkVolumeId").is_none());
        assert!(body.get("volumeMountPath").is_none());
        assert!(body.get("dataCenterIds").is_none());
        assert!(body.get("allowedCudaVersions").is_none());
        let env_val = body.get("env").expect("create body must include env");
        assert!(
            env_val.is_object(),
            "REST v1 env must be a JSON object, got {env_val}"
        );
        assert!(
            !env_val.is_array(),
            "REST v1 env must not be a GraphQL-style [{{key,value}}] array: {env_val}"
        );
        assert_eq!(
            env_val
                .get("NVIDIA_DRIVER_CAPABILITIES")
                .and_then(|v| v.as_str()),
            Some("compute,utility,graphics"),
            "default driver-caps pair must appear as an object field in the create JSON"
        );
        assert_eq!(
            env_val.get("PUBLIC_KEY").and_then(|v| v.as_str()),
            Some("ssh-ed25519 AAAA")
        );
    }

    /// Empty optional fields stay off the body; volume/datacenter/cuda filters serialize when set.
    #[test]
    fn create_pod_body_omits_unset_optional_placement_fields() {
        let env = HashMap::new();
        let owner = owner();
        let unset = CreateSpec {
            owner: &owner,
            run_id: "n",
            attempt: 1,
            image: "img",
            gpu_type_id: "NVIDIA GeForce RTX 3090",
            cloud_type: "COMMUNITY",
            gpu_count: 1,
            container_disk_gb: 20,
            ports: &["22/tcp"],
            env: &env,
            network_volume_id: None,
            data_center_ids: &[],
            allowed_cuda_versions: &[],
        };
        let body = create_pod_body(&unset);
        assert!(
            body.get("networkVolumeId").is_none(),
            "unset network volume must be omitted, got {body}"
        );
        assert!(body.get("volumeMountPath").is_none());
        assert!(body.get("dataCenterIds").is_none());
        assert!(body.get("allowedCudaVersions").is_none());
        assert!(body["env"].is_object());

        let cuda = vec!["12.8".to_string(), "12.9".to_string()];
        let set = CreateSpec {
            owner: &owner,
            run_id: "n",
            attempt: 1,
            image: "img",
            gpu_type_id: "NVIDIA GeForce RTX 3090",
            cloud_type: "COMMUNITY",
            gpu_count: 1,
            container_disk_gb: 20,
            ports: &["22/tcp"],
            env: &env,
            network_volume_id: Some("vol-1"),
            data_center_ids: &["US-CA-2"],
            allowed_cuda_versions: &cuda,
        };
        let body = create_pod_body(&set);
        assert_eq!(body["networkVolumeId"], "vol-1");
        assert_eq!(body["volumeMountPath"], "/workspace");
        assert_eq!(body["dataCenterIds"], serde_json::json!(["US-CA-2"]));
        assert_eq!(
            body["allowedCudaVersions"],
            serde_json::json!(["12.8", "12.9"])
        );
        assert!(body["env"].is_object());
    }

    /// Exact wire shape for `dataCenterIds`: absent when empty, exact array when one or many.
    /// Dropping the `dataCenterIds` insertion in `create_pod_body` reddens the set/multi rows.
    #[test]
    fn create_pod_body_data_center_ids_shape() {
        let env = HashMap::new();
        let owner = owner();
        let mk = |dcs: &'static [&'static str]| CreateSpec {
            owner: &owner,
            run_id: "n",
            attempt: 1,
            image: "img",
            gpu_type_id: "NVIDIA RTX A5000",
            cloud_type: "SECURE",
            gpu_count: 1,
            container_disk_gb: 30,
            ports: &["22/tcp"],
            env: &env,
            network_volume_id: None,
            data_center_ids: dcs,
            allowed_cuda_versions: &[],
        };

        // Absent: field must not appear (keeps existing behavior when no --data-center).
        let none = create_pod_body(&mk(&[]));
        assert!(
            none.get("dataCenterIds").is_none(),
            "empty data_center_ids must omit the field, got {none}"
        );

        // One id: exact field name and value.
        let one = create_pod_body(&mk(&["EU-RO-1"]));
        assert_eq!(
            one["dataCenterIds"],
            serde_json::json!(["EU-RO-1"]),
            "single data center must serialize as a one-element array"
        );

        // Multiple: order preserved, one array entry per id.
        let many = create_pod_body(&mk(&["EU-RO-1", "US-CA-2", "AP-IN-1"]));
        assert_eq!(
            many["dataCenterIds"],
            serde_json::json!(["EU-RO-1", "US-CA-2", "AP-IN-1"]),
            "multiple data centers must serialize in order"
        );
    }

    /// MI300X: both the catalog id spelling and the display name normalize to the display name
    /// the REST `gpuTypeIds` enum requires (HTTP 400 otherwise). Unknown tokens pass through.
    #[test]
    fn normalize_gpu_type_id_mi300x_both_spellings() {
        assert_eq!(
            normalize_gpu_type_id("AMD_Instinct_MI300X_OAM"),
            "AMD Instinct MI300X OAM"
        );
        assert_eq!(
            normalize_gpu_type_id("AMD Instinct MI300X OAM"),
            "AMD Instinct MI300X OAM"
        );
        // Unknown tokens are left alone (already a display name, or not in the table yet).
        assert_eq!(normalize_gpu_type_id("Some Future GPU"), "Some Future GPU");
        // Display-name defaults already used by the CLI stay identity through the table.
        assert_eq!(
            normalize_gpu_type_id("NVIDIA GeForce RTX 3090"),
            "NVIDIA GeForce RTX 3090"
        );
        assert_eq!(
            normalize_gpu_type_id("NVIDIA RTX A5000"),
            "NVIDIA RTX A5000"
        );
    }

    /// The create body applies normalization: id-style input lands on the wire as the display name.
    #[test]
    fn create_pod_body_normalizes_gpu_type_ids() {
        let env = HashMap::new();
        let owner = owner();
        let spec = CreateSpec {
            owner: &owner,
            run_id: "mi300x",
            attempt: 1,
            image: "img",
            gpu_type_id: "AMD_Instinct_MI300X_OAM",
            cloud_type: "SECURE",
            gpu_count: 1,
            container_disk_gb: 30,
            ports: &["22/tcp"],
            env: &env,
            network_volume_id: None,
            data_center_ids: &[],
            allowed_cuda_versions: &[],
        };
        let body = create_pod_body(&spec);
        assert_eq!(
            body["gpuTypeIds"],
            serde_json::json!(["AMD Instinct MI300X OAM"]),
            "id-style gpu_type_id must be normalized to the REST display name"
        );
    }
}
