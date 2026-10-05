//! Spec 248 hot-load/unload admin surface (epic 129 B8): add or remove a LoRA adapter from a running
//! server's live pool without a restart and without disrupting in-flight requests for a different
//! adapter or the base model.
//!
//! Card 315 defines the trust boundary. A loopback listener without an admin key uses a local-process
//! trust model. A non-loopback listener without a key disables every admin route. Setting
//! `POOT_LORA_ADMIN_KEY` on either listener class requires the exact bearer credential. Only admin
//! routes are gated; inference stays unauthenticated. After bounded request-head parsing,
//! authorization runs before body handling, route-shape handling, JSON parsing, or any adapter
//! backend selection or call.
//!
//! The route family is `/v1/lora_adapters`, alongside the existing `/v1/models` naming (see
//! `http.rs`'s `handle` dispatch for the other routes).
//!
//! - `GET /v1/lora_adapters`: list registered adapters (name, pool index, in-flight count).
//! - `POST /v1/lora_adapters` `{"name": "...", "path": "..."}`: hot-load an adapter from a PEFT
//!   directory. 400 on a bad/unsupported adapter, an already-registered name, a rank above the pool's
//!   traced ceiling, or no free reserved slot (see `Runner::hot_load_lora_adapter_dir`).
//! - `DELETE /v1/lora_adapters/{name}`: hot-unload. 404 if `name` is not registered; 409 Conflict if
//!   a decode slot is in flight for it (see `Runner::hot_unload_lora_adapter`); retry once those
//!   requests finish.
//!
//! Only meaningful for `Backend::Decoder` (the `Runner`-backed generic wgpu `batch_engine_loop`, the
//! only LoRA-batched-capable engine loop so far; PTX/ROCm hot-load is a follow-on). Other backend
//! variants (VLM/Gemma4/Qwen3Next/Encoder/CrossEncoder) return a 400 rather than silently no-oping.

use std::net::{IpAddr, TcpStream};

use anyhow::{Result, anyhow, bail};
use poot_llm::Runner;
use ring::digest;
use serde::Deserialize;
use subtle::ConstantTimeEq;

use crate::http::{
    AuthorizationHeader, error_body, respond, respond_error, respond_typed, respond_unsupported,
};
use crate::types::Backend;

const LORA_ADMIN_KEY_ENV: &str = "POOT_LORA_ADMIN_KEY";

/// Deployment policy for the LoRA administration routes. The credential-bearing variant has no
/// `Debug` or `Display` implementation, so structured logging cannot format its secret.
pub(crate) struct LoraAdminPolicy {
    access: LoraAdminAccess,
}

enum LoraAdminAccess {
    LocalOnly,
    Bearer(AdminCredential),
    Disabled,
}

struct AdminCredential([u8; digest::SHA256_OUTPUT_LEN]);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoraAdminPolicyMode {
    LocalOnly,
    BearerRequired,
    Disabled,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LoraAdminDenial {
    Authentication,
    Disabled,
}

impl LoraAdminPolicy {
    /// Build the policy from the address the listener bound and an optional configured key.
    /// `configured_key` is injected for deterministic tests; production passes the environment value.
    pub(crate) fn from_configured_key(
        listener_ip: IpAddr,
        configured_key: Option<&str>,
    ) -> Result<Self> {
        let access = match configured_key {
            Some(key) => {
                if !is_b64token(key) {
                    bail!("{LORA_ADMIN_KEY_ENV} must be one non-empty RFC 6750 b64token");
                }
                LoraAdminAccess::Bearer(AdminCredential(credential_digest(key.as_bytes())))
            }
            None if listener_ip.is_loopback() => LoraAdminAccess::LocalOnly,
            None => LoraAdminAccess::Disabled,
        };
        Ok(Self { access })
    }

    pub(crate) fn from_env(listener_ip: IpAddr) -> Result<Self> {
        let configured = std::env::var_os(LORA_ADMIN_KEY_ENV)
            .map(|value| {
                value.into_string().map_err(|_| {
                    anyhow!("{LORA_ADMIN_KEY_ENV} must contain valid UTF-8 bearer-token bytes")
                })
            })
            .transpose()?;
        Self::from_configured_key(listener_ip, configured.as_deref())
    }

    pub(crate) fn mode(&self) -> LoraAdminPolicyMode {
        match self.access {
            LoraAdminAccess::LocalOnly => LoraAdminPolicyMode::LocalOnly,
            LoraAdminAccess::Bearer(_) => LoraAdminPolicyMode::BearerRequired,
            LoraAdminAccess::Disabled => LoraAdminPolicyMode::Disabled,
        }
    }

    pub(crate) fn authorize(
        &self,
        authorization: &AuthorizationHeader,
    ) -> std::result::Result<(), LoraAdminDenial> {
        let authorization = match authorization {
            AuthorizationHeader::Missing => None,
            AuthorizationHeader::Single(value) => Some(value.as_str()),
            AuthorizationHeader::Duplicate => return Err(LoraAdminDenial::Authentication),
        };
        match &self.access {
            LoraAdminAccess::LocalOnly => Ok(()),
            LoraAdminAccess::Disabled => Err(LoraAdminDenial::Disabled),
            LoraAdminAccess::Bearer(expected) => {
                let presented = authorization
                    .and_then(parse_bearer_credential)
                    .ok_or(LoraAdminDenial::Authentication)?;
                if credential_matches(&expected.0, presented.as_bytes()) {
                    Ok(())
                } else {
                    Err(LoraAdminDenial::Authentication)
                }
            }
        }
    }
}

/// Parse RFC 6750's `Bearer` credentials shape: a case-insensitive scheme, one or more HTTP SP
/// bytes, and exactly one case-sensitive `b64token`. HTAB is not the separator. Malformed headers
/// take the same response path as a missing or incorrect credential.
fn parse_bearer_credential(raw: &str) -> Option<&str> {
    let separator = raw.find(' ')?;
    let (scheme, rest) = raw.split_at(separator);
    let token = rest.trim_start_matches(' ');
    (scheme.eq_ignore_ascii_case("Bearer") && is_b64token(token)).then_some(token)
}

/// RFC 6750 `b64token`: one or more alphanumeric or `-._~+/` bytes, optionally followed by padding
/// `=`. Narrower than "visible ASCII"; rejects all Unicode and whitespace.
fn is_b64token(token: &str) -> bool {
    let bytes = token.as_bytes();
    let unpadded_len = bytes
        .iter()
        .position(|&byte| byte == b'=')
        .unwrap_or(bytes.len());
    unpadded_len > 0
        && bytes[..unpadded_len].iter().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'.' | b'_' | b'~' | b'+' | b'/')
        })
        && bytes[unpadded_len..].iter().all(|&byte| byte == b'=')
}

fn credential_digest(credential: &[u8]) -> [u8; digest::SHA256_OUTPUT_LEN] {
    digest::digest(&digest::SHA256, credential)
        .as_ref()
        .try_into()
        .expect("SHA-256 has the declared fixed output size")
}

/// Compare fixed-size SHA-256 digests with `subtle`'s constant-time primitive. Parsing and hashing
/// scale with the presented field length, which the caller already knows; stored state and
/// comparison work are independent of the configured credential length.
fn credential_matches(expected_digest: &[u8; digest::SHA256_OUTPUT_LEN], presented: &[u8]) -> bool {
    let presented_digest = credential_digest(presented);
    bool::from(expected_digest.ct_eq(&presented_digest))
}

pub(crate) fn respond_lora_admin_denial(
    stream: &mut TcpStream,
    denial: LoraAdminDenial,
) -> Result<()> {
    match denial {
        LoraAdminDenial::Authentication => respond_error(
            stream,
            401,
            "LoRA administration requires valid bearer credentials",
            "authentication_error",
        ),
        LoraAdminDenial::Disabled => respond_error(
            stream,
            403,
            "LoRA administration is disabled on this listener",
            "permission_error",
        ),
    }
}

/// Seam around adapter administration. Production delegates to `Runner`; card 315's socket tests use
/// a fake so they can prove authorization precedes filesystem and pool access without a checkpoint
/// or device.
pub(crate) trait LoraAdminBackend {
    fn list(&self) -> Vec<(String, usize, usize)>;
    fn hot_load_dir(&self, name: &str, path: &str) -> Result<usize>;
    fn index_of(&self, name: &str) -> Option<usize>;
    fn hot_unload(&self, name: &str) -> Result<usize>;
}

/// Lazy production seam used only after the request head passed admin authorization and its body
/// was accepted. Tests count resolver calls separately from backend calls so an early backend match
/// cannot hide behind an inert fake backend.
pub(crate) trait LoraAdminBackendResolver {
    fn resolve_lora_admin_backend(&self) -> Option<&dyn LoraAdminBackend>;
}

impl LoraAdminBackendResolver for Backend {
    fn resolve_lora_admin_backend(&self) -> Option<&dyn LoraAdminBackend> {
        match self {
            Backend::Decoder(runner) => Some(runner.as_ref()),
            _ => None,
        }
    }
}

impl LoraAdminBackend for Runner {
    fn list(&self) -> Vec<(String, usize, usize)> {
        self.lora_pool_names()
    }

    fn hot_load_dir(&self, name: &str, path: &str) -> Result<usize> {
        Ok(self.hot_load_lora_adapter_dir(name, path)?)
    }

    fn index_of(&self, name: &str) -> Option<usize> {
        self.lora_pool_index_of(name)
    }

    fn hot_unload(&self, name: &str) -> Result<usize> {
        Ok(self.hot_unload_lora_adapter(name)?)
    }
}

#[derive(Deserialize)]
struct LoraHotLoadReq {
    name: String,
    path: String,
}

/// Route one already-authorized `/v1/lora_adapters[/*]` request. Request-head authorization and
/// lazy backend resolution are owned by `http.rs`, so this function cannot move the guard below an
/// adapter operation.
pub(crate) fn handle_lora_admin_with_backend(
    stream: &mut TcpStream,
    admin: Option<&dyn LoraAdminBackend>,
    method: &str,
    path: &str,
    body: &str,
) -> Result<()> {
    if path.contains('?') {
        return match method {
            "GET" | "POST" | "DELETE" => respond_error(
                stream,
                404,
                "unknown lora_adapters path",
                "invalid_request_error",
            ),
            _ => respond_error(stream, 405, "method not allowed", "invalid_request_error"),
        };
    }
    let Some(admin) = admin else {
        return respond_unsupported(
            stream,
            "LoRA adapter hot-load/unload is only supported for a plain decoder model served via the \
             generic (wgpu) continuous-batching engine - see docs/tasks/done/248-lora-adapter-serving.md",
        );
    };
    let name_in_path = path.strip_prefix("/v1/lora_adapters/");

    match (method, name_in_path) {
        ("GET", None) => respond(stream, 200, &list_lora_adapters_json(admin)),
        ("POST", None) => handle_hot_load(stream, admin, body),
        ("DELETE", Some(name)) if !name.is_empty() => handle_hot_unload(stream, admin, name),
        ("GET" | "POST" | "DELETE", _) => respond_error(
            stream,
            404,
            "unknown lora_adapters path",
            "invalid_request_error",
        ),
        _ => respond_error(stream, 405, "method not allowed", "invalid_request_error"),
    }
}

fn list_lora_adapters_json(admin: &dyn LoraAdminBackend) -> String {
    let adapters: Vec<serde_json::Value> = admin
        .list()
        .into_iter()
        .map(|(name, index, in_flight)| {
            serde_json::json!({ "name": name, "index": index, "in_flight": in_flight })
        })
        .collect();
    serde_json::json!({ "object": "list", "data": adapters }).to_string()
}

fn handle_hot_load(stream: &mut TcpStream, admin: &dyn LoraAdminBackend, body: &str) -> Result<()> {
    let req: LoraHotLoadReq = match serde_json::from_str(body) {
        Ok(r) => r,
        Err(e) => {
            return respond_error(stream, 400, e, "invalid_request_error");
        }
    };
    if req.name.is_empty() || req.path.is_empty() {
        return respond_error(
            stream,
            400,
            "both \"name\" and \"path\" must be non-empty",
            "invalid_request_error",
        );
    }
    match admin.hot_load_dir(&req.name, &req.path) {
        Ok(index) => respond(
            stream,
            200,
            &serde_json::json!({ "name": req.name, "index": index }).to_string(),
        ),
        Err(e) => respond_error(stream, 400, format_args!("{e:#}"), "invalid_request_error"),
    }
}

fn handle_hot_unload(
    stream: &mut TcpStream,
    admin: &dyn LoraAdminBackend,
    name: &str,
) -> Result<()> {
    if admin.index_of(name).is_none() {
        return respond_error(
            stream,
            404,
            format_args!("lora adapter {name:?} is not registered"),
            "invalid_request_error",
        );
    }
    match admin.hot_unload(name) {
        Ok(index) => respond(
            stream,
            200,
            &serde_json::json!({ "name": name, "freed_index": index }).to_string(),
        ),
        // The only failure once `name` is confirmed registered is "still in flight" (see
        // `Runner::hot_unload_lora_adapter`). 409 Conflict, not 400: the request is well-formed but cannot
        // be satisfied right now; retry later.
        Err(e) => {
            let body = error_body(409, format_args!("{e:#}"), "conflict_error");
            respond_typed(stream, 409, "application/json", &body)
        }
    }
}
