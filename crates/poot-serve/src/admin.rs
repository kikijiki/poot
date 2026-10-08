//! The LoRA administration surface (spec 248, epic 129 B8) and its trust boundary.
//!
//! Card 315 defines the trust boundary. A loopback listener without an admin key uses a local-process
//! trust model. A non-loopback listener without a key disables every admin route. Setting
//! `POOT_LORA_ADMIN_KEY` on either listener class requires the exact bearer credential. Only admin
//! routes are gated; inference stays unauthenticated. After bounded request-head parsing,
//! authorization runs before body handling, route-shape handling or JSON parsing.
//!
//! The route family is `/v1/lora_adapters`, alongside the existing `/v1/models` naming (see
//! `http.rs`'s `handle` dispatch for the other routes): `GET` lists, `POST {"name", "path"}` hot-loads
//! and `DELETE /v1/lora_adapters/{name}` hot-unloads. Adapters load into the driver's adapter sets,
//! which this server has no engine for yet, so an authorized request of a well-formed shape gets the
//! typed 503 until POOT-589 restores them; a malformed path is 404 and a method no
//! route takes is 405.

use std::net::{IpAddr, TcpStream};

use anyhow::{Result, anyhow, bail};
use ring::digest;
use subtle::ConstantTimeEq;

use crate::http::{AuthorizationHeader, respond_error, respond_no_driver_route};

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

/// Route one already-authorized `/v1/lora_adapters[/*]` request. Request-head authorization is owned
/// by `http.rs`, so this function cannot move the guard below an adapter operation. A request of a
/// shape a route takes is the typed 503 (see the module docs); any other is a 404 or a 405.
pub(crate) fn handle_lora_admin(stream: &mut TcpStream, method: &str, path: &str) -> Result<()> {
    let unknown_path = |stream: &mut TcpStream| {
        respond_error(
            stream,
            404,
            "unknown lora_adapters path",
            "invalid_request_error",
        )
    };
    if path.contains('?') {
        return match method {
            "GET" | "POST" | "DELETE" => unknown_path(stream),
            _ => respond_error(stream, 405, "method not allowed", "invalid_request_error"),
        };
    }
    match (method, path.strip_prefix("/v1/lora_adapters/")) {
        ("GET" | "POST", None) => respond_no_driver_route(stream, "/v1/lora_adapters"),
        ("DELETE", Some(name)) if !name.is_empty() => {
            respond_no_driver_route(stream, "/v1/lora_adapters")
        }
        ("GET" | "POST" | "DELETE", _) => unknown_path(stream),
        _ => respond_error(stream, 405, "method not allowed", "invalid_request_error"),
    }
}
