//! The local Hugging Face token, checked against Hugging Face before it is sent to a pod. A stale token (revoked,
//! rotated) makes every download on the pod fail with a 401 even for public repos, so a token that Hugging Face
//! rejects must never be the one the pod receives.

use std::time::Duration;

const BASE: &str = "https://huggingface.co";

/// Why a token could not be vouched for.
#[derive(Debug, thiserror::Error)]
pub(crate) enum WhoamiError {
    /// Hugging Face answered, and the answer is no.
    #[error("Hugging Face rejected it (HTTP {0})")]
    Rejected(u16),
    /// No verdict: the request failed or the answer was not a user record.
    #[error("Hugging Face could not be asked: {0}")]
    Unverified(String),
}

/// The `whoami` endpoint of Hugging Face.
pub(crate) struct Whoami {
    client: reqwest::blocking::Client,
    base: String,
}

impl Whoami {
    pub(crate) fn new() -> anyhow::Result<Self> {
        Self::with_base(BASE)
    }

    /// Build a client against an alternate origin. Host tests serve loopback responses with it.
    pub(crate) fn with_base(base: impl Into<String>) -> anyhow::Result<Self> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(20))
                .build()?,
            base: base.into().trim_end_matches('/').to_owned(),
        })
    }

    /// The account name `token` belongs to.
    pub(crate) fn user(&self, token: &str) -> Result<String, WhoamiError> {
        let response = self
            .client
            .get(format!("{}/api/whoami-v2", self.base))
            .bearer_auth(token)
            .send()
            .map_err(|e| WhoamiError::Unverified(e.to_string()))?;
        let status = response.status();
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(WhoamiError::Rejected(status.as_u16()));
        }
        if !status.is_success() {
            return Err(WhoamiError::Unverified(format!("HTTP {}", status.as_u16())));
        }
        let record: serde_json::Value = response
            .json()
            .map_err(|e| WhoamiError::Unverified(e.to_string()))?;
        record["name"]
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| WhoamiError::Unverified("the answer names no user".into()))
    }
}
