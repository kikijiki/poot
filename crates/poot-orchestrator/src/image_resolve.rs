//! Digest-pinned image resolution for `exec`.
//!
//! `exec --image` names a mutable tag. It is resolved once, before any state-DB or RunPod side effect,
//! through [`DigestResolver`], and the pod is created from the digest reference
//! ([`image_ref_by_digest`]), so a tag that moves mid-run cannot change the image the run uses. The run
//! row records the digest reference too, so a kept-warm pod is adopted only by the same image.
//!
//! Production [`RegistryV2Resolver`] speaks the OCI/Docker registry v2 HTTP API:
//! anonymous bearer challenge (`WWW-Authenticate`) -> token endpoint -> manifest GET
//! with OCI index and Docker manifest-list `Accept` headers. The digest is the
//! `Docker-Content-Digest` response header; for multi-arch tags that is the INDEX
//! digest, which is what `repo@sha256:...` pins.

use thiserror::Error;

/// Accept headers so a multi-arch tag returns the index/list manifest (digest = index).
pub(crate) const MANIFEST_ACCEPT: &str = "application/vnd.oci.image.index.v1+json, \
     application/vnd.docker.distribution.manifest.list.v2+json, \
     application/vnd.oci.image.manifest.v1+json, \
     application/vnd.docker.distribution.manifest.v2+json";

/// Resolve a mutable image tag reference to an immutable `sha256:<64-hex>` digest.
pub(crate) trait DigestResolver {
    fn resolve_digest(&self, image_tag: &str) -> Result<String, ImageResolveError>;
}

/// Failures beneath the tag-to-digest seam (fail-closed, before DB/RunPod side effects).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub(crate) enum ImageResolveError {
    #[error("invalid image reference {image:?}: {reason}")]
    InvalidRef { image: String, reason: String },
    #[error("registry request failed ({context}): {detail}")]
    Http { context: String, detail: String },
    #[error("registry denied anonymous pull ({context}): {detail}")]
    Unauthorized { context: String, detail: String },
    #[error("manifest response has no Docker-Content-Digest header (status {status})")]
    MissingDigestHeader { status: u16 },
    #[error("manifest Docker-Content-Digest is not a sha256 digest: {value:?}")]
    InvalidDigest { value: String },
    #[error("token response has no bearer token: {detail}")]
    MissingToken { detail: String },
}

/// Split `registry/repo[:tag]` or `registry/repo@sha256:...` into (registry, repo, reference).
///
/// The first path component is the registry when it contains `.` or `:`, or is
/// `localhost`; otherwise the name is on Docker Hub (`docker.io`, official images under `library/`). A `:` is a tag separator only when it appears after the final `/`
/// (so `localhost:5000/app` is host:port, not host + tag). Missing tag defaults to
/// `latest`.
pub(crate) fn parse_image_ref(image: &str) -> Result<(String, String, String), ImageResolveError> {
    let invalid = |reason: &str| ImageResolveError::InvalidRef {
        image: image.to_owned(),
        reason: reason.to_owned(),
    };
    let image = image.trim();
    if image.is_empty() || image.starts_with('@') {
        return Err(invalid("empty or digest-only reference"));
    }
    let (path_part, explicit_digest) = match image.split_once('@') {
        Some((path, digest)) => {
            let bare = digest.strip_prefix("sha256:").unwrap_or("");
            if bare.len() != 64 || !bare.bytes().all(|b| b.is_ascii_hexdigit()) {
                return Err(invalid("malformed @sha256 digest"));
            }
            (path, Some(format!("sha256:{bare}")))
        }
        None => (image, None),
    };
    if path_part.is_empty() {
        return Err(invalid("missing repository path"));
    }
    let last_slash = path_part.rfind('/');
    let tag = match path_part.rfind(':') {
        Some(colon) if last_slash.is_none_or(|slash| colon > slash) => {
            let tag = &path_part[colon + 1..];
            if tag.is_empty() {
                return Err(invalid("empty tag"));
            }
            Some(tag.to_owned())
        }
        _ => None,
    };
    let path = match tag {
        Some(_) => &path_part[..path_part.rfind(':').expect("tag implies a colon")],
        None => path_part,
    };
    let mut parts = path.splitn(2, '/');
    let first = parts.next().unwrap_or_default();
    let rest = parts.next().unwrap_or_default();
    let has_registry = first.contains('.') || first.contains(':') || first == "localhost";
    // Docker's reference grammar: a name with no registry host lives on Docker Hub.
    let (registry, repo) = if has_registry {
        (first, rest.to_owned())
    } else {
        (DOCKER_HUB, path.to_owned())
    };
    if repo.is_empty() {
        return Err(invalid("missing repository name after registry"));
    }
    // Docker Hub keeps its official images under `library/`.
    let repo = if registry == DOCKER_HUB && !repo.contains('/') {
        format!("library/{repo}")
    } else {
        repo
    };
    let reference = explicit_digest
        .or(tag)
        .unwrap_or_else(|| "latest".to_owned());
    Ok((registry.to_owned(), repo, reference))
}

/// `repo[:tag]` + `sha256:...` -> `repo@sha256:...` (strip tag; keep registry host).
pub(crate) fn image_ref_by_digest(
    image_tag: &str,
    digest: &str,
) -> Result<String, ImageResolveError> {
    if !digest.starts_with("sha256:") || digest.len() != "sha256:".len() + 64 {
        return Err(ImageResolveError::InvalidDigest {
            value: digest.to_owned(),
        });
    }
    let (registry, repo, _) = parse_image_ref(image_tag)?;
    Ok(format!("{registry}/{repo}@{digest}"))
}

/// Parse a `WWW-Authenticate: Bearer realm="...",service="...",scope="..."` challenge.
pub(crate) fn parse_bearer_challenge(header: &str) -> Result<BearerChallenge, ImageResolveError> {
    let invalid = |detail: &str| ImageResolveError::Unauthorized {
        context: "WWW-Authenticate".to_owned(),
        detail: detail.to_owned(),
    };
    let header = header.trim();
    let Some(rest) = header.strip_prefix("Bearer ") else {
        return Err(invalid("expected Bearer challenge"));
    };
    let mut realm = None;
    let mut service = None;
    let mut scope = None;
    for part in split_auth_params(rest) {
        let Some((key, value)) = part.split_once('=') else {
            continue;
        };
        let value = value.trim().trim_matches('"').to_owned();
        match key.trim() {
            "realm" => realm = Some(value),
            "service" => service = Some(value),
            "scope" => scope = Some(value),
            _ => {}
        }
    }
    let realm = realm.ok_or_else(|| invalid("challenge missing realm"))?;
    Ok(BearerChallenge {
        realm,
        service,
        scope,
    })
}

/// One `key="value"` parameter list (comma-separated; values may contain commas inside quotes).
fn split_auth_params(rest: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quotes = false;
    for ch in rest.chars() {
        match ch {
            '"' => {
                in_quotes = !in_quotes;
                cur.push(ch);
            }
            ',' if !in_quotes => {
                if !cur.trim().is_empty() {
                    out.push(cur.trim().to_owned());
                }
                cur.clear();
            }
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_owned());
    }
    out
}

/// Anonymous bearer challenge parameters from the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BearerChallenge {
    pub(crate) realm: String,
    pub(crate) service: Option<String>,
    pub(crate) scope: Option<String>,
}

/// Docker Hub names its registry `docker.io` in image references but serves the v2 API elsewhere.
const DOCKER_HUB: &str = "docker.io";
const DOCKER_HUB_API_HOST: &str = "registry-1.docker.io";

/// The host that serves `registry`'s v2 API.
fn api_host(registry: &str) -> &str {
    if registry == DOCKER_HUB {
        DOCKER_HUB_API_HOST
    } else {
        registry
    }
}

/// Resolve `image` (a tag reference) to the digest reference the pod is created from.
pub(crate) fn pin_image(
    resolver: &dyn DigestResolver,
    image: &str,
) -> Result<String, ImageResolveError> {
    let digest = resolver.resolve_digest(image)?;
    image_ref_by_digest(image, &digest)
}

/// Registry v2 digest resolver. Production uses `https://<registry>` from the image
/// ref; host tests point [`RegistryV2Resolver::with_base`] at a loopback fixture
/// (same pattern as `RunPod::with_base`).
pub(crate) struct RegistryV2Resolver {
    client: reqwest::blocking::Client,
    /// When set, replace `https://<registry>` with this origin (e.g. `http://127.0.0.1:PORT`).
    base_override: Option<String>,
}

impl RegistryV2Resolver {
    pub(crate) fn new() -> Result<Self, ImageResolveError> {
        Ok(Self {
            client: reqwest::blocking::Client::builder()
                .timeout(std::time::Duration::from_secs(30))
                .build()
                .map_err(|e| ImageResolveError::Http {
                    context: "build client".to_owned(),
                    detail: e.to_string(),
                })?,
            base_override: None,
        })
    }

    /// Loopback fixture origin; the image ref still names registry/repo/tag for path shape.
    #[cfg(test)]
    pub(crate) fn with_base(base: impl Into<String>) -> Result<Self, ImageResolveError> {
        let mut this = Self::new()?;
        this.base_override = Some(base.into().trim_end_matches('/').to_owned());
        Ok(this)
    }

    fn origin(&self, registry: &str) -> String {
        match &self.base_override {
            Some(base) => base.clone(),
            None => format!("https://{}", api_host(registry)),
        }
    }

    fn manifest_url(&self, registry: &str, repo: &str, reference: &str) -> String {
        format!(
            "{}/v2/{}/manifests/{}",
            self.origin(registry),
            repo.trim_matches('/'),
            reference
        )
    }
}

impl DigestResolver for RegistryV2Resolver {
    fn resolve_digest(&self, image_tag: &str) -> Result<String, ImageResolveError> {
        let (registry, repo, reference) = parse_image_ref(image_tag)?;
        // Already a digest reference: no registry round trip required.
        if reference.starts_with("sha256:") {
            return Ok(reference);
        }
        let url = self.manifest_url(&registry, &repo, &reference);
        let attempt = |auth: Option<&str>| {
            let mut rb = self
                .client
                .get(&url)
                .header(reqwest::header::ACCEPT, MANIFEST_ACCEPT);
            if let Some(token) = auth {
                rb = rb.bearer_auth(token);
            }
            rb.send().map_err(|e| ImageResolveError::Http {
                context: format!("GET {url}"),
                detail: e.to_string(),
            })
        };

        let resp = attempt(None)?;
        let status = resp.status();
        if status.as_u16() == 401 {
            let challenge_header = resp
                .headers()
                .get(reqwest::header::WWW_AUTHENTICATE)
                .and_then(|v| v.to_str().ok())
                .unwrap_or_default()
                .to_owned();
            let challenge = parse_bearer_challenge(&challenge_header)?;
            let token = fetch_anonymous_token(&self.client, &challenge)?;
            let resp = attempt(Some(&token))?;
            return digest_from_manifest_response(resp);
        }
        if !status.is_success() {
            let detail = resp.text().unwrap_or_default();
            if status.as_u16() == 401 || status.as_u16() == 403 {
                return Err(ImageResolveError::Unauthorized {
                    context: format!("GET {url}"),
                    detail,
                });
            }
            return Err(ImageResolveError::Http {
                context: format!("GET {url} ({status})"),
                detail,
            });
        }
        digest_from_manifest_response(resp)
    }
}

/// GET the challenge realm (anonymous) and parse `{"token"|"access_token": "..."}`.
fn fetch_anonymous_token(
    client: &reqwest::blocking::Client,
    challenge: &BearerChallenge,
) -> Result<String, ImageResolveError> {
    let mut rb = client.get(&challenge.realm);
    if let Some(service) = &challenge.service {
        rb = rb.query(&[("service", service.as_str())]);
    }
    if let Some(scope) = &challenge.scope {
        rb = rb.query(&[("scope", scope.as_str())]);
    }
    let resp = rb.send().map_err(|e| ImageResolveError::Http {
        context: format!("GET {}", challenge.realm),
        detail: e.to_string(),
    })?;
    let status = resp.status();
    let text = resp.text().unwrap_or_default();
    if !status.is_success() {
        return Err(ImageResolveError::Unauthorized {
            context: format!("GET {} ({status})", challenge.realm),
            detail: text,
        });
    }
    let json: serde_json::Value =
        serde_json::from_str(&text).map_err(|e| ImageResolveError::MissingToken {
            detail: format!("token body is not JSON: {e}; body={text}"),
        })?;
    let token = json
        .get("token")
        .or_else(|| json.get("access_token"))
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .ok_or_else(|| ImageResolveError::MissingToken {
            detail: format!("no token/access_token field; body={text}"),
        })?;
    Ok(token)
}

/// Read `Docker-Content-Digest` from a successful manifest response.
fn digest_from_manifest_response(
    resp: reqwest::blocking::Response,
) -> Result<String, ImageResolveError> {
    let status = resp.status();
    if !status.is_success() {
        let detail = resp.text().unwrap_or_default();
        return Err(ImageResolveError::Http {
            context: format!("manifest status {status}"),
            detail,
        });
    }
    let value = resp
        .headers()
        .get("docker-content-digest")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
        .trim()
        .to_owned();
    if value.is_empty() {
        return Err(ImageResolveError::MissingDigestHeader {
            status: status.as_u16(),
        });
    }
    let bare = value.strip_prefix("sha256:").unwrap_or("");
    if bare.len() != 64 || !bare.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(ImageResolveError::InvalidDigest { value });
    }
    Ok(format!("sha256:{}", bare.to_ascii_lowercase()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    const DIGEST: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

    #[test]
    fn parse_image_ref_splits_registry_repo_and_tag() {
        assert_eq!(
            parse_image_ref("ghcr.io/kikijiki/poot-bench:latest").unwrap(),
            (
                "ghcr.io".to_owned(),
                "kikijiki/poot-bench".to_owned(),
                "latest".to_owned()
            )
        );
        assert_eq!(
            parse_image_ref("ghcr.io/kikijiki/poot-bench").unwrap(),
            (
                "ghcr.io".to_owned(),
                "kikijiki/poot-bench".to_owned(),
                "latest".to_owned()
            )
        );
        // Port colon is not a tag separator.
        assert_eq!(
            parse_image_ref("localhost:5000/team/app:v1").unwrap(),
            (
                "localhost:5000".to_owned(),
                "team/app".to_owned(),
                "v1".to_owned()
            )
        );
        // No registry host: Docker Hub, official images under `library/`.
        assert_eq!(
            parse_image_ref("runpod/pytorch:1.0.2").unwrap(),
            (
                "docker.io".to_owned(),
                "runpod/pytorch".to_owned(),
                "1.0.2".to_owned()
            )
        );
        assert_eq!(
            parse_image_ref("ubuntu").unwrap(),
            (
                "docker.io".to_owned(),
                "library/ubuntu".to_owned(),
                "latest".to_owned()
            )
        );
        assert_eq!(
            parse_image_ref("docker.io/ubuntu:24.04").unwrap(),
            (
                "docker.io".to_owned(),
                "library/ubuntu".to_owned(),
                "24.04".to_owned()
            )
        );
        assert!(matches!(
            parse_image_ref("ghcr.io/"),
            Err(ImageResolveError::InvalidRef { .. })
        ));
    }

    #[test]
    fn image_ref_by_digest_strips_tag_and_keeps_registry() {
        assert_eq!(
            image_ref_by_digest("ghcr.io/kikijiki/poot-bench:latest", DIGEST).unwrap(),
            format!("ghcr.io/kikijiki/poot-bench@{DIGEST}")
        );
        assert_eq!(
            image_ref_by_digest("example.com/img:tag", DIGEST).unwrap(),
            format!("example.com/img@{DIGEST}")
        );
        assert!(image_ref_by_digest("ghcr.io/a/b:tag", "sha256:zz").is_err());
    }

    #[test]
    fn docker_hub_references_resolve_through_the_hub_api_host_and_library_namespace() {
        assert_eq!(api_host("docker.io"), "registry-1.docker.io");
        assert_eq!(api_host("ghcr.io"), "ghcr.io");
        assert_eq!(
            image_ref_by_digest("ubuntu:24.04", DIGEST).unwrap(),
            format!("docker.io/library/ubuntu@{DIGEST}")
        );
        let resolver = RegistryV2Resolver::new().unwrap();
        let (registry, repo, reference) = parse_image_ref("ubuntu:24.04").unwrap();
        assert_eq!(
            resolver.manifest_url(&registry, &repo, &reference),
            "https://registry-1.docker.io/v2/library/ubuntu/manifests/24.04"
        );
    }

    #[test]
    fn bearer_challenge_parses_realm_service_scope() {
        let challenge = parse_bearer_challenge(
            r#"Bearer realm="https://ghcr.io/token",service="ghcr.io",scope="repository:kikijiki/poot-bench:pull""#,
        )
        .unwrap();
        assert_eq!(challenge.realm, "https://ghcr.io/token");
        assert_eq!(challenge.service.as_deref(), Some("ghcr.io"));
        assert_eq!(
            challenge.scope.as_deref(),
            Some("repository:kikijiki/poot-bench:pull")
        );
        assert!(parse_bearer_challenge("Basic realm=x").is_err());
        assert!(parse_bearer_challenge("Bearer realm=\"https://t\"").is_ok());
    }

    /// Multi-connection loopback: 401 challenge -> token -> 200 manifest with digest.
    fn serve_registry_sequence() -> (String, mpsc::Receiver<Vec<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback fixture");
        let addr = listener.local_addr().expect("fixture address");
        let (tx, rx) = mpsc::channel();
        let token_url = format!("http://{addr}/token");
        thread::spawn(move || {
            let mut requests = Vec::new();
            for _ in 0..3 {
                let (mut stream, _) = listener.accept().expect("accept fixture connection");
                let mut buf = Vec::new();
                let mut chunk = [0u8; 4096];
                loop {
                    let n = stream.read(&mut chunk).expect("read fixture request");
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                        // Read Content-Length body if present.
                        let head = String::from_utf8_lossy(&buf).into_owned();
                        let content_length = head
                            .lines()
                            .find_map(|l| {
                                let (k, v) = l.split_once(':')?;
                                k.eq_ignore_ascii_case("content-length")
                                    .then(|| v.trim().parse::<usize>().ok())
                                    .flatten()
                            })
                            .unwrap_or(0);
                        let head_len = head.find("\r\n\r\n").map(|i| i + 4).unwrap_or(0);
                        while buf.len() < head_len + content_length {
                            let n = stream.read(&mut chunk).expect("read body");
                            if n == 0 {
                                break;
                            }
                            buf.extend_from_slice(&chunk[..n]);
                        }
                        break;
                    }
                    if n == 0 {
                        break;
                    }
                }
                let request = String::from_utf8_lossy(&buf).into_owned();
                let is_token = request.starts_with("GET /token");
                let has_auth = request
                    .to_ascii_lowercase()
                    .contains("authorization: bearer");
                requests.push(request);
                let response = if is_token {
                    let body = r#"{"token":"fixture-bearer-token"}"#;
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else if has_auth {
                    let body = r#"{"schemaVersion":2,"mediaType":"application/vnd.oci.image.index.v1+json"}"#;
                    format!(
                        "HTTP/1.1 200 OK\r\nContent-Type: application/vnd.oci.image.index.v1+json\r\nDocker-Content-Digest: {DIGEST}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else {
                    format!(
                        "HTTP/1.1 401 Unauthorized\r\nWWW-Authenticate: Bearer realm=\"{token_url}\",service=\"fixture.registry\",scope=\"repository:kikijiki/poot-bench:pull\"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    )
                };
                stream
                    .write_all(response.as_bytes())
                    .expect("write fixture");
            }
            let _ = tx.send(requests);
        });
        (format!("http://{addr}"), rx)
    }

    #[test]
    fn registry_resolver_walks_challenge_token_manifest_and_reads_index_digest() {
        let (base, requests) = serve_registry_sequence();
        let resolver = RegistryV2Resolver::with_base(base).expect("fixture resolver");
        let digest = resolver
            .resolve_digest("ghcr.io/kikijiki/poot-bench:latest")
            .expect("resolve via fixture");
        assert_eq!(digest, DIGEST);

        let requests = requests.recv().expect("three requests observed");
        assert_eq!(
            requests.len(),
            3,
            "challenge then token then manifest: {requests:?}"
        );

        let head = |r: &str| {
            r.split("\r\n\r\n")
                .next()
                .unwrap_or_default()
                .to_ascii_lowercase()
        };
        let h0 = head(&requests[0]);
        assert!(
            h0.starts_with("get /v2/kikijiki/poot-bench/manifests/latest "),
            "{h0}"
        );
        assert!(
            h0.contains("accept:") && h0.contains("application/vnd.oci.image.index.v1+json"),
            "manifest Accept must request OCI index: {h0}"
        );
        assert!(
            h0.contains("application/vnd.docker.distribution.manifest.list.v2+json"),
            "manifest Accept must request Docker manifest list: {h0}"
        );
        assert!(
            !h0.contains("authorization:"),
            "first manifest GET is anonymous"
        );

        let h1 = head(&requests[1]);
        assert!(h1.starts_with("get /token?"), "{h1}");
        assert!(
            h1.contains("service=fixture.registry"),
            "token query carries service: {h1}"
        );
        // reqwest percent-encodes `:` `/` in the scope value (case of hex digits varies).
        assert!(
            h1.contains("scope=repository")
                && h1.contains("kikijiki")
                && h1.contains("poot-bench")
                && h1.contains("pull"),
            "token query carries scope: {h1}"
        );
        assert!(!h1.contains("authorization:"), "token fetch is anonymous");

        let h2 = head(&requests[2]);
        assert!(
            h2.starts_with("get /v2/kikijiki/poot-bench/manifests/latest "),
            "{h2}"
        );
        assert!(
            h2.contains("authorization: bearer fixture-bearer-token"),
            "second manifest GET carries the token: {h2}"
        );
        assert!(
            h2.contains("application/vnd.oci.image.index.v1+json"),
            "retry keeps OCI Accept headers: {h2}"
        );
    }

    #[test]
    fn resolve_rejects_manifest_without_digest_header() {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf);
            let body = "{}";
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            let _ = stream.write_all(resp.as_bytes());
        });
        let resolver = RegistryV2Resolver::with_base(format!("http://{addr}")).unwrap();
        let err = resolver
            .resolve_digest("ghcr.io/kikijiki/poot-bench:latest")
            .unwrap_err();
        assert!(
            matches!(err, ImageResolveError::MissingDigestHeader { status: 200 }),
            "got {err:?}"
        );
    }
}
