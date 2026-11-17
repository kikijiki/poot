//! The identity route (POOT-1334): what the server runs on, stated through the same type the benchmark
//! runner states it with.

use super::lifecycle::{json_body, roundtrip, status_line};
use super::*;
use poot_build_identity::BuildIdentity;
use poot_llm::driver::identity::{COMPUTE_DTYPE, RunIdentity};

fn identity() -> RunIdentity {
    RunIdentity {
        build: BuildIdentity {
            git_sha: "0123456789abcdef0123456789abcdef01234567",
            profile: "release",
            features: "",
            rustc_version: "rustc 1.0.0",
        },
        family: "qwen2".to_string(),
        precision: "bf16".to_string(),
        adapter: "Test Adapter".to_string(),
        driver_version: "driver 1.2.3".to_string(),
        checkpoint_digest: format!("blake3:{}", "ab".repeat(32)),
        compute_dtype: COMPUTE_DTYPE,
    }
}

/// `GET /v1/identity` answers the identity the server was started with, with exactly the fields the runner's
/// result lines carry ([`RunIdentity::fields`]) and the values of [`RunIdentity::fields`]. Mutation: leave the
/// device's `adapter` out of [`RunIdentity::fields`] (or rename it); the field-set assertion goes red, and so
/// does the runner's row on the same list.
#[test]
fn the_identity_route_states_the_run_identity_fields() {
    let (mut served, _queue) = served_over(tiny_model(), 1);
    Arc::get_mut(&mut served).expect("the only handle").identity = Some(identity());
    let response = roundtrip(
        &served,
        &test_metrics("identity"),
        "GET",
        "/v1/identity",
        "",
    );
    assert!(status_line(&response).contains("200"), "{response}");
    let body = json_body(&response);
    let mut keys: Vec<&str> = body
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    let mut want: Vec<&str> = identity().fields().iter().map(|&(name, _)| name).collect();
    want.sort_unstable();
    assert_eq!(keys, want);
    assert_eq!(body, identity().to_json());
}

/// A server with no single device (a group of devices) has no identity to state: the route is a typed 404,
/// not a body of empty strings. Mutation: answer `identity.unwrap_or_default()`-style with an empty object;
/// the status assertion goes red.
#[test]
fn a_server_without_a_single_device_has_no_identity_to_state() {
    let (served, _queue) = served_over(tiny_model(), 1);
    let response = roundtrip(
        &served,
        &test_metrics("identity"),
        "GET",
        "/v1/identity",
        "",
    );
    assert!(status_line(&response).contains("404"), "{response}");
    assert_eq!(
        json_body(&response)["error"]["type"],
        "invalid_request_error"
    );
}
