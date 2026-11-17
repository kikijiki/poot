//! What a run states about what it ran on (POOT-1334), through the public surface the benchmark runner and the
//! server share: the checkpoint digest follows the stored bytes, and the identity states one fixed list of fields.

use poot_build_identity::BuildIdentity;
use poot_llm::driver::identity::{COMPUTE_DTYPE, RunIdentity, checkpoint_digest};
use poot_models::test_support::weight_map::{DIMS, Projections, qwen2};

/// The checkpoint digest follows the stored bytes: the same store digests the same, a store differing in one
/// weight dtype does not, and the digest names its algorithm. Mutation: hash only the names and shapes (drop
/// the dtype name and the bytes); the stores differ only in a stored dtype and the inequality goes red.
#[test]
fn the_checkpoint_digest_follows_the_stored_bytes() {
    let bf16 = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::BF16);
    let f32_embed = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::F32);
    let again = qwen2(DIMS, Projections::Bf16, poot_tensor::DType::BF16);

    let digest = checkpoint_digest(&bf16.store).unwrap();
    assert!(digest.starts_with("blake3:") && digest.len() == "blake3:".len() + 64);
    assert_eq!(digest, checkpoint_digest(&again.store).unwrap());
    assert_ne!(digest, checkpoint_digest(&f32_embed.store).unwrap());
}

/// The field list and the values agree: every name is stated once, in order, by both [`RunIdentity::fields`]
/// and its JSON. The names are those `result_contract.py` reads (`test_benchmark_serve.py` checks the two lists
/// against each other). Mutation: rename `adapter` in [`RunIdentity::fields`]; the list assertion goes red.
#[test]
fn the_identity_states_exactly_the_named_fields() {
    let identity = RunIdentity {
        build: BuildIdentity {
            git_sha: "a",
            profile: "release",
            features: "",
            rustc_version: "rustc 1",
        },
        family: "a-family".into(),
        precision: "bf16".into(),
        adapter: "adapter".into(),
        driver_version: "driver".into(),
        checkpoint_digest: "blake3:x".into(),
        compute_dtype: COMPUTE_DTYPE,
    };
    let want = [
        "build_sha",
        "build_profile",
        "build_features",
        "rustc_version",
        "family",
        "precision",
        "adapter",
        "driver_version",
        "checkpoint_digest",
        "compute_dtype",
    ];
    let names: Vec<&str> = identity.fields().iter().map(|&(name, _)| name).collect();
    assert_eq!(names, want);
    let json = identity.to_json();
    let mut keys: Vec<&str> = json
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    let mut sorted = want;
    sorted.sort_unstable();
    assert_eq!(keys, sorted);
    assert_eq!(json["adapter"], "adapter");
    assert_eq!(json["build_features"], "");
}
