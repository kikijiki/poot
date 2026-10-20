//! The words the family's source and the public docs use about DeepSeek-V4's packed weights match the
//! loader that exists (POOT-1258 SC-003): `poot_load::safetensors::pack_quantized_linears` reads
//! `.weight_scale`/`.weight_scale_inv` siblings and refuses V4's `.scale` ones, so no document may say a
//! loader joins V4's quantized pairs or that its packed experts are read as stored from a checkpoint.

/// The claims a loader that does not exist would make, as they were written before this row.
const LOADER_CLAIMS: [&str; 3] = [
    "the loader joins a quantized",
    "are read packed, as stored",
    "This family loads with `ModelHandle::load`",
];

/// Mutation: restore the sentence "the loader joins a quantized `.weight`/`.scale` pair into one packed
/// entry" to `names.rs`; the `names.rs` row names the claim it found.
#[test]
fn no_source_or_doc_claims_a_v4_packed_loader() {
    let names = include_str!("../names.rs");
    let models = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../website/docs/develop/models.md"
    ))
    .expect("the models page of the public docs");
    // Only the DeepSeek-V4 section of the page: other families' sections load their checkpoints.
    let section = models
        .split("\n## ")
        .find(|section| section.starts_with("DeepSeek-V4"))
        .expect("the models page has a DeepSeek-V4 section");
    assert!(
        names.contains("No loader produces the packed form"),
        "names.rs must state that no loader builds V4's packed entries"
    );
    for (file, text) in [("names.rs", names), ("develop/models.md", section)] {
        for claim in LOADER_CLAIMS {
            assert!(
                !text.contains(claim),
                "{file} claims a DeepSeek-V4 packed loader: `{claim}`"
            );
        }
    }
}
