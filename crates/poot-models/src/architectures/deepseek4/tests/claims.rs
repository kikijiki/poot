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
    // The public pages that could state a V4 loader: the models page (model-neutral) and the feature matrix
    // (the per-model support table). The claims are specific sentences, so the whole page is scanned.
    let page = |path: &str| {
        std::fs::read_to_string(format!(
            "{}/../../website/docs/{path}",
            env!("CARGO_MANIFEST_DIR")
        ))
        .unwrap_or_else(|e| panic!("the public docs page {path}: {e}"))
    };
    let models = page("develop/models.md");
    let matrix = page("reference/feature-matrix.md");
    assert!(
        names.contains("No loader produces the packed form"),
        "names.rs must state that no loader builds V4's packed entries"
    );
    for (file, text) in [
        ("names.rs", names.to_string()),
        ("develop/models.md", models),
        ("reference/feature-matrix.md", matrix),
    ] {
        for claim in LOADER_CLAIMS {
            assert!(
                !text.contains(claim),
                "{file} claims a DeepSeek-V4 packed loader: `{claim}`"
            );
        }
    }
}
