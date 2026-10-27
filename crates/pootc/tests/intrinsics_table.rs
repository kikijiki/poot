//! The dispatch-axis intrinsics the importer recognizes are exactly the ones `poot_kernel_intrinsics` exports
//! (R2-L03-013): the importer's table is `poot_kernel_ir::naming::AXIS_INTRINSICS`, and this reads the intrinsics
//! crate's own source, so an axis function added or removed on one side and not the other fails here by name.

use std::collections::BTreeSet;

fn intrinsics_source() -> String {
    let manifest = poot_test_util::manifest_dir();
    std::fs::read_to_string(manifest.join("../poot-kernel-intrinsics/src/lib.rs"))
        .expect("read the intrinsics crate source")
}

/// Mutation: delete `group_index_z,` from the intrinsics crate's `axis_intrinsics!` list (or an entry from `AXIS_INTRINSICS`); the
/// two name sets differ and the failure names the one that is missing.
#[test]
fn the_importer_recognizes_exactly_the_axis_functions_the_intrinsics_crate_exports() {
    // The nine are declared by one `axis_intrinsics! { .. }` list: its entries are the lines `name,` after the
    // doc comments.
    let source = intrinsics_source();
    let list = source
        .split("axis_intrinsics! {")
        .nth(1)
        .and_then(|rest| rest.split("\n}\n").next())
        .expect("the axis_intrinsics! list");
    let exported: BTreeSet<String> = list
        .lines()
        .map(str::trim)
        .filter(|line| !line.starts_with("//"))
        .filter_map(|line| line.strip_suffix(','))
        .map(str::to_string)
        .collect();
    let recognized: BTreeSet<String> = poot_kernel_ir::naming::AXIS_INTRINSICS
        .iter()
        .map(|(name, _)| name.to_string())
        .collect();
    assert_eq!(
        exported, recognized,
        "the intrinsics crate exports {exported:?}, the importer recognizes {recognized:?}"
    );
    assert_eq!(recognized.len(), 9, "three axes of three kinds");
}
