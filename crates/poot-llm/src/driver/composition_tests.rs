//! The driver is a composition of owned components, not one struct holding every field of every concern.

/// The most fields `Driver` has: its owned components and the state it is in.
const MAX_DRIVER_FIELDS: usize = 8;

/// The fields of the struct whose declaration starts with `header` in `source`.
fn fields_of(source: &str, header: &str) -> Vec<String> {
    let start = source
        .find(header)
        .unwrap_or_else(|| panic!("{header} is declared in mod.rs"));
    let body = &source[start + header.len()..];
    let end = body.find("\n}\n").expect("the struct ends");
    body[..end]
        .lines()
        .filter(|line| line.starts_with("    ") && !line.starts_with("     "))
        .map(str::trim)
        .filter(|line| !line.starts_with("//") && line.contains(':'))
        .map(|line| line.split(':').next().expect("a field name").to_string())
        .collect()
}

/// SC-003: `Driver` has at most eight fields. Mutation: inline the prepared-set ledger's fields back into
/// `Driver` (`entries`, `reorders`, `owners`, `retained_bytes`, `residency`, `caps`, `encode_shapes`); this
/// fails naming the count and the fields.
#[test]
fn the_driver_has_at_most_eight_fields() {
    let fields = fields_of(
        include_str!("mod.rs"),
        "pub struct Driver<S: State = Unprepared> {",
    );
    assert!(
        fields.len() >= 4,
        "the guard must read the struct it guards, found {fields:?}"
    );
    assert!(
        fields.len() <= MAX_DRIVER_FIELDS,
        "Driver has {} fields, over {MAX_DRIVER_FIELDS}: {fields:?}. Give the new state to the component that owns it.",
        fields.len()
    );
}
