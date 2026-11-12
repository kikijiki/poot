//! The format-list gate (POOT-1539 SC-3): no Rust file outside `poot_quant::test_support::formats` names
//! three or more `WeightFormat` variants in a row.
//!
//! A test that covers "every format" iterates `GGUF_FORMATS` or `SAFETENSORS_FORMATS` (or a filter of
//! them over the descriptor) so that a new format is covered with no edit. A hand-kept list drifts: review
//! round 1 of POOT-1197 found IQ3_S missing from a parity list. The gate reads the workspace sources and
//! lists every comma-separated sequence of variants (an array, a `vec!`) so the next hand list is red. An
//! or-pattern is not a list: a `match` without a wildcard is checked by the compiler and cannot drift.

use std::path::{Path, PathBuf};

/// The sequences of this many variants or more are lists.
const LIST_LENGTH: usize = 3;

/// One `WeightFormat::Variant` occurrence: where it starts and ends, payload included.
struct Occurrence {
    start: usize,
    end: usize,
}

/// The index just past the group opened by `bytes[open]` (`{`, `(` or `[`), or `None` if it never closes.
fn skip_group(bytes: &[u8], open: usize) -> Option<usize> {
    let mut depth = 0usize;
    for (index, &byte) in bytes.iter().enumerate().skip(open) {
        match byte {
            b'{' | b'(' | b'[' => depth += 1,
            b'}' | b')' | b']' => {
                depth -= 1;
                if depth == 0 {
                    return Some(index + 1);
                }
            }
            _ => {}
        }
    }
    None
}

fn occurrences(source: &str) -> Vec<Occurrence> {
    const PREFIX: &str = "WeightFormat::";
    let bytes = source.as_bytes();
    let mut found = Vec::new();
    let mut at = 0;
    while let Some(offset) = source[at..].find(PREFIX) {
        let start = at + offset;
        let mut end = start + PREFIX.len();
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        // A payload: `{ scale }`, `{ groups: GroupMap::Indexed { groups } }`, `{ .. }`.
        let mut after = end;
        while after < bytes.len() && bytes[after].is_ascii_whitespace() {
            after += 1;
        }
        if bytes.get(after) == Some(&b'{')
            && let Some(close) = skip_group(bytes, after)
        {
            end = close;
        }
        found.push(Occurrence { start, end });
        at = end;
    }
    found
}

/// Whether only whitespace and one `,` separate two occurrences.
fn adjacent(source: &str, left: &Occurrence, right: &Occurrence) -> bool {
    // The path that leads to the second variant (`poot_quant::format::`) is not code between them.
    let between = source[left.end..right.start].trim_end_matches(|character: char| {
        character.is_alphanumeric() || character == '_' || character == ':'
    });
    let mut separators = 0;
    for character in between.chars() {
        match character {
            ',' => separators += 1,
            character if character.is_whitespace() => {}
            _ => return false,
        }
    }
    separators <= 1
}

/// `(line, length)` of every sequence of `LIST_LENGTH` or more variants in `source`.
fn lists(source: &str) -> Vec<(usize, usize)> {
    let found = occurrences(source);
    let mut lists = Vec::new();
    let mut run_start = 0;
    for index in 1..=found.len() {
        let continues = index < found.len() && adjacent(source, &found[index - 1], &found[index]);
        if !continues {
            let length = index - run_start;
            if length >= LIST_LENGTH {
                let line = source[..found[run_start].start].matches('\n').count() + 1;
                lists.push((line, length));
            }
            run_start = index;
        }
    }
    lists
}

fn rust_files(directory: &Path, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(directory).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_files(&path, files);
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
}

/// SC-3: no file but the format list names three variants in a row. Mutation: re-add one hand list (an array
/// of three variants) to any test; this goes red naming the file and line.
#[test]
fn no_format_list_exists_outside_the_shared_format_list() {
    let crates = poot_test_util::manifest_dir().join("..");
    let mut files = Vec::new();
    rust_files(&crates, &mut files);
    let allowed = Path::new("poot-quant").join("src/test_support/formats.rs");
    let mut offenders = Vec::new();
    for file in files {
        if file.ends_with(&allowed) || file.ends_with("poot-quant/tests/format_lists.rs") {
            continue;
        }
        let source = std::fs::read_to_string(&file).unwrap();
        for (line, length) in lists(&source) {
            offenders.push(format!(
                "{}:{line}: {length} variants in a row",
                file.strip_prefix(&crates).unwrap().display()
            ));
        }
    }
    assert!(
        offenders.is_empty(),
        "a list of formats belongs in poot_quant::test_support::formats (iterate GGUF_FORMATS or \
         SAFETENSORS_FORMATS, or a filter of them):\n{}",
        offenders.join("\n")
    );
}

/// The scanner itself: it sees a list in an array and in one with payload variants, and nothing in an
/// or-pattern, in two variants or in variants separated by code.
#[test]
fn the_scanner_finds_lists_and_only_lists() {
    let array = "let a = [WeightFormat::Q4_0, WeightFormat::Q8_0, WeightFormat::Mxfp4];";
    assert_eq!(lists(array), [(1, 3)]);
    let or_pattern =
        "match f {\n WeightFormat::F32\n | WeightFormat::F16\n | WeightFormat::Bf16 => 1,\n}";
    assert!(lists(or_pattern).is_empty());
    let payload = "[WeightFormat::E4m3PerChannel { scale: ScaleEncoding::F32 },\n WeightFormat::Awq { group_size: n },\n WeightFormat::Q4_0]";
    assert_eq!(lists(payload), [(1, 3)]);
    let pair = "[WeightFormat::Q4_0, WeightFormat::Q8_0]";
    assert!(lists(pair).is_empty());
    let separated = "[WeightFormat::Q4_0, 1, WeightFormat::Q8_0, 2, WeightFormat::Mxfp4]";
    assert!(lists(separated).is_empty());
    let pathed =
        "[format::WeightFormat::Q4_0, format::WeightFormat::Q8_0, format::WeightFormat::Mxfp4]";
    assert_eq!(lists(pathed), [(1, 3)]);
}
