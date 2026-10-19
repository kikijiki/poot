//! POOT-1220 SC-001: a contract type either has a production caller or is not defined. The inventory reads the
//! workspace sources and fails on a test-only survivor of the contracts the architecture review found without a
//! production producer (AP-15).
//!
//! "Production text" is a file without its trailing `#[cfg(test)]` module and without any item gated on
//! `cfg(test)` or `feature = "test-support"`: an item only a test build compiles is not a production caller.
//!
//! Mutation: re-add `prepare_target_graph` behind `#[cfg(test)]` anywhere in the workspace; `deleted_contracts_stay_deleted`
//! goes red naming the file.

use std::path::{Path, PathBuf};

/// Every `.rs` file under the workspace's `crates/*/{src,tests,benches,examples}`, with its text.
fn workspace_sources() -> Vec<(PathBuf, String)> {
    fn walk(dir: &Path, out: &mut Vec<(PathBuf, String)>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let text = std::fs::read_to_string(&path).unwrap();
                out.push((path, text));
            }
        }
    }
    let crates = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut out = Vec::new();
    for krate in std::fs::read_dir(crates).unwrap().flatten() {
        for part in ["src", "tests", "benches", "examples"] {
            walk(&krate.path().join(part), &mut out);
        }
    }
    out
}

/// Whether `text` defines an item called `name` (`fn name`, `struct name`, ...), not merely mentions it.
fn defines(text: &str, name: &str) -> bool {
    ["fn", "struct", "enum", "trait", "type", "const", "static"]
        .iter()
        .any(|keyword| {
            let head = format!("{keyword} {name}");
            text.match_indices(&head).any(|(at, _)| {
                !text[at + head.len()..].starts_with(|c: char| c.is_alphanumeric() || c == '_')
            })
        })
}

/// Whether an attribute line gates the item after it to a test build.
fn is_test_gate(line: &str) -> bool {
    let line = line.trim();
    line.starts_with("#[cfg(") && (line.contains("test") || line.contains("test-support"))
        // `not(any(test, ...))` gates the production twin of a test item: that one is production text.
        && !line.contains("not(")
}

/// `text` without its trailing `#[cfg(test)]` module and without items gated on a test build.
fn production_text(text: &str) -> String {
    let mut out = String::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() == "#[cfg(test)]" && line.starts_with("#[cfg(test)]") {
            // A module-level test block: the rest of the file by this workspace's convention.
            break;
        }
        if !is_test_gate(line) {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        // Skip the gated item: further attribute lines, then everything to the end of its braces or its `;`.
        let mut depth = 0i32;
        let mut seen_body = false;
        while let Some(next) = lines.peek() {
            let trimmed = next.trim_start();
            if !seen_body && (trimmed.starts_with("#[") || trimmed.starts_with("///")) {
                lines.next();
                continue;
            }
            let next = lines.next().unwrap();
            depth += next.matches('{').count() as i32 - next.matches('}').count() as i32;
            seen_body |= next.contains('{');
            if (depth <= 0 && seen_body) || (!seen_body && next.trim_end().ends_with(';')) {
                break;
            }
        }
    }
    out
}

/// Whether a non-comment line of production text calls `needle` (and is not its definition).
fn called_in_production(sources: &[(PathBuf, String)], needle: &str) -> bool {
    sources.iter().any(|(_, text)| {
        production_text(text).lines().any(|line| {
            !line.trim_start().starts_with("//")
                && line.contains(needle)
                && !line.trim_start().starts_with("pub fn ")
                && !line.contains(&format!("fn {needle}"))
        })
    })
}

#[test]
fn production_text_ignores_items_only_a_test_build_compiles() {
    let text = "\
pub fn kept() { used(); }

#[cfg(any(test, feature = \"test-support\"))]
pub fn only_for_tests() {
    gated_call();
}

#[cfg(not(any(test, feature = \"test-support\")))]
pub(crate) fn production_twin() { twin_call(); }

#[cfg(test)]
mod tests {
    fn t() { tail_call(); }
}
";
    let production = production_text(text);
    assert!(production.contains("used()"), "{production}");
    assert!(production.contains("twin_call()"), "{production}");
    assert!(!production.contains("gated_call()"), "{production}");
    assert!(!production.contains("tail_call()"), "{production}");
}

/// Contracts the review found without a production producer. None is defined anywhere in the workspace, test
/// code included: a definition that only tests reach is dead code, not a contract.
#[test]
fn deleted_contracts_stay_deleted() {
    const DELETED: &[(&str, &str)] = &[
        (
            "prepare_target_graph",
            "a second, test-only preparation pipeline; tests call `compile`",
        ),
        (
            "select_packed_kernel",
            "the probe generation behind the dropped `PackedLowering` refusal",
        ),
        ("journal_donated_updates", "rollback journal"),
        ("DonationFootprint", "rollback journal"),
        ("NativeStateTransactionPlan", "rollback journal"),
        ("plan_native_state_transaction", "rollback journal"),
        ("UnrecoverableDonation", "rollback journal error"),
        (
            "validate_graph_execution",
            "tensor-bound executor gate, no caller",
        ),
        (
            "validate_wgpu_tensor_execution",
            "tensor-bound executor gate, no caller",
        ),
        ("EvalBudget", "oracle budget, only tests set a finite one"),
        ("max_dispatch_work", "no device runtime calibrates it"),
        ("SerialWork", "only `max_dispatch_work` read it"),
        ("SerialLoop", "only `max_dispatch_work` read it"),
        ("tokens_within", "only `max_dispatch_work` read it"),
        (
            "ScanPhase",
            "scan token ranges existed to fit `max_dispatch_work`",
        ),
        (
            "Submission",
            "`Eager` had no producer; the engine refused it",
        ),
    ];
    let sources = workspace_sources();
    assert!(
        sources.len() > 100,
        "the inventory walked {} files; the workspace layout changed",
        sources.len()
    );
    for (name, why) in DELETED {
        for (path, text) in &sources {
            assert!(
                !defines(text, name),
                "{name} ({why}) is defined again in {}",
                path.display()
            );
        }
    }
}

/// The packed graph builders are fixtures with no production caller (a model declares dense weights and
/// `bind_packed_weights` rewrites them), so they live only in the IR crate's `test_support` fixtures, never among
/// the `ops` a tracer composes.
#[test]
fn packed_graph_builders_live_only_in_the_fixture_module() {
    const FIXTURE: &str = "poot-graph-ir/src/test_support/packed_graphs.rs";
    let sources = workspace_sources();
    for name in [
        "packed_linear",
        "packed_embedding",
        "packed_block_diagonal_linear",
        "packed_indexed_linear",
        "PackedExpertStack",
    ] {
        let homes: Vec<_> = sources
            .iter()
            .filter(|(_, text)| defines(text, name))
            .map(|(path, _)| path.to_string_lossy().replace('\\', "/"))
            .filter(|path| !path.ends_with(FIXTURE))
            .collect();
        // `poot-load` has an unrelated test helper of the same name over a weight store.
        let homes: Vec<_> = homes.iter().filter(|p| !p.contains("poot-load/")).collect();
        assert!(homes.is_empty(), "{name} is also defined in {homes:?}");
    }
}

/// Contracts that survive have a production call site (code a default build compiles).
#[test]
fn surviving_contracts_have_a_production_call_site() {
    const CALLED: &[(&str, &str)] = &[
        (
            "admit_packed(",
            "the packed lowering table, consulted by the planner",
        ),
        ("PACKED_LOWERING", "the packed lowering table"),
        (
            "flash_attention_capped(",
            "the attention matcher `compile` runs, which yields the declines",
        ),
        (
            "rope_fusion(",
            "the rope matcher `compile` runs, which yields the declines",
        ),
        ("StorageGap::F16PackedLanes", "storage analysis"),
        ("StorageGap::E4m3PackedLanes", "storage analysis"),
        ("StorageGap::VpuDtype", "the Intel NPU planner"),
        (
            ".allocation(",
            "the oracle walk reports each materialization to its observer",
        ),
        (
            ".published(",
            "the oracle walk reports each publication to its observer",
        ),
    ];
    let sources = workspace_sources();
    for (needle, why) in CALLED {
        assert!(
            called_in_production(&sources, needle),
            "{needle} ({why}) has no call site outside test-gated code"
        );
    }
}
