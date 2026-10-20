//! Crate dependency direction (POOT-755, R463-004/006/017): the workspace's crates form a declared ranking, and
//! every manifest edge points strictly down it.
//!
//! The check runs `cargo metadata --locked --offline` over the real workspace and enforces six rules:
//!
//! 1. [`RANKS`] and the workspace members agree: a member without a row, or a row naming no member, fails, so a
//!    new crate declares its place.
//! 2. Every normal and build edge between members goes to a strictly lower rank.
//! 3. Every dev edge goes to a strictly lower rank too. There is no exception list: a test that needs a crate
//!    above its own moves up to that crate's layer. (A crate's dev-dependency on itself, the idiom that turns
//!    on its own features in tests, is not an edge between two crates and is skipped.)
//! 4. No edge in [`FORBIDDEN_NORMAL`]: executors and devices never depend on the CPU oracle (ruling E6).
//! 5. Every external dependency has one requirement string across all members (members inherit it from
//!    `[workspace.dependencies]`), and every lockfile name with more than one version is a [`DUPLICATE_ALLOW`]
//!    row naming what pulls the extra version; a row that is no longer duplicated fails, so the list shrinks
//!    when an upstream upgrade lands.
//! 6. No member declares a test-only feature other than the one ADR-0113 sanctions (`test-support`): a
//!    `test-hooks`-style feature puts test-only API into the production crate behind a switch (ruling XC2-06).
//!
//! `cargo` and the manifest directory are read at run time (`CARGO`, `CARGO_MANIFEST_DIR`), never with `env!`:
//! the compile cache reuses compile-time values across worktrees.
//!
//! Prior art for a declared rank table checked against the build graph, retrieved 2026-10-11:
//! - cargo-deny's `bans` check (`multiple-versions`, `skip`): reports duplicate crate versions and
//!   skips named ones, which rule 5 mirrors with a reason per row; it has no notion of internal layering.
//!   <https://embarkstudios.github.io/cargo-deny/checks/bans/cfg.html>
//! - cargo-hakari's workspace-hack crate: unifies external feature sets and versions through one crate every
//!   member depends on; it solves build-time duplication, not direction, and adds an edge to every member.
//!   <https://docs.rs/cargo-hakari/latest/cargo_hakari/>
//! - Bazel package visibility: a dependency is legal only when the target's `visibility` lists the dependent,
//!   enforced by the build; the rank table is the one-axis version of it (a dependent may see only lower rows).
//!   <https://bazel.build/concepts/visibility>
//!
//! MLIR/LLVM layering docs were looked for and not found to state a layering rule; none is cited.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::process::Command;

use serde_json::Value;

/// Every workspace member and its rank. An edge of any kind (normal, build, dev) must point to a strictly
/// lower rank. The rank numbers are this table's, not the layer numbers of the target architecture: they are
/// the order a crate may be depended upon in, so a test that needs a higher crate belongs in that crate.
const RANKS: &[(&str, u8)] = &[
    // 0: leaf vocabulary and standalone tools
    ("poot-target", 0),
    ("poot-kernel-ir", 0),
    ("poot-kernel-intrinsics", 0),
    ("poot-orchestrator", 0),
    ("poot-tensor", 0),
    ("poot-text", 0),
    // 1: shared runtime helpers, test helpers, kernel attributes
    ("poot-runtime-common", 1),
    ("poot-test-util", 1),
    ("poot-kernel-attr", 1),
    // 2: storage formats, code generation, the profiling reports (they key on runtime-common's counters)
    ("poot-quant", 2),
    ("poot-codegen", 2),
    ("poot-profile", 2),
    // 3: the weight vocabulary and map (storage formats and tensors only; graph IR and everything above read it)
    ("poot-weights", 3),
    // 4: graph IR, the checkpoint reader (it depends on poot-quant and poot-tensor only, so it sits just above
    // them; FORBIDDEN_NORMAL keeps the oracle, executors and devices off it) and the runtimes whose tests
    // compile kernels
    ("poot-graph-ir", 4),
    ("poot-load", 4),
    ("poot-wgpu-runtime", 4),
    ("poot-rocm-runtime", 4),
    ("poot-l0-runtime", 4),
    ("poot-ptx-runtime", 4),
    // 5: the CPU oracle and the raw Vulkan runtime
    ("poot-eval", 5),
    ("poot-vulkan-runtime", 5),
    // 6: the kernel generator (its tests compile and run generated bodies)
    ("poot-kernelgen", 6),
    // 7: the compiler (its tests check passes against the oracle)
    ("poot-graph-plan", 7),
    // 8: the executor contract and the import compiler
    ("poot-executor", 8),
    ("pootc", 8),
    // 9: the dev-only executor parity fixtures (ruling XC2-06: the device crates dev-depend on them)
    ("poot-executor-parity", 9),
    // 10: devices
    ("poot-wgpu-device", 10),
    ("poot-rocm-device", 10),
    ("poot-ptx-device", 10),
    ("poot-vulkan-device", 10),
    ("poot-npu", 10),
    // 11: models; 12: the driver; 13: products
    ("poot-models", 11),
    ("poot-llm", 12),
    ("poot-serve", 13),
    ("poot-bench-runner", 13),
];

/// Edges the ranks allow but the architecture forbids: `(dependents, dependency, reason)`.
const FORBIDDEN_NORMAL: &[(&[&str], &str, &str)] = &[
    (
        &[
            "poot-executor",
            "poot-wgpu-device",
            "poot-rocm-device",
            "poot-ptx-device",
            "poot-vulkan-device",
            "poot-npu",
            "poot-llm",
        ],
        "poot-eval",
        "executors, devices and the driver never run the CPU oracle: the host sampler is a plain exact function \
         (ruling E6, ADR-0102 decision 5, POOT-1222)",
    ),
    (
        &[
            "poot-eval",
            "poot-executor",
            "poot-wgpu-device",
            "poot-rocm-device",
            "poot-ptx-device",
            "poot-vulkan-device",
            "poot-npu",
        ],
        "poot-load",
        "engine crates read weights through poot-quant's WeightStore; checkpoint-source vocabulary stays in \
         the loader (R463-004, ADR-0103)",
    ),
    (
        &["poot-vulkan-runtime"],
        "poot-wgpu-runtime",
        "the two SPIR-V runtimes are siblings: what they share lives in poot-runtime-common, so a raw Vulkan \
         build never links wgpu (POOT-1231)",
    ),
];

/// Lockfile names allowed more than one version, each with what pulls the extra one. A duplicate with no row
/// fails; a row whose name is no longer duplicated fails.
const DUPLICATE_ALLOW: &[(&str, &str)] = &[
    (
        "base64",
        "tokenizers 0.20 (spm_precompiled) pins 0.13; poot-serve uses 0.22",
    ),
    (
        "foldhash",
        "object 0.36 (hashbrown 0.15) vs wgpu 29 (hashbrown 0.16)",
    ),
    (
        "getrandom",
        "ring 0.17 uses 0.2; reqwest 0.12 (quinn) uses 0.3 (poot-orchestrator)",
    ),
    (
        "hashbrown",
        "rusqlite 0.32 (hashlink), object 0.36, wgpu 29 and indexmap each pin their own line",
    ),
    (
        "itertools",
        "criterion 0.5 uses 0.10; tokenizers 0.20 uses 0.11 and 0.12",
    ),
    (
        "jni-sys",
        "wgpu-hal 29's ndk-sys 0.6 uses 0.3; a second dependent uses 0.4 (Android targets only)",
    ),
    ("libloading", "ash 0.38 uses 0.8; cudarc 0.19 uses 0.9"),
    (
        "rand",
        "tokenizers 0.20 uses 0.8; reqwest 0.12 (quinn-proto) uses 0.9",
    ),
    (
        "rand_chacha",
        "follows rand: tokenizers 0.20 uses 0.3; quinn-proto uses 0.9",
    ),
    (
        "rand_core",
        "follows rand: tokenizers 0.20 uses 0.6; quinn-proto uses 0.9",
    ),
    (
        "rustc-hash",
        "wgpu 29's naga uses 1.x; reqwest 0.12 (quinn) uses 2.x",
    ),
    (
        "thiserror",
        "tokenizers 0.20 uses 1.x; the workspace requirement is 2.0",
    ),
    ("thiserror-impl", "follows thiserror"),
    // The Windows crates compile only for Windows targets but are in the lockfile, so each major line of
    // the family that two dependents pin differently is listed. sysinfo 0.32 pins windows 0.57, wgpu-hal 29
    // pins windows 0.62, ring 0.17 pins windows-sys 0.52, tokenizers' indicatif/console 0.59, reqwest's
    // quinn-udp 0.60 and ctrlc 0.61.
    (
        "windows",
        "sysinfo 0.32 uses 0.57; wgpu-hal 29 uses 0.62 (Windows only)",
    ),
    ("windows-core", "follows windows (Windows only)"),
    ("windows-implement", "follows windows (Windows only)"),
    ("windows-interface", "follows windows (Windows only)"),
    ("windows-result", "follows windows (Windows only)"),
    (
        "windows-sys",
        "ring, console, quinn-udp and ctrlc each pin a different line (Windows only)",
    ),
    ("windows-targets", "follows windows-sys (Windows only)"),
    (
        "windows_aarch64_gnullvm",
        "follows windows-targets (Windows only)",
    ),
    (
        "windows_aarch64_msvc",
        "follows windows-targets (Windows only)",
    ),
    ("windows_i686_gnu", "follows windows-targets (Windows only)"),
    (
        "windows_i686_gnullvm",
        "follows windows-targets (Windows only)",
    ),
    (
        "windows_i686_msvc",
        "follows windows-targets (Windows only)",
    ),
    (
        "windows_x86_64_gnu",
        "follows windows-targets (Windows only)",
    ),
    (
        "windows_x86_64_gnullvm",
        "follows windows-targets (Windows only)",
    ),
    (
        "windows_x86_64_msvc",
        "follows windows-targets (Windows only)",
    ),
];

/// How one member depends on another.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Kind {
    Normal,
    Build,
    Dev,
}

/// One declared dependency of a workspace member.
#[derive(Clone, Debug)]
struct Dep {
    from: String,
    to: String,
    kind: Kind,
    requirement: String,
}

/// What the rules read from `cargo metadata`.
#[derive(Debug)]
struct Graph {
    members: BTreeSet<String>,
    deps: Vec<Dep>,
    /// Every package in the lockfile (members included) and the versions it appears at.
    versions: BTreeMap<String, BTreeSet<String>>,
    /// The feature names each member declares.
    features: BTreeMap<String, BTreeSet<String>>,
}

/// The tables the rules check a [`Graph`] against.
struct Policy<'a> {
    ranks: &'a [(&'a str, u8)],
    forbidden: &'a [(&'a [&'a str], &'a str, &'a str)],
    duplicate_allow: &'a [(&'a str, &'a str)],
}

const POLICY: Policy<'static> = Policy {
    ranks: RANKS,
    forbidden: FORBIDDEN_NORMAL,
    duplicate_allow: DUPLICATE_ALLOW,
};

/// Parse `cargo metadata` output (with dependency resolution, for the lockfile's version set).
fn parse(metadata: &Value) -> Graph {
    let packages = metadata["packages"].as_array().expect("packages");
    let member_ids: BTreeSet<&str> = metadata["workspace_members"]
        .as_array()
        .expect("workspace_members")
        .iter()
        .map(|id| id.as_str().expect("member id"))
        .collect();
    let mut members = BTreeSet::new();
    let mut deps = Vec::new();
    let mut versions: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    let mut features: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for package in packages {
        let name = package["name"].as_str().expect("name").to_string();
        versions
            .entry(name.clone())
            .or_default()
            .insert(package["version"].as_str().expect("version").to_string());
        if !member_ids.contains(package["id"].as_str().expect("id")) {
            continue;
        }
        members.insert(name.clone());
        features.insert(
            name.clone(),
            package["features"]
                .as_object()
                .expect("features")
                .keys()
                .cloned()
                .collect(),
        );
        for dep in package["dependencies"].as_array().expect("dependencies") {
            let kind = match dep["kind"].as_str() {
                None => Kind::Normal,
                Some("build") => Kind::Build,
                Some("dev") => Kind::Dev,
                Some(other) => panic!("unknown dependency kind {other:?}"),
            };
            deps.push(Dep {
                from: name.clone(),
                to: dep["name"].as_str().expect("dep name").to_string(),
                kind,
                requirement: dep["req"].as_str().expect("dep req").to_string(),
            });
        }
    }
    Graph {
        members,
        deps,
        versions,
        features,
    }
}

/// The one test-only feature name the workspace sanctions (ADR-0113): an owner's own fixtures and observers
/// behind `test-support`, enabled only from dev edges. Any other test-flavoured name is rule 6's finding.
const SANCTIONED_TEST_FEATURE: &str = "test-support";

/// Whether `name` is a test-only feature switch other than [`SANCTIONED_TEST_FEATURE`]: `test-hooks`,
/// `testing`, `test-utils`, `for-tests` and the like.
fn is_unsanctioned_test_feature(name: &str) -> bool {
    name != SANCTIONED_TEST_FEATURE
        && (name.starts_with("test")
            || name.ends_with("-hooks")
            || name.contains("for-test")
            || name.contains("testing"))
}

/// Every violation of the six rules, each one a sentence that names the edge or the crate.
fn violations(graph: &Graph, policy: &Policy<'_>) -> Vec<String> {
    let mut out = Vec::new();
    let rank_of: BTreeMap<&str, u8> = policy.ranks.iter().copied().collect();

    // Rule 1: the table and the member list agree.
    for member in &graph.members {
        if !rank_of.contains_key(member.as_str()) {
            out.push(format!(
                "rule 1: workspace member `{member}` has no row in RANKS; declare its rank"
            ));
        }
    }
    for (name, _) in policy.ranks {
        if !graph.members.contains(*name) {
            out.push(format!(
                "rule 1: RANKS row `{name}` names no workspace member; delete the stale row"
            ));
        }
    }

    // Rules 2-4: edges between members.
    let mut seen_edges = BTreeSet::new();
    for dep in &graph.deps {
        if !graph.members.contains(&dep.to) || dep.from == dep.to {
            continue;
        }
        if !seen_edges.insert((dep.from.as_str(), dep.to.as_str(), dep.kind)) {
            continue;
        }
        let (Some(&from), Some(&to)) =
            (rank_of.get(dep.from.as_str()), rank_of.get(dep.to.as_str()))
        else {
            continue; // reported by rule 1
        };
        if to >= from {
            let (rule, kind) = match dep.kind {
                Kind::Normal => ("rule 2", "normal"),
                Kind::Build => ("rule 2", "build"),
                Kind::Dev => ("rule 3", "dev"),
            };
            out.push(format!(
                "{rule}: {kind} edge {} (rank {from}) -> {} (rank {to}) does not point down; \
                 move the code (or the test) to the higher layer",
                dep.from, dep.to
            ));
        }
        if dep.kind == Kind::Normal {
            for (dependents, forbidden, reason) in policy.forbidden {
                if dep.to == *forbidden && dependents.contains(&dep.from.as_str()) {
                    out.push(format!(
                        "rule 4: {} depends on {} ({reason})",
                        dep.from, dep.to
                    ));
                }
            }
        }
    }

    // Rule 5a: one requirement string per external name.
    let mut requirements: BTreeMap<&str, BTreeMap<&str, BTreeSet<&str>>> = BTreeMap::new();
    for dep in &graph.deps {
        if !graph.members.contains(&dep.to) {
            requirements
                .entry(dep.to.as_str())
                .or_default()
                .entry(dep.requirement.as_str())
                .or_default()
                .insert(dep.from.as_str());
        }
    }
    for (name, by_requirement) in &requirements {
        if by_requirement.len() > 1 {
            let detail: Vec<String> = by_requirement
                .iter()
                .map(|(requirement, from)| {
                    format!(
                        "`{requirement}` in {}",
                        from.iter().copied().collect::<Vec<_>>().join(", ")
                    )
                })
                .collect();
            out.push(format!(
                "rule 5: `{name}` has {} requirement strings ({}); declare it once in \
                 [workspace.dependencies] and inherit it",
                by_requirement.len(),
                detail.join("; ")
            ));
        }
    }

    // Rule 5b: the lockfile's duplicate versions are exactly the allow list.
    let allowed: BTreeMap<&str, &str> = policy.duplicate_allow.iter().copied().collect();
    for (name, versions) in &graph.versions {
        if versions.len() > 1 && !allowed.contains_key(name.as_str()) {
            out.push(format!(
                "rule 5: `{name}` is locked at {} versions ({}) and has no DUPLICATE_ALLOW row; unify it \
                 or add a row naming what pulls the extra version",
                versions.len(),
                versions.iter().cloned().collect::<Vec<_>>().join(", ")
            ));
        }
    }
    for (name, _) in policy.duplicate_allow {
        let count = graph.versions.get(*name).map_or(0, BTreeSet::len);
        if count < 2 {
            out.push(format!(
                "rule 5: DUPLICATE_ALLOW row `{name}` is stale: the lockfile holds {count} version(s); \
                 delete the row"
            ));
        }
    }

    // Rule 6: no test-only feature switch beyond `test-support`.
    for (member, names) in &graph.features {
        for name in names {
            if is_unsanctioned_test_feature(name) {
                out.push(format!(
                    "rule 6: {member} declares the test-only feature `{name}`; test-only API lives behind \
                     `{SANCTIONED_TEST_FEATURE}` (ADR-0113) or in a `#[cfg(test)]` module, never behind \
                     another feature (ruling XC2-06)"
                ));
            }
        }
    }

    out
}

/// `cargo metadata` of the real workspace.
fn workspace_metadata() -> Value {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets CARGO_MANIFEST_DIR for test runs"),
    );
    let workspace_manifest = manifest_dir.join("../../Cargo.toml");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let output = Command::new(cargo)
        .args(["metadata", "--format-version", "1", "--locked", "--offline"])
        .arg("--manifest-path")
        .arg(&workspace_manifest)
        .output()
        .expect("run cargo metadata");
    assert!(
        output.status.success(),
        "cargo metadata failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("cargo metadata is JSON")
}

/// The real workspace satisfies every rule.
#[test]
fn the_workspace_points_down_the_declared_ranks() {
    let graph = parse(&workspace_metadata());
    let found = violations(&graph, &POLICY);
    assert!(
        found.is_empty(),
        "{} dependency-direction violation(s):\n  {}",
        found.len(),
        found.join("\n  ")
    );
}

/// `poot-text` is a leaf of the workspace: its dependency closure is external crates only. It reads a GGUF's
/// vocabulary as plain data (`GgufVocab`), so it has no edge to the checkpoint reader (`poot-load`), the model
/// registry (`poot-models`) or the driver (`poot-llm`), and no `wgpu`/`cudarc` reach its tests. The rank table
/// already rejects any member edge of a rank-0 crate (rules 2 and 3); this names the property.
#[test]
fn poot_text_depends_on_no_workspace_crate() {
    let graph = parse(&workspace_metadata());
    assert!(
        graph.members.contains("poot-text"),
        "poot-text must be a workspace member"
    );
    let edges: Vec<String> = graph
        .deps
        .iter()
        .filter(|dep| dep.from == "poot-text" && graph.members.contains(&dep.to))
        .map(|dep| format!("{:?} edge poot-text -> {}", dep.kind, dep.to))
        .collect();
    assert!(
        edges.is_empty(),
        "poot-text must depend on external crates only, but has:\n  {}",
        edges.join("\n  ")
    );
}

/// A small synthetic workspace the rule tests mutate: `low` (0) <- `mid` (1) <- `top` (2), one external
/// requirement used by `mid` and `top`, one unduplicated lockfile name.
mod synthetic {
    use super::*;

    pub(crate) const RANKS: &[(&str, u8)] = &[("low", 0), ("mid", 1), ("top", 2)];
    pub(crate) const FORBIDDEN: &[(&[&str], &str, &str)] =
        &[(&["top"], "low", "top never uses low directly")];
    pub(crate) const ALLOW: &[(&str, &str)] = &[("dup", "pulled by an external crate")];

    pub(crate) fn policy() -> Policy<'static> {
        Policy {
            ranks: RANKS,
            forbidden: FORBIDDEN,
            duplicate_allow: ALLOW,
        }
    }

    fn dep(from: &str, to: &str, kind: Kind, requirement: &str) -> Dep {
        Dep {
            from: from.into(),
            to: to.into(),
            kind,
            requirement: requirement.into(),
        }
    }

    pub(crate) fn graph() -> Graph {
        let mut versions = BTreeMap::new();
        for (name, vs) in [
            ("low", vec!["0.0.0"]),
            ("mid", vec!["0.0.0"]),
            ("top", vec!["0.0.0"]),
            ("serde", vec!["1.0.200"]),
            ("dup", vec!["1.0.0", "2.0.0"]),
        ] {
            versions.insert(
                name.to_string(),
                vs.into_iter().map(String::from).collect::<BTreeSet<_>>(),
            );
        }
        Graph {
            features: [
                ("low", vec!["serde"]),
                ("mid", vec!["serde", "test-support"]),
                ("top", vec![]),
            ]
            .into_iter()
            .map(|(member, names)| {
                (
                    member.to_string(),
                    names.into_iter().map(String::from).collect(),
                )
            })
            .collect(),
            members: ["low", "mid", "top"].map(String::from).into(),
            deps: vec![
                dep("mid", "low", Kind::Normal, "*"),
                dep("top", "mid", Kind::Normal, "*"),
                dep("top", "low", Kind::Dev, "*"),
                dep("mid", "serde", Kind::Normal, "^1"),
                dep("top", "serde", Kind::Dev, "^1"),
            ],
            versions,
        }
    }
}

#[test]
fn the_synthetic_workspace_is_clean() {
    let found = violations(&synthetic::graph(), &synthetic::policy());
    assert_eq!(found, Vec::<String>::new());
}

/// SC-001: an upward normal edge is a rule 2 violation.
#[test]
fn an_upward_normal_edge_fails() {
    let mut graph = synthetic::graph();
    graph.deps.push(Dep {
        from: "low".into(),
        to: "top".into(),
        kind: Kind::Normal,
        requirement: "*".into(),
    });
    let found = violations(&graph, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("rule 2: normal edge low (rank 0) -> top (rank 2)"),
        "{found:?}"
    );
}

/// Equal ranks do not point down either.
#[test]
fn a_same_rank_edge_fails() {
    let mut graph = synthetic::graph();
    graph.members.insert("peer".into());
    let ranks: &[(&str, u8)] = &[("low", 0), ("mid", 1), ("top", 2), ("peer", 1)];
    graph.deps.push(Dep {
        from: "mid".into(),
        to: "peer".into(),
        kind: Kind::Build,
        requirement: "*".into(),
    });
    let policy = Policy {
        ranks,
        ..synthetic::policy()
    };
    let found = violations(&graph, &policy);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("rule 2: build edge mid (rank 1) -> peer (rank 1)"),
        "{found:?}"
    );
}

/// SC-002: a dev edge that does not point down is a rule 3 violation, and a self dev-dependency is not an edge.
#[test]
fn an_upward_dev_edge_fails() {
    let mut graph = synthetic::graph();
    graph.deps.push(Dep {
        from: "mid".into(),
        to: "top".into(),
        kind: Kind::Dev,
        requirement: "*".into(),
    });
    graph.deps.push(Dep {
        from: "mid".into(),
        to: "mid".into(),
        kind: Kind::Dev,
        requirement: "*".into(),
    });
    let found = violations(&graph, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("rule 3: dev edge mid (rank 1) -> top (rank 2)"),
        "{found:?}"
    );
}

/// SC-003: a second requirement string for one external name, and an undeclared duplicate version, both fail;
/// so does an allow row that is no longer duplicated.
#[test]
fn version_drift_and_stale_allow_rows_fail() {
    let mut pinned = synthetic::graph();
    pinned.deps.push(Dep {
        from: "low".into(),
        to: "serde".into(),
        kind: Kind::Normal,
        requirement: "=1.0.200".into(),
    });
    let found = violations(&pinned, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("rule 5: `serde` has 2 requirement strings"),
        "{found:?}"
    );

    let mut duplicated = synthetic::graph();
    duplicated
        .versions
        .get_mut("serde")
        .unwrap()
        .insert("2.0.0".into());
    let found = violations(&duplicated, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("`serde` is locked at 2 versions"),
        "{found:?}"
    );

    let mut unified = synthetic::graph();
    unified.versions.get_mut("dup").unwrap().remove("2.0.0");
    let found = violations(&unified, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("DUPLICATE_ALLOW row `dup` is stale"),
        "{found:?}"
    );
}

/// SC-005: a member with no rank row and a row naming no member both fail.
#[test]
fn a_member_without_a_row_and_a_stale_row_fail() {
    let mut added = synthetic::graph();
    added.members.insert("newcomer".into());
    let found = violations(&added, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("workspace member `newcomer` has no row"),
        "{found:?}"
    );

    let mut removed = synthetic::graph();
    removed.members.remove("top");
    removed.deps.retain(|dep| dep.from != "top");
    let found = violations(&removed, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("RANKS row `top` names no workspace member"),
        "{found:?}"
    );
}

/// SC-006: a forbidden normal edge fails even when its ranks point down; a dev edge to the same crate is
/// not forbidden.
#[test]
fn a_forbidden_normal_edge_fails() {
    let mut graph = synthetic::graph();
    graph.deps.push(Dep {
        from: "top".into(),
        to: "low".into(),
        kind: Kind::Normal,
        requirement: "*".into(),
    });
    let found = violations(&graph, &synthetic::policy());
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].contains("rule 4: top depends on low (top never uses low directly)"),
        "{found:?}"
    );
}

/// SC-002: a member declaring a `test-hooks` feature (or another test-only switch) fails; the sanctioned
/// `test-support` and ordinary feature names do not.
#[test]
fn a_test_hooks_feature_fails() {
    for name in [
        "test-hooks",
        "testing",
        "test-utils",
        "for-tests",
        "kernel-hooks",
    ] {
        let mut graph = synthetic::graph();
        graph.features.get_mut("top").unwrap().insert(name.into());
        let found = violations(&graph, &synthetic::policy());
        assert_eq!(found.len(), 1, "{name}: {found:?}");
        assert!(
            found[0].contains(&format!(
                "rule 6: top declares the test-only feature `{name}`"
            )),
            "{name}: {found:?}"
        );
    }
    let mut graph = synthetic::graph();
    for name in ["test-support", "cli", "rocm", "serde", "read-counter"] {
        graph.features.get_mut("top").unwrap().insert(name.into());
    }
    assert_eq!(
        violations(&graph, &synthetic::policy()),
        Vec::<String>::new()
    );
}

/// The embedded policy tables are well-formed: unique rows, and every forbidden pair names ranked members.
#[test]
fn the_policy_tables_are_well_formed() {
    let mut names = BTreeSet::new();
    for (name, _) in RANKS {
        assert!(names.insert(*name), "duplicate RANKS row `{name}`");
    }
    for (dependents, dependency, _) in FORBIDDEN_NORMAL {
        for name in dependents.iter().chain(std::iter::once(dependency)) {
            assert!(
                names.contains(name),
                "FORBIDDEN_NORMAL names unranked `{name}`"
            );
        }
    }
    let mut allowed = BTreeSet::new();
    for (name, reason) in DUPLICATE_ALLOW {
        assert!(
            allowed.insert(*name),
            "duplicate DUPLICATE_ALLOW row `{name}`"
        );
        assert!(
            !reason.is_empty(),
            "DUPLICATE_ALLOW row `{name}` names no introducer"
        );
    }
}
