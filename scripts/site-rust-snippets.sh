#!/usr/bin/env bash
# Fails when a Rust snippet on the develop/ pages that is meant to compile does not. The ```rust blocks of
# website/docs/develop/{embedding,compute-engine}.md are extracted into one scratch example of poot-llm, each block
# as the body of a function whose parameters stand for the values the page leaves out (a loaded handle, a traced
# graph), and `cargo check`ed. Parameters, not `todo!()` locals: rustc skips borrow checking after a diverging
# statement, which would hide a use-after-move. Needs the poot toolchain (run it inside `nix develop`), so it is part
# of `just site-check` and not of the cd-docs workflow.
#
# A new ```rust block on these pages needs an entry below, so a snippet cannot join a page unchecked.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

example=crates/poot-llm/examples/zz_site_snippets.rs
trap 'rm -f "$example"' EXIT

python3 - "$example" <<'PY' || exit 1
import re, sys

# (page, block index) -> the function parameters that stand for what the page leaves out.
PARAMS = {
    ("embedding.md", 0): "",
    ("embedding.md", 1): "handle: poot_llm::driver::ModelHandle",
    ("embedding.md", 2): "handle: poot_llm::driver::ModelHandle",
    ("embedding.md", 3): "handle: poot_llm::driver::ModelHandle",
    ("embedding.md", 4): "",
    ("compute-engine.md", 0): "",
    ("compute-engine.md", 1): "g: poot_graph_ir::Graph",
    ("compute-engine.md", 2): "g: poot_graph_ir::Graph, store: poot_quant::weights::WeightStore, map: poot_quant::weights::WeightMap",
    ("compute-engine.md", 3): "kernel: poot_runtime::CompiledKernel",
}
out = "#![allow(unused)]\ntype R = Result<(), Box<dyn std::error::Error>>;\nfn main() {}\n"
seen = set()
for page in ("embedding.md", "compute-engine.md"):
    blocks = re.findall(r"```rust\n(.*?)```", open("website/docs/develop/" + page).read(), re.S)
    for i, body in enumerate(blocks):
        if (page, i) not in PARAMS:
            sys.exit(f"{page}: rust block {i} has no entry in scripts/site-rust-snippets.sh")
        seen.add((page, i))
        out += f"fn {page.split('.')[0].replace('-', '_')}_{i}({PARAMS[(page, i)]}) -> R {{\n{body}\nOk(())\n}}\n"
stale = set(PARAMS) - seen
if stale:
    sys.exit(f"entries without a block: {sorted(stale)}")
open(sys.argv[1], "w").write(out)
PY

cargo check -p poot-llm --example zz_site_snippets
