//! Card 461 reproducer: the body whose branch condition at nesting depth 4 is a two-operand `&&`, which
//! `pootc` used to spin on forever. Nothing outside this file compiles it - `just regen-kernel-assets`
//! and `import_run.rs`'s asset list are explicit - so it is inert data handed to `pootc` by the normal
//! (not `#[ignore]`d) `llc_spin_depth4_and_reproducer` test in `crates/pootc/tests/import_run.rs`.
//!
//! What the symptom was, from running `pootc` on this file under `timeout 120` on the Strix Halo iGPU box
//! (llc LLVM 22.1.5), which exits 124 with the `.kir.json` staged and no `.spv`:
//!
//! ```text
//! pootc: found 1 kernel(s):
//!   llc_spin_depth4_and::__poot_kernel_llc_spin_depth4_and: imported -> Body { 3 params, ... }
//! ```
//!
//! then a single `pootc` process at 100% CPU with **no child process** at all (the `ps` tree over 24s
//! showed only `pootc`), while `run_llc` would have spawned `llc` as a child - so `llc` was never
//! reached. Phase markers around `emit_llvm_ir` then put the spin in `poot-codegen::structurize`, before
//! the `.ll` scratch is written.
//!
//! Depth is NOT the trigger; an else arm is. Bisecting down to eight lines, all on this same packed
//! row-gather skeleton:
//!
//! | body | condition | outcome (pre-fix) |
//! | ---- | --------- | ----------------- |
//! | `&&` at depth 1 with an else arm | `if i < out.len() && index.len() > 0 { .. } else { .. }` | **spins** |
//! | `&&` at depth 2 with an else arm | `if row == 0 && rest == 1 { .. } else { .. }` | **spins** |
//! | `&&` at depth 4 with no else arm | `if e == 15 && m == 7 { out[i] = 0.0; }` | lowers |
//! | nested `if`s at depth 1-2 with else arms | as written in `packed_e4m3_row_gather.rs` | lowers |
//!
//! The shallower `&&` guards in this kernel (depths 1 and 3) have no else arm, which is why the original
//! bisection read as "depth 4".
//!
//! Root cause (the exact repeat, from instrumenting the pass): MIR gives `if a && b { T } else { E }`
//! ONE else block `E`, targeted by both conditions. `E` is therefore not dominated by the inner
//! condition's header, the inner selection's arms can only reconverge at the join through an edge from
//! outside its region, and the merge-redirect - which only rewires edges from *inside* the region -
//! appended one goto-forwarding block per round without ever making the join private. The loop's own
//! guard compared against a block count that grew with it, so it never tripped. The fix clones the shared
//! tail for the escaping selection and bounds the fixpoint against the body's entry size; see
//! `crates/poot-codegen/src/structurize.rs`.
//!
//! The landed workaround is in `packed_e4m3_row_gather.rs`: it spells `e == 15 && m == 7` as two nested
//! `if`s. This file keeps the original failing shape as the in-repo reproducer.

#![crate_type = "lib"]
use poot_kernel_intrinsics::thread_index;

pub fn __poot_kernel_llc_spin_depth4_and(table_words: &[u32], index: &[i32], out: &mut [f32]) {
    let i = thread_index();
    if i < out.len() && index.len() > 0 {
        let row_width = out.len() / index.len();
        if row_width > 0 {
            let s = i / row_width;
            let column = i % row_width;
            let words_per_row = (row_width + 3) / 4;
            let outer = table_words.len() / words_per_row;
            let row = index[s];
            if row >= 0 && (row as usize) < outer {
                let word = (row as usize) * words_per_row + column / 4;
                let shift = ((column % 4) * 8) as u32;
                let byte = (table_words[word] >> shift) & 0xff;
                let exponent = (byte >> 3) & 0x0f;
                let mantissa = byte & 0x07;
                // Depth 4, two-operand `&&`: the body that spins llc.
                if exponent == 15 && mantissa == 7 {
                    out[i] = 0.0;
                } else {
                    out[i] = 1.0;
                }
            }
        }
    }
}
