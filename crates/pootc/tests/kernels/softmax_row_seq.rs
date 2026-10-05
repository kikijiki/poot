//! Sequential row-wise softmax kernel for the AIE-core (XDNA2 NPU).
//!
//! Numerically-stable softmax over each row of a [L,L] matrix:
//!   y[i,j] = exp(x[i,j] - max_j(x[i,:])) / sum_j(exp(x[i,:] - max))
//!
//! Used in the attention 3-tile cascade (compile_attention): tile 2 converts scores[L,L] -> probs[L,L]
//! (L independent row softmaxes). This tile has only tile-to-tile inputs and outputs (0 S2MM from host).
//!
//! # Length encoding
//!
//! The IRON harness calls this as `kern(et1, L*L, et2, L)`: `et2` is the L*L output buffer but its length
//! argument is L. The kernel recovers L = y.len() and processes L rows of L elements. All index accesses
//! to `y` stay within [0, L*L).
//!
//! Requires exp via Peano's math library at link time (as softmax_seq / vsilu_seq).
//!
//! Build with:
//!   POOT_AIE_LLC=~/refs/peano/peano-llc \
//!   cargo test -p pootc --test import_run \
//!     imported_softmax_row_sequential_kernel_lowers_to_aie2p --release

#![crate_type = "lib"]

pub fn __poot_kernel_softmax_row_seq(x: &[f32], y: &mut [f32]) {
    // x.len() = L*L (row-major), y.len() = L (the IRON caller passes L, not L*L). Recover L:
    let l = y.len(); // L (rows = cols = sequence length)
    let mut row = 0usize;
    while row < l {
        let base = row * l;
        // Pass 1: find running max over x[row, :].
        let mut mx = x[base];
        let mut j = 1usize;
        while j < l {
            if x[base + j] > mx {
                mx = x[base + j];
            }
            j += 1;
        }
        // Pass 2: sum exp(x[row,j] - mx).
        let mut s = 0.0f32;
        j = 0;
        while j < l {
            s += (x[base + j] - mx).exp();
            j += 1;
        }
        // Pass 3: normalize into y.
        j = 0;
        while j < l {
            y[base + j] = (x[base + j] - mx).exp() / s;
            j += 1;
        }
        row += 1;
    }
}
