//! Sequential RMSNorm kernel for the AIE-core (XDNA2 NPU) import path.
//!
//! y[i] = x[i] * rsqrt(mean(x^2) + eps) * w[i], where mean(x^2) = sum_j(x[j]^2) / N.
//!
//! Two inputs (x[N], w[N]), one output (y[N]); fits the 2-S2MM IRON design (same channel budget as
//! vadd/vmul/vsub).
//!
//! Two passes: (1) ss = sum(x[i]^2); r = exp(-0.5 * ln(ss/N + eps)); (2) y[i] = x[i] * r * w[i].
//!
//! AIE2p has no hardware float sqrt/rsqrt (G_FSQRT is unlegalized by Peano), so rsqrt is computed as
//! exp(-0.5 * ln(v)) with the software exp and ln sequences in poot-codegen's AieCore emitter (as
//! vsilu_seq and softmax_row_seq use exp).
//!
//! eps is fixed at 1e-6. N = y.len(). stack_size is set to 4096 in the IRON script (the two
//! transcendental calls each expand to ~30 IR instructions; more than 2048 bytes were needed).
//!
//! No thread_index(): the IRON harness tiles data and dispatches this loop over the full input buffer on
//! one AIE2p tile.
//!
//! Build with:
//!   POOT_AIE_LLC=~/refs/peano/peano-llc \
//!   cargo test -p pootc --test import_run \
//!     imported_rmsnorm_sequential_kernel_lowers_to_aie2p --release

#![crate_type = "lib"]

pub fn __poot_kernel_rmsnorm_seq(x: &[f32], w: &[f32], y: &mut [f32]) {
    let n = y.len();
    // Pass 1: sum of squares.
    let mut ss = 0.0f32;
    let mut i = 0usize;
    while i < n {
        let v = x[i];
        ss = ss + v * v;
        i += 1;
    }
    // rsqrt(ss/N + eps) via exp(-0.5 * ln(v)); AIE2p has no hardware sqrt.
    let v = ss / (n as f32) + 1e-6f32;
    let r = (-0.5f32 * v.ln()).exp();
    // Pass 2: scale and apply weight.
    i = 0;
    while i < n {
        y[i] = x[i] * r * w[i];
        i += 1;
    }
}
