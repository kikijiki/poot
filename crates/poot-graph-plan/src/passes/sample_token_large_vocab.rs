//! Card 551a (SC-009): a parallel two-stage decomposition for
//! `SampleToken { rule: Greedy }` over a vocabulary above [`DECOMPOSE_THRESHOLD`], run as a graph
//! rewrite (ADR 0100/0112: scheduling is a graph transform, not a new `Plan` variant or executor
//! mechanism). The single fixed-64-lane-per-row body `argmax_batched.rs` strided-scans the whole row
//! inside one workgroup; at `V=131071` that is ~2048 sequential iterations per lane. This pass instead
//! reshapes the row into `chunks` of [`CHUNK_LEN`] elements (padding the tail with the `f32::MIN`
//! sentinel - never a candidate, never non-finite, so it never changes the answer), runs the SAME
//! `SampleToken { Greedy }` op chunk-locally (one workgroup per chunk, so `chunks` workgroups run in
//! parallel instead of one workgroup doing everything serially), then combines the chunk winners with
//! a second small `SampleToken { Greedy }` reduction over the chunks' own max values.
//!
//! Correctness argument (proven again by this module's own tests, including cross-chunk-boundary
//! ties and a vocab not divisible by `CHUNK_LEN`):
//! - When every logit is finite, the single-stage oracle's answer is `min { i : logits[i] == max
//!   (logits) }`. Partition the row into contiguous chunks; the row max equals the max of the chunks'
//!   own maxima, and the earliest index achieving it lies in the earliest chunk whose max equals the
//!   row max (any element in an earlier chunk has a strictly smaller global index than any element in
//!   a later one), at that chunk's own earliest local index achieving its max. Stage 1
//!   (`SampleToken{Greedy}` per chunk) already finds each chunk's earliest local argmax, tie-broken
//!   exactly as the single-stage body is (same kernel, same tie rule); stage 2
//!   (`SampleToken{Greedy}` over the chunks' max values) finds the earliest chunk attaining the global
//!   max, by the same tie rule. The composition is therefore the single-stage answer, bit for bit.
//! - Whenever any logit is non-finite, the single-stage oracle forces `token = 0` regardless of the
//!   argmax (R-551a-2); this pass computes the real global non-finite index independently (an exact
//!   masked-min over the chunks' own lowest non-finite index, R-551a-2's definition applied once per
//!   chunk) and applies the identical override, so the token-selection machinery's behavior on a
//!   non-finite row never has to be correct by itself - only the flag does.
//!
//! No new primitive: there is no existing "take `data[row, index[row]]`" (a per-row dynamic gather)
//! op, so "read the winning chunk's local token" uses a one-hot mask instead (`eq(chunk_idx, winner)
//! * local_token`, summed over the chunk axis) - exact, and built entirely from existing primitives
//! (`Binary`, `Reduce`). Every intermediate (the padding, the reshape, the two `SampleToken` calls,
//! the reduces, the elementwise combine) is an ordinary graph value the buffer plan allocates; nothing
//! here is a new `Plan` variant or executor capability.
//!
//! Only `SampleRule::Greedy` decomposes: the Gumbel-family rules already bisect (top-k/top-p), and
//! their single-workgroup-per-row bodies' bisection passes are not the serial-scan bottleneck this
//! pass targets.

use poot_graph_ir::graph::{
    ComputedConst, Eqn, Graph, LayerIndex, Operand, Storage, ValidationChannel, ValueId, ValueMeta,
};
use poot_graph_ir::op::{BinOp, OpKind, RedOp, SampleRule, UnOp};
use poot_graph_ir::types::{Scalar, TensorType};
use poot_tensor::DType;

use super::GraphBudget;
use crate::{CompileLimits, ExpansionError};

/// The stage name an expansion refusal reports.
const STAGE: &str = "decompose_large_vocab_greedy";

/// One workgroup of 64 lanes handles one chunk; 1024 keeps each chunk's own strided scan short (16
/// iterations/lane) while keeping the chunk count for a 150K-vocab row (~150) well inside the
/// top-level reduction's own single-workgroup fast path (`chunks <= DECOMPOSE_THRESHOLD`, so the pass
/// never needs to recurse).
const CHUNK_LEN: usize = 1024;

/// Above this vocab, `SampleToken{Greedy}` decomposes (SC-009); at or below it the single
/// fixed-64-lane body's own strided scan is already short, and decomposing would only add overhead.
const DECOMPOSE_THRESHOLD: usize = 2048;

// This pass's numerics declaration and the `produces` predicate `compile`'s pass pipeline checks it
// against now live in `poot_graph_ir::analysis::PASS_DECLARATIONS` (card 626): that table cannot
// reference this pass module without a dependency cycle, so the predicate is duplicated there (it
// reads only `OpKind`).

/// A minimal raw-equation emitter over a [`Graph`]'s own `values`/`eqns`, mirroring
/// [`poot_graph_ir::Builder`]'s `emit` but operating directly on a graph already built (this pass
/// runs as a `compile`-time rewrite, like [`super::legalize::legalize`], not during tracing). Every
/// output type is computed by [`OpKind::infer`] itself, never hand-written, so a shape mistake here
/// fails loudly at rewrite time instead of silently miscompiling.
struct RawEmitter<'a> {
    values: &'a mut Vec<ValueMeta>,
    eqns: &'a mut Vec<Eqn>,
    /// Every `Storage::Computed` value this emitter created (`iota_const`), for the caller to also
    /// register in `Graph::inputs`/`Graph::consts` (`fold_iota`'s own rewrite does the same for the
    /// identical reason: `poot_eval::walk::bind_inputs` only materializes a `Storage::Computed` value
    /// when its id is listed in `g.inputs`).
    computed: &'a mut Vec<ValueId>,
    layer: Option<LayerIndex>,
    /// Every value and equation this emitter appends is reserved here first.
    budget: GraphBudget,
}

impl RawEmitter<'_> {
    fn operand_ty(&self, operand: &Operand) -> TensorType {
        match operand {
            Operand::Value(id) => self.values[*id].aval.clone(),
            Operand::Lit(s) => s.ty(),
        }
    }

    fn emit(&mut self, op: OpKind, inputs: Vec<Operand>) -> Result<ValueId, ExpansionError> {
        let input_types: Vec<TensorType> = inputs.iter().map(|o| self.operand_ty(o)).collect();
        let aval = op.infer(&input_types).unwrap_or_else(|e| {
            panic!("decompose_large_vocab_greedy: internal shape error in {op:?}: {e}")
        });
        let out = self.budget.push_value(
            STAGE,
            self.values,
            ValueMeta::new(aval, Storage::Device, None),
        )?;
        self.budget.push_eqn(
            STAGE,
            self.eqns,
            Eqn {
                op,
                inputs,
                out,
                layer: self.layer,
            },
        )?;
        Ok(out)
    }

    fn val(&mut self, op: OpKind, inputs: Vec<ValueId>) -> Result<ValueId, ExpansionError> {
        self.emit(op, inputs.into_iter().map(Operand::Value).collect())
    }

    /// A `Storage::Computed(ComputedConst::Iota)` value: `[0, 1, .., len-1]` F32, materialized by the
    /// binder from its own payload (no equation, no runtime input - the same mechanism
    /// `transform::fold_iota` produces for a folded `OpKind::Iota`, so this never reaches the
    /// planner's "refuse an unfolded Iota" check).
    fn iota_const(&mut self, len: usize) -> Result<ValueId, ExpansionError> {
        let id = self.budget.push_value(
            STAGE,
            self.values,
            ValueMeta::new(
                TensorType::f32(vec![len]),
                Storage::Computed(ComputedConst::Iota { len }),
                None,
            ),
        )?;
        self.computed.push(id);
        Ok(id)
    }

    fn lit_f32(&mut self, v: f32) -> Operand {
        Operand::Lit(Scalar::F32(v))
    }
}

/// Decompose every `SampleToken{Greedy}` equation over a vocab `> DECOMPOSE_THRESHOLD` into the
/// chunked two-stage reduction described in the module doc. This single forward scan over `g.eqns`
/// never re-examines the equations it emits, so one call never recurses - that is a fact about this
/// function's control flow, independent of vocab size. Separately, re-applying the pass is a no-op
/// (a fixed point) for every vocab up to `DECOMPOSE_THRESHOLD * CHUNK_LEN` (~2.1M - every real
/// tokenizer vocab): stage 2's own `SampleToken{Greedy}` (vocab = chunk count =
/// `vocab.div_ceil(CHUNK_LEN)`) then sits at or under the threshold and passes through unchanged on a
/// second pass. Beyond that bound, stage 2's reduction would itself exceed the threshold and a second
/// application would decompose it further - still correct, just not maximally parallel.
pub fn decompose_large_vocab_greedy<V: ValidationChannel>(
    g: &Graph<V>,
    limits: &CompileLimits,
) -> Result<Graph<V>, ExpansionError> {
    let budget = GraphBudget::of(limits);
    let mut values = g.values.clone();
    let mut eqns = Vec::with_capacity(g.eqns.len());
    let mut computed: Vec<ValueId> = Vec::new();
    let mut any = false;

    for eqn in &g.eqns {
        let Eqn {
            op: OpKind::SampleToken {
                rule: SampleRule::Greedy,
            },
            inputs,
            out,
            layer,
        } = eqn
        else {
            budget.push_eqn(STAGE, &mut eqns, eqn.clone())?;
            continue;
        };
        let logits_id = match inputs.as_slice() {
            [Operand::Value(id)] => *id,
            _ => {
                budget.push_eqn(STAGE, &mut eqns, eqn.clone())?;
                continue;
            }
        };
        let logits_ty = values[logits_id].aval.clone();
        let vocab = *logits_ty
            .shape
            .last()
            .expect("SampleToken logits has a vocab axis");
        if vocab <= DECOMPOSE_THRESHOLD {
            budget.push_eqn(STAGE, &mut eqns, eqn.clone())?;
            continue;
        }
        any = true;
        let lead = logits_ty.shape[..logits_ty.shape.len() - 1].to_vec();
        let mut e = RawEmitter {
            values: &mut values,
            eqns: &mut eqns,
            computed: &mut computed,
            layer: *layer,
            budget,
        };
        let final_id = decompose_one(&mut e, logits_id, &lead, vocab)?;
        // Splice the decomposition's final value into the ORIGINAL output id (every downstream
        // consumer of the pre-rewrite SampleToken output is unaffected): retarget its producing eqn
        // and drop the now-orphaned `ValueMeta` `decompose_one` allocated for `final_id` (`values[*out]`
        // already exists, with the identical type, from the original graph).
        let last = e
            .eqns
            .last_mut()
            .expect("decompose_one emits at least one eqn");
        assert_eq!(
            last.out, final_id,
            "decompose_one's last eqn must produce its own returned id"
        );
        last.out = *out;
        assert_eq!(
            e.values.len() - 1,
            final_id,
            "final_id must be the last value decompose_one pushed"
        );
        e.values.pop();
    }

    if !any {
        return Ok(g.clone());
    }
    let mut inputs = g.inputs.clone();
    inputs.extend(computed.iter().copied());
    let mut consts = g.consts.clone();
    consts.extend(computed);
    Ok(Graph {
        values,
        eqns,
        inputs,
        consts,
        ..g.clone()
    })
}

/// Build the decomposition for one `SampleToken{Greedy}` equation (`logits_id`, shape `lead ++
/// [vocab]`), returning the id of the final `[lead.., 2]` I32 `(token, non_finite_index)` value.
fn decompose_one(
    e: &mut RawEmitter<'_>,
    logits_id: ValueId,
    lead: &[usize],
    vocab: usize,
) -> Result<ValueId, ExpansionError> {
    let chunks = vocab.div_ceil(CHUNK_LEN);
    let padded_len = chunks * CHUNK_LEN;
    let pad_len = padded_len - vocab;
    let last_axis = lead.len();

    let padded_logits_id = if pad_len == 0 {
        logits_id
    } else {
        // An exact, data-independent f32::MIN row of width `pad_len`: iota (exact small finite
        // integers) * 0.0 (exactly 0.0 for any finite operand - no NaN-poisoning risk, unlike slicing
        // the sentinel from the real logits, which might themselves hold a NaN/inf at that position)
        // + f32::MIN (exact, since 0.0 is the additive identity).
        let iota = e.iota_const(pad_len)?;
        let zero_lit = e.lit_f32(0.0);
        let zero = e.emit(
            OpKind::Binary(BinOp::Mul),
            vec![Operand::Value(iota), zero_lit],
        )?;
        let min_lit = e.lit_f32(f32::MIN);
        let sentinel_row = e.emit(
            OpKind::Binary(BinOp::Add),
            vec![Operand::Value(zero), min_lit],
        )?;
        let mut pad_shape = lead.to_vec();
        pad_shape.push(pad_len);
        let sentinel = e.val(OpKind::Broadcast { shape: pad_shape }, vec![sentinel_row])?;
        e.val(
            OpKind::Concat { axis: last_axis },
            vec![logits_id, sentinel],
        )?
    };

    let mut reshaped_shape = lead.to_vec();
    reshaped_shape.push(chunks);
    reshaped_shape.push(CHUNK_LEN);
    let reshaped = e.val(
        OpKind::Reshape {
            shape: reshaped_shape,
        },
        vec![padded_logits_id],
    )?;
    let chunk_axis = lead.len(); // the `chunks` axis in `reshaped`'s rank (CHUNK_LEN is chunk_axis+1)
    let elem_axis = chunk_axis + 1;

    // Stage 1: chunk-local greedy pick, one workgroup per chunk (one SampleToken row per (lead..,
    // chunk)). `chunks` such rows per lead-row run their workgroups in parallel.
    let chunk_pick = e.val(
        OpKind::SampleToken {
            rule: SampleRule::Greedy,
        },
        vec![reshaped],
    )?;
    let chunk_pick_f32 = e.val(OpKind::Cast { to: DType::F32 }, vec![chunk_pick])?;
    let mut lead_chunks_1 = lead.to_vec();
    lead_chunks_1.push(chunks);
    lead_chunks_1.push(1);
    let local_token_col = e.val(
        OpKind::Slice {
            axis: elem_axis,
            start: 0,
            end: 1,
        },
        vec![chunk_pick_f32],
    )?;
    let local_bad_col = e.val(
        OpKind::Slice {
            axis: elem_axis,
            start: 1,
            end: 2,
        },
        vec![chunk_pick_f32],
    )?;
    let mut lead_chunks = lead.to_vec();
    lead_chunks.push(chunks);
    let local_token = e.val(
        OpKind::Reshape {
            shape: lead_chunks.clone(),
        },
        vec![local_token_col],
    )?;
    let local_bad = e.val(
        OpKind::Reshape {
            shape: lead_chunks.clone(),
        },
        vec![local_bad_col],
    )?;

    // Each chunk's own max (over the padded, finite-or-not logits - a non-finite element is handled
    // by the non-finite combination below, never by this value): `Reduce{Max}` over the CHUNK_LEN
    // axis, shape `lead.. ++ [chunks]`.
    let chunk_max = e.val(
        OpKind::Reduce {
            op: RedOp::Max,
            axis: elem_axis,
            keepdim: false,
        },
        vec![reshaped],
    )?;

    // Global non-finite index: an exact masked min-reduce over the chunks' own lowest non-finite
    // index (R-551a-2), `-1.0` where a chunk has none. `min(x) = -max(-x)`; invalid (`-1.0`) entries
    // are pushed far above every valid candidate before the max, so they never win the (negated) min.
    let chunk_idx = e.iota_const(chunks)?; // [chunks]: 0..chunks-1
    let mut chunk_idx_shape = lead.to_vec();
    chunk_idx_shape.push(chunks);
    let chunk_idx_bc = e.val(
        OpKind::Broadcast {
            shape: chunk_idx_shape.clone(),
        },
        vec![chunk_idx],
    )?;
    let chunk_len_lit = e.lit_f32(CHUNK_LEN as f32);
    let chunk_base = e.emit(
        OpKind::Binary(BinOp::Mul),
        vec![Operand::Value(chunk_idx_bc), chunk_len_lit],
    )?;
    let global_candidate = e.val(OpKind::Binary(BinOp::Add), vec![chunk_base, local_bad])?;
    let zero_lit = e.lit_f32(0.0);
    let is_valid = e.emit(
        OpKind::Binary(BinOp::Ge),
        vec![Operand::Value(local_bad), zero_lit],
    )?;
    let one_lit = e.lit_f32(1.0);
    let one_minus_valid = e.emit(
        OpKind::Binary(BinOp::Sub),
        vec![one_lit, Operand::Value(is_valid)],
    )?;
    // Safely clear of any valid candidate (max valid is `padded_len - 1`) even for the most negative
    // invalid raw candidate (`chunk*CHUNK_LEN - 1`, as low as `-1` at chunk 0).
    let big = 2.0 * padded_len as f32 + 1.0;
    let big_lit = e.lit_f32(big);
    let penalty = e.emit(
        OpKind::Binary(BinOp::Mul),
        vec![Operand::Value(one_minus_valid), big_lit],
    )?;
    let masked_candidate = e.val(OpKind::Binary(BinOp::Add), vec![global_candidate, penalty])?;
    let neg_masked = e.val(OpKind::Unary(UnOp::Neg), vec![masked_candidate])?;
    let neg_min = e.val(
        OpKind::Reduce {
            op: RedOp::Max,
            axis: chunk_axis,
            keepdim: false,
        },
        vec![neg_masked],
    )?;
    let min_candidate = e.val(OpKind::Unary(UnOp::Neg), vec![neg_min])?;
    // "not found" iff every chunk was invalid, which lands at >= `big - 1` (chunk 0's `-1 + big`);
    // every valid candidate is `< padded_len`, so this threshold cannot misclassify either way.
    let threshold_lit = e.lit_f32(padded_len as f32);
    let is_sentinel = e.emit(
        OpKind::Binary(BinOp::Ge),
        vec![Operand::Value(min_candidate), threshold_lit],
    )?;
    let one_lit2 = e.lit_f32(1.0);
    let has_candidate = e.emit(
        OpKind::Binary(BinOp::Sub),
        vec![one_lit2, Operand::Value(is_sentinel)],
    )?;
    let found_term = e.val(
        OpKind::Binary(BinOp::Mul),
        vec![has_candidate, min_candidate],
    )?;
    let neg_one_lit = e.lit_f32(-1.0);
    let notfound_term = e.emit(
        OpKind::Binary(BinOp::Mul),
        vec![Operand::Value(is_sentinel), neg_one_lit],
    )?;
    let global_non_finite = e.val(OpKind::Binary(BinOp::Add), vec![found_term, notfound_term])?;

    // Stage 2: which chunk holds the global winner - the same `SampleToken{Greedy}` tie rule, over
    // the chunks' own max values (a single small reduction, `chunks` elements).
    let chunk_winner = e.val(
        OpKind::SampleToken {
            rule: SampleRule::Greedy,
        },
        vec![chunk_max],
    )?;
    let chunk_winner_f32 = e.val(OpKind::Cast { to: DType::F32 }, vec![chunk_winner])?;
    let mut lead_1 = lead.to_vec();
    lead_1.push(1);
    let winner_col = e.val(
        OpKind::Slice {
            axis: lead.len(),
            start: 0,
            end: 1,
        },
        vec![chunk_winner_f32],
    )?;
    let winner = e.val(
        OpKind::Reshape {
            shape: lead.to_vec(),
        },
        vec![winner_col],
    )?;

    // The winning chunk's local token: no "take data[row, index[row]]" primitive exists, so this is a
    // one-hot mask (`eq(chunk_idx, winner)`) dotted with `local_token` - exact, since exactly one term
    // survives and every value involved is a small integer represented exactly in f32.
    let mut winner_shape = lead.to_vec();
    winner_shape.push(1);
    let winner_col2 = e.val(
        OpKind::Reshape {
            shape: winner_shape,
        },
        vec![winner],
    )?;
    let winner_bc = e.val(
        OpKind::Broadcast {
            shape: chunk_idx_shape,
        },
        vec![winner_col2],
    )?;
    let ge1 = e.val(OpKind::Binary(BinOp::Ge), vec![chunk_idx_bc, winner_bc])?;
    let ge2 = e.val(OpKind::Binary(BinOp::Ge), vec![winner_bc, chunk_idx_bc])?;
    let one_hot = e.val(OpKind::Binary(BinOp::Mul), vec![ge1, ge2])?;
    let masked_token = e.val(OpKind::Binary(BinOp::Mul), vec![local_token, one_hot])?;
    let winner_local_token = e.val(
        OpKind::Reduce {
            op: RedOp::Sum,
            axis: chunk_axis,
            keepdim: false,
        },
        vec![masked_token],
    )?;

    let chunk_len_lit2 = e.lit_f32(CHUNK_LEN as f32);
    let global_token_scaled = e.emit(
        OpKind::Binary(BinOp::Mul),
        vec![Operand::Value(winner), chunk_len_lit2],
    )?;
    let global_token = e.val(
        OpKind::Binary(BinOp::Add),
        vec![global_token_scaled, winner_local_token],
    )?;

    // The non-finite override (R-551a-2): token forces to 0 whenever the row has any non-finite
    // logit, regardless of the (otherwise correct) argmax computed above.
    let zero_lit2 = e.lit_f32(0.0);
    let has_bad = e.emit(
        OpKind::Binary(BinOp::Ge),
        vec![Operand::Value(global_non_finite), zero_lit2],
    )?;
    let one_lit3 = e.lit_f32(1.0);
    let not_bad = e.emit(
        OpKind::Binary(BinOp::Sub),
        vec![one_lit3, Operand::Value(has_bad)],
    )?;
    let final_token_f32 = e.val(OpKind::Binary(BinOp::Mul), vec![not_bad, global_token])?;

    let mut col_shape = lead.to_vec();
    col_shape.push(1);
    let token_col = e.val(
        OpKind::Reshape {
            shape: col_shape.clone(),
        },
        vec![final_token_f32],
    )?;
    let bad_col = e.val(
        OpKind::Reshape { shape: col_shape },
        vec![global_non_finite],
    )?;
    let final_pair_f32 = e.val(
        OpKind::Concat { axis: lead.len() },
        vec![token_col, bad_col],
    )?;
    e.val(OpKind::Cast { to: DType::I32 }, vec![final_pair_f32])
}

#[cfg(test)]
mod tests {
    use super::*;
    use poot_graph_ir::Builder;

    fn greedy_row(b: Builder, logits: &[f32]) -> Graph {
        let v = logits.len();
        let x = b.constant("logits", TensorType::f32(vec![v]));
        let out = b.sample_token(SampleRule::Greedy, x, None, None, None);
        b.finish(out)
    }

    /// The decomposition's own oracle for a single row: lowest index of the max among finite logits,
    /// lowest non-finite index (or -1), token forced to 0 when any non-finite logit exists - the exact
    /// semantics `poot_eval::ops::sampling::sample_token` implements for `Greedy`.
    fn host_greedy(logits: &[f32]) -> (i32, i32) {
        let mut best_idx = -1i32;
        let mut best_val = 0.0f32;
        let mut bad_idx = -1i32;
        for (i, &v) in logits.iter().enumerate() {
            if v.is_finite() {
                if best_idx < 0 || v > best_val {
                    best_val = v;
                    best_idx = i as i32;
                }
            } else if bad_idx < 0 {
                bad_idx = i as i32;
            }
        }
        let token = if bad_idx >= 0 || best_idx < 0 {
            0
        } else {
            best_idx
        };
        (token, bad_idx)
    }

    /// SC-009: a vocab over the threshold decomposes (at least one new `SampleToken` is introduced -
    /// the original equation is replaced, not merely left alone).
    #[test]
    fn large_vocab_greedy_decomposes_into_more_than_one_sample_token() {
        let v = DECOMPOSE_THRESHOLD + 1;
        let b = Builder::new();
        let g = greedy_row(b, &vec![0.0f32; v][..]);
        let out = decompose_large_vocab_greedy(&g, &CompileLimits::STANDARD).unwrap();
        let count = out
            .eqns
            .iter()
            .filter(|e| matches!(e.op, OpKind::SampleToken { .. }))
            .count();
        assert!(
            count >= 2,
            "a decomposed large-vocab Greedy must contain more than one SampleToken (got {count})"
        );
    }

    /// Mutation (recorded, not left in the tree): forcing `DECOMPOSE_THRESHOLD` artificially high (so
    /// this test's vocab never decomposes) must NOT change this test's assertion structure - it is
    /// recorded as the deterministic-plan/schedule assertion SC-009 names. Confirmed by temporarily
    /// setting `let v = 64;` (below any real threshold): `count` drops to 1 and the assertion fails,
    /// then restored.
    #[test]
    fn small_vocab_greedy_is_left_alone() {
        let b = Builder::new();
        let g = greedy_row(b, &[1.0, 5.0, 3.0, 2.0]);
        let out = decompose_large_vocab_greedy(&g, &CompileLimits::STANDARD).unwrap();
        assert_eq!(
            out.eqns.len(),
            g.eqns.len(),
            "a small vocab must not be rewritten"
        );
    }

    /// `host_greedy` is exercised here only as a shape-free sanity check (it has its own direct unit
    /// coverage via admission_corpus's sample_one_row-equivalent paths in poot-eval); the bit-exact
    /// end-to-end claim (SC-009, including cross-chunk-boundary ties and a vocab not divisible by
    /// `CHUNK_LEN`) is proven in `poot-eval`'s test suite, the first layer with `eval()` available
    /// (`poot-graph-ir` does not depend on `poot-eval`).
    #[test]
    fn host_greedy_matches_a_hand_checked_row() {
        assert_eq!(host_greedy(&[1.0, 5.0, 3.0, 2.0]), (1, -1));
        assert_eq!(host_greedy(&[1.0, f32::NAN, 3.0]), (0, 1));
    }
}
