//! Graph binders shared across backends: computed constants, token and position slots, masks.

use poot_tensor::HostTensor;

/// Additive causal mask `[1,1,L,L]`: 0 where key j is visible to query i, -1e30 elsewhere.
/// `window = None`: full causal, visible iff `j <= i`.
/// `window = Some(w)`: sliding window, visible iff `j <= i && i - j < w`.
/// `window = None` and `window = Some(w)` with `w >= L` produce byte-identical outputs.
/// The host value of a `Slot::Token` input: the ids as the slot declares them - authoritative I32
/// words for an I32 slot (the storage-aware CPU oracle refuses an f32-only I32 input), f32 otherwise.
pub(crate) fn token_slot(aval: &poot_graph_ir::TensorType, ids: &[u32]) -> HostTensor {
    match aval.dtype {
        poot_tensor::DType::I32 => {
            HostTensor::i32(aval.shape.clone(), ids.iter().map(|&t| t as i32).collect())
        }
        _ => HostTensor::f32(aval.shape.clone(), ids.iter().map(|&t| t as f32).collect()),
    }
}

/// `Slot::Pos` for a one-shot prefill that always starts at position 0 (card 550): the
/// `[rows, tokens]` I32 absolute positions `[0, 1, .., tokens-1]`, repeated per row (`rows` is always 1
/// for every dense-family one-shot prefill tracer today). `l` is the caller's prompt/chunk length,
/// asserted to match the slot's declared `tokens` axis.
pub(crate) fn prefill_pos_rows(aval: &poot_graph_ir::TensorType, l: usize) -> HostTensor {
    let (rows, tokens) = (aval.shape[0], aval.shape[1]);
    debug_assert_eq!(
        tokens, l,
        "Slot::Pos token axis must match the prompt length"
    );
    let row: Vec<i32> = (0..tokens as i32).collect();
    let mut data = Vec::with_capacity(rows * tokens);
    for _ in 0..rows {
        data.extend_from_slice(&row);
    }
    HostTensor::i32(aval.shape.clone(), data)
}

pub(crate) fn prefill_causal_mask(l: usize, window: Option<usize>) -> HostTensor {
    let mut data = vec![-1e30f32; l * l];
    for i in 0..l {
        let lo = window.map_or(0, |w| i.saturating_sub(w - 1));
        for j in lo..=i {
            data[i * l + j] = 0.0;
        }
    }
    HostTensor::f32(vec![1, 1, l, l], data)
}

/// Additive causal mask `[1,1,L,L]`: 0 where key j <= query i, large-negative above. Legacy alias used by
/// VLM paths; new code should use `prefill_causal_mask(l, None)`.
pub(crate) fn causal_mask(l: usize) -> HostTensor {
    prefill_causal_mask(l, None)
}

#[cfg(test)]
mod mask_tests {
    use super::prefill_causal_mask;

    /// window=None must equal the classic full-causal mask (0 below diagonal, -1e30 above).
    #[test]
    fn prefill_window_none_equals_full_causal() {
        let l = 4;
        let got = prefill_causal_mask(l, None);
        // reference: hand-built lower-triangular additive mask
        let mut expected = vec![-1e30f32; l * l];
        for i in 0..l {
            for j in 0..=i {
                expected[i * l + j] = 0.0;
            }
        }
        assert_eq!(got.shape(), vec![1, 1, l, l]);
        assert_eq!(
            got.as_f32().unwrap(),
            &expected[..],
            "window=None must match full-causal"
        );
    }

    /// window=Some(w) zeros exactly the in-window band; the rest is -1e30.
    #[test]
    fn prefill_window_some_zeroes_correct_band() {
        let l = 5;
        let w = 2;
        let got = prefill_causal_mask(l, Some(w));
        // for each query row i, visible keys are max(0, i-w+1)..=i
        let mut expected = vec![-1e30f32; l * l];
        for i in 0..l {
            let lo = i.saturating_sub(w - 1);
            for j in lo..=i {
                expected[i * l + j] = 0.0;
            }
        }
        assert_eq!(
            got.as_f32().unwrap(),
            &expected[..],
            "window=Some({w}) must match band mask"
        );
    }

    /// w=1 means only self-attention is visible (diagonal only).
    #[test]
    fn prefill_window_1_is_diagonal_only() {
        let l = 4;
        let got = prefill_causal_mask(l, Some(1));
        let mut expected = vec![-1e30f32; l * l];
        for i in 0..l {
            expected[i * l + i] = 0.0; // only self
        }
        assert_eq!(
            got.as_f32().unwrap(),
            &expected[..],
            "window=1 must be diagonal only"
        );
    }

    /// w >= L must be byte-identical to window=None (full causal).
    #[test]
    fn prefill_window_ge_l_equals_full_causal() {
        let l = 4;
        let full = prefill_causal_mask(l, None);
        let wide = prefill_causal_mask(l, Some(l));
        let wider = prefill_causal_mask(l, Some(l + 10));
        assert_eq!(
            full.as_f32().unwrap(),
            wide.as_f32().unwrap(),
            "window=L must equal full-causal"
        );
        assert_eq!(
            full.as_f32().unwrap(),
            wider.as_f32().unwrap(),
            "window>L must equal full-causal"
        );
    }
}
