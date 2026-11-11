//! The stable rank fixture (POOT-1543): an independent plain-Rust sort that shares no code with the op's
//! primitive definition or with the device kernel, and the score rows that exercise its order.

/// The canonical score the definition ranks: NaN and `-inf` are `f32::MIN`, `+inf` is `f32::MAX`, and `-0.0`
/// equals `0.0`.
fn canonical(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        f32::MIN
    } else if x == f32::INFINITY {
        f32::MAX
    } else if x == 0.0 {
        0.0
    } else {
        x
    }
}

/// The rank of every element of `scores` by the plain definition: its position under "eligible first,
/// canonical score descending, lower index first". `eligible` holds the eligibility operand's values when the
/// rank is masked (an element is eligible when its value is at least `1.0`, a NaN is not).
pub fn reference_rank(scores: &[f32], eligible: Option<&[f32]>) -> Vec<f32> {
    let class = |i: usize| eligible.is_none_or(|e| e[i] >= 1.0);
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|&a, &b| {
        class(b)
            .cmp(&class(a))
            .then(canonical(scores[b]).total_cmp(&canonical(scores[a])))
            .then(a.cmp(&b))
    });
    let mut rank = vec![0.0; scores.len()];
    for (position, &index) in order.iter().enumerate() {
        rank[index] = position as f32;
    }
    rank
}

/// Rows of `e` scores that exercise the order: descending and ascending runs, an all-equal row, a row of three
/// levels (ties across any padding boundary), a row drawn from the special values (signed zeros, NaN, both
/// infinities, the finite extremes) and a seeded row of seven levels.
pub fn score_rows(e: usize) -> Vec<Vec<f32>> {
    let special = [
        0.0,
        -0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
        1.5,
        -1.5,
    ];
    let mut state = 0x9e37_79b9_u32;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    vec![
        (0..e).map(|i| (e - i) as f32).collect(),
        (0..e).map(|i| i as f32).collect(),
        vec![2.0; e],
        (0..e).map(|i| (i % 3) as f32).collect(),
        (0..e)
            .map(|_| special[next() as usize % special.len()])
            .collect(),
        (0..e).map(|_| (next() % 7) as f32 - 3.0).collect(),
    ]
}

/// One eligibility row per score row: a repeating pattern that is the class `eligible >= 1` for `1.0` and `2.0`
/// and not for `0.0`, `0.5` and NaN, shifted by the row so every element is eligible in some row.
pub fn eligibility_rows(rows: usize, e: usize) -> Vec<Vec<f32>> {
    let levels = [1.0, 0.0, 2.0, 0.5, f32::NAN, 1.0];
    (0..rows)
        .map(|r| (0..e).map(|i| levels[(i + r) % levels.len()]).collect())
        .collect()
}
