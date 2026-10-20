//! The block top-k fixture (POOT-1143): an independent plain-Rust selection that shares no code with the
//! op's primitive definition or with the device kernel, and seeded scores with controlled ties.

/// `canonical_router_scores`: NaN and `-inf` are `f32::MIN`, `+inf` is `f32::MAX`.
fn canonical(x: f32) -> f32 {
    if x.is_nan() || x == f32::NEG_INFINITY {
        f32::MIN
    } else if x == f32::INFINITY {
        f32::MAX
    } else {
        x
    }
}

/// What one query keeps, by the plain definition, as the op's packed `[k + blocks]` row: the kept block
/// ids ascending then `-1` padding, then the `0`/`1` indicator of each block.
///
/// The eligible blocks are `0..min(blocks, (pos + 1) / ratio)`; they sort by canonical score descending,
/// the stable sort keeping the lower block first on equal scores (`-0.0` equals `0.0`), and the first `k`
/// are kept.
pub fn reference_row(scores: &[f32], pos: i32, ratio: usize, k: usize) -> Vec<i32> {
    let blocks = scores.len();
    let eligible = ((pos + 1).max(0) as usize / ratio).min(blocks);
    let mut order: Vec<usize> = (0..eligible).collect();
    order.sort_by(|&a, &b| {
        canonical(scores[b])
            .partial_cmp(&canonical(scores[a]))
            .expect("canonical scores are ordered")
    });
    let mut kept: Vec<usize> = order.into_iter().take(k).collect();
    kept.sort_unstable();
    let mut row: Vec<i32> = kept.iter().map(|&b| b as i32).collect();
    row.resize(k, -1);
    row.extend((0..blocks).map(|b| i32::from(kept.contains(&b))));
    row
}

/// Deterministic scores in `[-1, 1)` drawn from `levels` distinct values, so a small `levels` makes equal
/// scores (ties) common.
pub fn grid_scores(blocks: usize, levels: u32, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..blocks)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 33) as u32 % levels) as f32 / levels as f32 * 2.0 - 1.0
        })
        .collect()
}

/// The score rows the selection stage is exercised on (design section 5, tier 1), as `(label, scores,
/// positions, ratio, k)`: distinct, coarse (many ties, some straddling the threshold) and all-equal
/// scores at block counts that are not powers of two; eligibility edges (`E` of 0, 1, `k - 1`, `k`, `k + 1`
/// and every block); and NaN, infinities and signed zeros.
#[expect(clippy::type_complexity)]
pub fn selection_cases() -> Vec<(String, Vec<Vec<f32>>, Vec<i32>, usize, usize)> {
    let mut cases = Vec::new();
    let (ratio, k) = (4, 5);
    for blocks in [7usize, 12, 64, 100] {
        let positions: Vec<i32> = [
            0,
            3,
            4,
            19,
            20,
            23,
            24,
            4 * blocks as i32 - 1,
            4 * blocks as i32 + 9,
        ]
        .to_vec();
        for (name, levels) in [("distinct", 1 << 20), ("coarse", 4), ("all equal", 1)] {
            let row = grid_scores(blocks, levels, blocks as u64);
            cases.push((
                format!("{name} {blocks}"),
                vec![row; positions.len()],
                positions.clone(),
                ratio,
                k,
            ));
        }
    }
    let (ratio, k, blocks) = (2, 6, 20);
    let positions = vec![
        -1,
        0,
        1,
        2 * (k as i32 - 1) - 1,
        2 * k as i32 - 1,
        2 * (k as i32 + 1) - 1,
        39,
    ];
    cases.push((
        "eligibility edges".into(),
        vec![grid_scores(blocks, 1 << 20, 5); positions.len()],
        positions,
        ratio,
        k,
    ));
    let odd = vec![
        f32::NAN,
        1.0,
        f32::NEG_INFINITY,
        0.0,
        -0.0,
        f32::INFINITY,
        0.0,
        f32::NAN,
        -0.0,
        2.0,
        f32::MIN,
        f32::MAX,
    ];
    let positions: Vec<i32> = (0..odd.len() as i32).collect();
    cases.push((
        "non-finite and signed zero".into(),
        vec![odd; positions.len()],
        positions,
        1,
        4,
    ));
    cases
}
