//! POOT-1268: the sampler suffix (penalties, bias, a guided mask, the draw, top-n logprobs and the
//! device-resident history) on the CPU oracle, held to the exact host [`Sampler`].
//!
//! The suffix is evaluated through `poot_eval::eval`, the bit-exact reference of the device kernels for
//! the draw and a plain interpretation of the composed adjustment; the reference is the host sampler's own
//! `adjust`, `pick` and logprob record, never a copy of their formulas.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use poot_eval::{EvalOptions, Value, eval};
use poot_graph_ir::op::SampleRule;
use poot_graph_ir::ops::sampling::{
    Adjust, HISTORY_READ_TAG, HISTORY_WRITE_TAG, LogprobRecord, MASK_TAG, MASKED_LOGIT,
    PENALTY_TAG, SampleSpec, sample_head,
};
use poot_graph_ir::{Builder, Graph, Outputs, Slot, SlotKey, TensorType, ValueId};
use poot_tensor::{DType, HostTensor};

use super::Sampler;

/// A compiled suffix and the ids of the inputs a step binds.
struct Head {
    graph: Graph,
    spec: SampleSpec,
    rows: usize,
    vocab: usize,
    pool_rows: usize,
    logits: ValueId,
    state: Option<ValueId>,
    slots: HashMap<String, ValueId>,
}

/// One step row's inputs.
struct Row {
    logits: Vec<f32>,
    seed: i32,
    /// `[inv_temp, floor_offset, noise_scale]`, and `top_p` for the top-p rule.
    params: Vec<f32>,
    top_k: i32,
    penalty: [f32; 3],
    mask: Vec<f32>,
    read: i32,
    write: i32,
}

/// What a step produced.
struct Run {
    record: HostTensor,
    /// The carried history pool after the step, `[pool_rows, 2, vocab]` flattened.
    pool: Option<Vec<f32>>,
}

impl Head {
    fn new(spec: SampleSpec, rows: usize, vocab: usize, pool_rows: usize) -> Self {
        let b = Builder::for_tests();
        let logits = b
            .slot(
                Slot::Activation,
                TensorType::new(vec![rows, vocab], DType::F32),
            )
            .unwrap();
        let pool = (spec.adjust != Adjust::None)
            .then(|| NonZeroUsize::new(pool_rows).expect("a pool has rows"));
        let head = sample_head(&b, logits, spec, pool).unwrap();
        let mut outputs = Outputs::new(head.out);
        if let Some(pair) = head.history {
            outputs = outputs.with_state(&[pair]);
        }
        let state = head.history.map(|(input, _)| input.id());
        let graph = b.finish(outputs).unwrap();
        let slots = graph
            .slots()
            .iter()
            .filter_map(|&(id, _)| {
                let key: &SlotKey = graph.meta(id).slot_key()?;
                Some((key.to_string(), id))
            })
            .collect();
        Self {
            graph,
            spec,
            rows,
            vocab,
            pool_rows,
            logits: logits.id(),
            state,
            slots,
        }
    }

    fn slot(&self, name: &str) -> ValueId {
        *self
            .slots
            .get(name)
            .unwrap_or_else(|| panic!("the head declares no slot {name}: {:?}", self.slots.keys()))
    }

    fn run(&self, rows: &[Row], pool: &[f32]) -> Run {
        assert_eq!(rows.len(), self.rows);
        let mut inputs: HashMap<ValueId, Value> = HashMap::new();
        let mut put = |id: ValueId, tensor: HostTensor| {
            inputs.insert(id, Value::from(tensor));
        };
        let flat =
            |pick: &dyn Fn(&Row) -> Vec<f32>| -> Vec<f32> { rows.iter().flat_map(pick).collect() };
        put(
            self.logits,
            HostTensor::f32(vec![self.rows, self.vocab], flat(&|r| r.logits.clone())),
        );
        if self.spec.rule != SampleRule::Greedy {
            let cols = self.spec.rule.params_cols();
            put(
                self.slot("sampler.seed"),
                HostTensor::i32(vec![self.rows], rows.iter().map(|r| r.seed).collect()),
            );
            put(
                self.slot("sampler.params"),
                HostTensor::f32(vec![self.rows, cols], flat(&|r| r.params.clone())),
            );
            if self.spec.rule.has_top_k() {
                put(
                    self.slot("sampler.top_k"),
                    HostTensor::i32(vec![self.rows], rows.iter().map(|r| r.top_k).collect()),
                );
            }
        }
        if self.spec.adjust != Adjust::None {
            put(
                self.state.expect("an adjusted head carries the history"),
                HostTensor::f32(vec![self.pool_rows, 2, self.vocab], pool.to_vec()),
            );
            put(
                self.slot(&format!("sampler.{PENALTY_TAG}")),
                HostTensor::f32(vec![self.rows, 3], flat(&|r| r.penalty.to_vec())),
            );
            if self.spec.adjust == Adjust::Guided {
                put(
                    self.slot(&format!("sampler.{MASK_TAG}")),
                    HostTensor::f32(vec![self.rows, self.vocab], flat(&|r| r.mask.clone())),
                );
            }
            put(
                self.slot(&format!("sampler.{HISTORY_READ_TAG}")),
                HostTensor::i32(vec![self.rows], rows.iter().map(|r| r.read).collect()),
            );
            put(
                self.slot(&format!("writemap.{HISTORY_WRITE_TAG}")),
                HostTensor::i32(vec![self.rows], rows.iter().map(|r| r.write).collect()),
            );
        }
        let result = eval(&self.graph, &inputs, EvalOptions::default()).unwrap();
        let record = result.output.into_host().unwrap();
        let pool = self.state.map(|_| {
            result.state[0]
                .as_host()
                .unwrap()
                .as_f32()
                .unwrap()
                .to_vec()
        });
        Run { record, pool }
    }
}

impl Row {
    /// A row over `logits` that draws greedily, with the neutral value of every adjustment input.
    fn neutral(logits: Vec<f32>) -> Self {
        let vocab = logits.len();
        Self {
            logits,
            seed: 0,
            params: vec![1.0, f32::NEG_INFINITY, 0.0, 1.0],
            top_k: 0,
            penalty: [1.0, 0.0, 0.0],
            mask: vec![0.0; vocab],
            read: 0,
            write: -1,
        }
    }

    /// The row of `sampler`'s adjustment inputs: the penalty triple and the guided mask. The mask is built
    /// here from the constraint's own permitted set (the same set the host mask uses).
    fn adjusted_by(mut self, sampler: &Sampler) -> Self {
        self.penalty = [
            sampler.repetition_penalty,
            sampler.presence_penalty,
            sampler.frequency_penalty,
        ];
        if let Some(constraint) = &sampler.constraint {
            let (allowed, eos_ok) = constraint.allowed_next();
            let eos_ok = eos_ok || allowed.is_empty();
            self.mask = (0..self.logits.len() as u32)
                .map(|id| {
                    if allowed.contains(&id) || (eos_ok && id == constraint.eos()) {
                        0.0
                    } else {
                        MASKED_LOGIT
                    }
                })
                .collect();
        }
        self
    }
}

/// The history pool with `sampler`'s counts and bias in pool row `row`.
fn pool_with(pool_rows: usize, vocab: usize, entries: &[(usize, &Sampler)]) -> Vec<f32> {
    let mut pool = vec![0.0f32; pool_rows * 2 * vocab];
    for &(row, sampler) in entries {
        for (&token, &count) in &sampler.counts {
            pool[(row * 2) * vocab + token as usize] = count as f32;
        }
        for (&token, &bias) in &sampler.logit_bias {
            pool[(row * 2 + 1) * vocab + token as usize] = bias;
        }
    }
    pool
}

/// A logprob record row: `(token, flag, chosen_logprob, [(alternative, logprob)])`.
type Record = (usize, i32, f32, Vec<(usize, f32)>);

fn records(run: &Run, n: usize) -> Vec<Record> {
    let width = LogprobRecord { n }.width();
    run.record
        .as_f32()
        .expect("a logprob head publishes F32")
        .chunks(width)
        .map(|row| {
            let top = row[LogprobRecord::HEADER..]
                .chunks(2)
                .map(|pair| (pair[0] as usize, pair[1]))
                .collect();
            (row[0] as usize, row[1] as i32, row[2], top)
        })
        .collect()
}

fn zipf_logits(vocab: usize, s: f64) -> Vec<f32> {
    (0..vocab)
        .map(|i| (-s * ((i + 1) as f64).ln()) as f32)
        .collect()
}

/// A scrambled, tie-free logit row.
fn scrambled(vocab: usize, salt: u32) -> Vec<f32> {
    (0..vocab as u32)
        .map(|i| {
            ((i.wrapping_mul(2_654_435_761).wrapping_add(salt) >> 8) % 4096) as f32 * 0.0021 - 4.0
        })
        .collect()
}

/// A sampler with penalties, a logit bias and an observed history.
fn history_sampler(temperature: f32, top_p: f32) -> Sampler {
    let mut s = Sampler::new(temperature, 0, top_p, 11)
        .with_penalties(1.3, 0.2, 0.1)
        .with_logit_bias(HashMap::from([(4u32, 0.8f32), (9, -3.0), (21, 2.5)]));
    for (token, times) in [(0u32, 3), (2, 1), (5, 2), (21, 2), (30, 1)] {
        for _ in 0..times {
            s.observe(token);
        }
    }
    s
}

/// SC-003: an adjusted greedy head with logprobs publishes, for each row, the host sampler's token, the
/// drawn token's logprob and the top-n alternatives. The reference is `Sampler::pick` with logprob recording
/// over the same logits: bias, penalties and a guided mask all active in different rows, and a plain row.
///
/// Mutation: in `logprob_record` seed the alternatives from the unadjusted logits (`z` to `logits`'s raw
/// row); the penalized rows' alternatives differ.
#[test]
fn the_adjusted_head_publishes_the_host_token_and_logprobs() {
    const VOCAB: usize = 96;
    const N: usize = 4;
    let allowed: Vec<u32> = (0..24).map(|i| 5 + 3 * i).collect();
    let constraint =
        || poot_text::Constraint::choices(allowed.iter().map(|&id| vec![id]).collect(), 95);
    let samplers = || -> Vec<Sampler> {
        vec![
            history_sampler(0.0, 1.0).with_logprobs(N),
            Sampler::greedy().with_logprobs(N),
            Sampler::greedy()
                .with_constraint(constraint())
                .with_logprobs(N),
            history_sampler(0.0, 1.0)
                .with_constraint(constraint())
                .with_logprobs(N),
        ]
    };
    let logits: Vec<Vec<f32>> = (0..4).map(|i| scrambled(VOCAB, 17 * i + 3)).collect();
    let head = Head::new(
        SampleSpec::draw(SampleRule::Greedy)
            .guided()
            .with_logprobs(N),
        4,
        VOCAB,
        4,
    );
    let device_samplers = samplers();
    let rows: Vec<Row> = device_samplers
        .iter()
        .zip(&logits)
        .enumerate()
        .map(|(i, (s, l))| {
            let mut row = Row::neutral(l.clone()).adjusted_by(s);
            row.read = i as i32;
            row
        })
        .collect();
    let entries: Vec<(usize, &Sampler)> = device_samplers.iter().enumerate().collect();
    let run = head.run(&rows, &pool_with(4, VOCAB, &entries));
    let device = records(&run, N);

    for (i, mut host) in samplers().into_iter().enumerate() {
        let token = host.pick(&logits[i]).unwrap();
        let reference = host.take_logprobs().pop().expect("logprobs were requested");
        let (device_token, flag, chosen, top) = &device[i];
        assert_eq!(*flag, -1, "row {i}: finite logits");
        assert_eq!(*device_token, token, "row {i}: token");
        assert!(
            (chosen - reference.logprob).abs() < 1e-4,
            "row {i}: chosen logprob {chosen} vs host {}",
            reference.logprob
        );
        assert_eq!(top.len(), reference.top.len(), "row {i}: alternatives");
        for (k, ((id, lp), (want_id, want_lp))) in top.iter().zip(&reference.top).enumerate() {
            assert_eq!(*id as u32, *want_id, "row {i} alternative {k}: token");
            assert!(
                (lp - want_lp).abs() < 1e-4,
                "row {i} alternative {k}: logprob {lp} vs host {want_lp}"
            );
        }
    }
}

/// SC-002: on Zipf logits the suffix's draw under penalties, bias and top-p stays inside the exact host
/// nucleus of the *adjusted* distribution, within the ADR-0114 nucleus-mass tolerance `epsilon = vocab *
/// 2^-25`: no draw falls past the longest prefix of the adjusted descending order that tolerance admits, and
/// every token of the shortest admitted nucleus that the exact distribution draws at least 8 times in 1024
/// is drawn. The adjusted row is the host sampler's own `adjust`, apart from the device.
///
/// Mutation: in `adjust_logits` return `raw` instead of the adjusted logits (skip the penalties and the
/// bias); the boosted token 21 is never drawn although the adjusted distribution draws it often.
#[test]
fn the_draw_under_penalties_bias_and_top_p_stays_in_the_exact_nucleus() {
    const VOCAB: usize = 1500;
    const ROWS: usize = 1024;
    let (temperature, top_p) = (0.9f32, 0.9f32);
    let logits = zipf_logits(VOCAB, 1.1);
    // A deep token boosted by the bias and the head of the distribution penalized, so the adjusted
    // nucleus is a different set from the raw one.
    let mut sampler = Sampler::new(temperature, 0, top_p, 11)
        .with_penalties(1.5, 0.4, 0.2)
        .with_logit_bias(HashMap::from([(700u32, 9.0f32), (3, -4.0)]));
    for (token, times) in [(0u32, 6), (1, 2), (700, 1)] {
        for _ in 0..times {
            sampler.observe(token);
        }
    }

    // The host's adjusted row and its exact distribution.
    let mut adjusted = logits.clone();
    sampler.adjust(&mut adjusted);
    let mut order: Vec<usize> = (0..VOCAB).collect();
    order.sort_by(|&a, &b| adjusted[b].total_cmp(&adjusted[a]).then(a.cmp(&b)));
    let max = f64::from(adjusted[order[0]]);
    let weight = |i: usize| ((f64::from(adjusted[i]) - max) / f64::from(temperature)).exp();
    let total: f64 = order.iter().map(|&i| weight(i)).sum();
    let mass = |tokens: &[usize]| tokens.iter().map(|&i| weight(i)).sum::<f64>() / total;
    let epsilon = VOCAB as f64 * 2f64.powi(-25);

    // The adjustment must matter: the unadjusted nucleus is a different set.
    let mut plain_order: Vec<usize> = (0..VOCAB).collect();
    plain_order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then(a.cmp(&b)));
    assert_ne!(
        order[..40],
        plain_order[..40],
        "the fixture's penalties must reorder the head of the distribution"
    );

    let head = Head::new(
        SampleSpec::draw(SampleRule::GumbelTopKTopP).adjusted(),
        ROWS,
        VOCAB,
        1,
    );
    let rows: Vec<Row> = (0..ROWS)
        .map(|i| {
            let mut row = Row::neutral(logits.clone()).adjusted_by(&sampler);
            row.seed = i as i32;
            row.params = vec![1.0 / temperature, f32::NEG_INFINITY, 1.0, top_p];
            row
        })
        .collect();
    let run = head.run(&rows, &pool_with(1, VOCAB, &[(0, &sampler)]));
    let mut support: Vec<usize> = run
        .record
        .as_i32()
        .unwrap()
        .chunks(2)
        .map(|pair| pair[0] as usize)
        .collect();
    support.sort_unstable();
    support.dedup();

    // The nucleus the device may select is any prefix of the adjusted order whose mass reaches `top_p -
    // epsilon` and which holds less than `top_p + epsilon` without its last token: bounded by the shortest
    // and the longest such prefix.
    let shortest = (1..=VOCAB)
        .find(|&size| mass(&order[..size]) >= f64::from(top_p) - epsilon)
        .expect("the full order holds all the mass");
    let longest = (shortest..=VOCAB)
        .take_while(|&size| mass(&order[..size - 1]) < f64::from(top_p) + epsilon)
        .last()
        .expect("the shortest prefix qualifies");
    for &token in &support {
        let rank = order.iter().position(|&t| t == token).unwrap();
        assert!(
            rank < longest,
            "token {token} (adjusted rank {rank}) is drawn outside the {longest}-token nucleus bound"
        );
    }
    // Every token of the shortest nucleus that the exact distribution draws often enough to be missed with
    // probability below 1e-3 shows up in 1024 draws.
    let nucleus_mass = mass(&order[..shortest]);
    for &token in &order[..shortest] {
        let expected = ROWS as f64 * weight(token) / total / nucleus_mass;
        if expected >= 8.0 {
            assert!(
                support.contains(&token),
                "token {token} is expected {expected:.1} times in {ROWS} draws and never drawn"
            );
        }
    }
}

/// The head adds the drawn token to the counts of the rows its write map names, leaves the bias channel
/// and every other pool row as they were, and writes nothing for a row whose map entry is `-1`.
///
/// Mutation: in `Adjusted::carry` drop the `hit` term (carry `self.counts` unchanged); the written row's
/// count stays.
#[test]
fn the_head_adds_the_drawn_token_to_the_rows_it_writes() {
    const VOCAB: usize = 40;
    let sampler = history_sampler(0.0, 1.0);
    let head = Head::new(SampleSpec::draw(SampleRule::Greedy).adjusted(), 3, VOCAB, 4);
    let logits: Vec<Vec<f32>> = (0..3).map(|i| scrambled(VOCAB, i + 40)).collect();
    // Row 0 reads and writes pool row 2; row 1 reads pool row 1 but its write is dropped; row 2 reads
    // pool row 3 and writes pool row 0.
    let reads = [2, 1, 3];
    let writes = [2, -1, 0];
    let rows: Vec<Row> = logits
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let mut row = Row::neutral(l.clone()).adjusted_by(&sampler);
            (row.read, row.write) = (reads[i], writes[i]);
            row
        })
        .collect();
    let before = pool_with(4, VOCAB, &[(1, &sampler), (2, &sampler), (3, &sampler)]);
    let run = head.run(&rows, &before);
    let after = run.pool.expect("an adjusted head carries the history");
    let tokens: Vec<usize> = run
        .record
        .as_i32()
        .unwrap()
        .chunks(2)
        .map(|pair| pair[0] as usize)
        .collect();
    let at = |pool: &[f32], row: usize, channel: usize, token: usize| {
        pool[(row * 2 + channel) * VOCAB + token]
    };
    // Row 0: the count of its token in pool row 2 grew by one; the bias channel is as before.
    assert_eq!(
        at(&after, 2, 0, tokens[0]),
        at(&before, 2, 0, tokens[0]) + 1.0
    );
    // Row 2 wrote pool row 0: the drawn token's count is one over pool row 3's.
    assert_eq!(
        at(&after, 0, 0, tokens[2]),
        at(&before, 3, 0, tokens[2]) + 1.0
    );
    for channel_token in 0..VOCAB {
        assert_eq!(
            at(&after, 2, 1, channel_token),
            at(&before, 2, 1, channel_token)
        );
        assert_eq!(
            at(&after, 0, 1, channel_token),
            at(&before, 3, 1, channel_token)
        );
    }
    // Pool rows no write map named are untouched; the dropped write left pool row 1 as it was.
    let row_of = |pool: &[f32], row: usize| pool[row * 2 * VOCAB..(row + 1) * 2 * VOCAB].to_vec();
    assert_eq!(row_of(&after, 1), row_of(&before, 1), "a dropped write");
    assert_eq!(row_of(&after, 3), row_of(&before, 3), "an unnamed pool row");
}

/// A NaN under a guided mask is still the draw's fault: the mask is added to the raw logits, never
/// selected over them, so the non-finite flag names the index and the token is 0. The host sampler checks
/// the raw row before it masks, and the suffix agrees.
///
/// Mutation: in `adjust_logits` combine the mask with `Max` instead of `Add`; `Max` ignores a NaN, the
/// flag stays -1 and the row draws token 17 instead of faulting at 7.
#[test]
fn a_nan_under_the_mask_is_still_a_fault() {
    const VOCAB: usize = 32;
    let mut logits = scrambled(VOCAB, 5);
    logits[7] = f32::NAN;
    let mut row = Row::neutral(logits);
    row.mask = vec![MASKED_LOGIT; VOCAB];
    for allowed in [2, 3, 11] {
        row.mask[allowed] = 0.0;
    }
    let head = Head::new(SampleSpec::draw(SampleRule::Greedy).guided(), 1, VOCAB, 1);
    let run = head.run(&[row], &pool_with(1, VOCAB, &[]));
    assert_eq!(run.record.as_i32().unwrap(), &[0, 7]);
}
