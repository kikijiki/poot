//! The generated attention kernel against the textbook two-pass attention, on the kernel-IR reference
//! interpreter (no GPU). Every row builds its body through `generate`, the one seam the planner uses, and
//! launches it on the grid that call returns, so the lane count, the 2-D fold and the workgroup shape are
//! the production ones.

use poot_kernel_ir::interp::{Buffer, run};
use poot_kernelgen as kg;
use poot_target::DeviceCaps;
use poot_test_util::{assert_close, max_abs_error};

#[path = "support/attention.rs"]
mod attention;
use attention::{KeyExtras, attention_reference, attention_reference_keys};

fn spec(
    batch: usize,
    hq: usize,
    n_rep: usize,
    lq: usize,
    lk: usize,
    d: usize,
) -> kg::AttentionSpec {
    kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        batch,
        hq,
        n_rep,
        lq,
        lk,
        dk: d,
        dv: d,
        key: kg::KeyLayout::Single,
        score: kg::ScoreTransform::Scale {
            scale_bits: (1.0 / (d as f32).sqrt()).to_bits(),
        },
        mask: kg::MaskKind::Broadcast,
        bound: kg::KeyBound::Static,
        sink: kg::AttentionSink::Absent,
        schedule: kg::AttentionSchedule::Streaming {
            lanes: LANES,
            walk: walk_of(d, d, d),
        },
        precision: poot_graph_ir::OperandPrecision::Exact,
    }
}

/// The workgroup width of the fixtures: one aligned run of 32 lanes, the least a shuffling body admits.
const LANES: usize = 32;

/// The lanes of a fixture cluster, so four clusters at [`LANES`] and four keys per step.
const CLUSTER: usize = 8;

/// The clustered walk of a fixture whose key is `dk` wide (`head_dim` of it in the first part of a split key) and
/// whose value is `dv` wide: clusters of [`CLUSTER`] lanes, each lane a run of the narrowest width that lets the
/// cluster cover both heads and divides every width. A head of 4 leaves four lanes of a cluster idle.
fn walk_of(dk: usize, dv: usize, head_dim: usize) -> kg::KeyWalk {
    let run = (0..)
        .map(|shift| 1usize << shift)
        .find(|run| {
            [dk, dv, head_dim].iter().all(|d| d % run == 0)
                && dk.div_ceil(*run) <= CLUSTER
                && dv.div_ceil(*run) <= CLUSTER
        })
        .expect("a run of one channel divides every width");
    kg::KeyWalk {
        cluster: CLUSTER,
        run,
    }
}

/// `spec` with the walk its head widths admit ([`walk_of`]): call it after the widths and the key layout are set.
fn walked(spec: kg::AttentionSpec) -> kg::AttentionSpec {
    let head_dim = match spec.key {
        kg::KeyLayout::Single | kg::KeyLayout::Paged { .. } => spec.dk,
        kg::KeyLayout::Split { shared_dim } => spec.dk - shared_dim,
    };
    let walk = walk_of(spec.dk, spec.dv, head_dim);
    let schedule = match spec.schedule {
        kg::AttentionSchedule::Streaming { lanes, .. } => {
            kg::AttentionSchedule::Streaming { lanes, walk }
        }
        kg::AttentionSchedule::SplitKv {
            lanes,
            splits,
            min_units,
            ..
        } => kg::AttentionSchedule::SplitKv {
            lanes,
            splits,
            min_units,
            walk,
        },
        tiled @ kg::AttentionSchedule::Tiled { .. } => tiled,
    };
    kg::AttentionSpec { schedule, ..spec }
}

fn with_lanes(spec: kg::AttentionSpec, lanes: usize) -> kg::AttentionSpec {
    walked(kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        schedule: kg::AttentionSchedule::Streaming {
            lanes,
            walk: walk_of(spec.dk, spec.dv, spec.dk),
        },
        ..spec
    })
}

fn fill(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32) / ((1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// The operands a spec reads, filled from fixed seeds, with `mask` the caller's.
struct Operands {
    q: Vec<f32>,
    k: Vec<f32>,
    /// The shared part of a split key.
    k_shared: Option<Vec<f32>>,
    v: Vec<f32>,
    mask: Vec<f32>,
    /// The live length of each batch row.
    len: Option<Vec<i32>>,
    /// The `[B, Lk / BLOCK]` block table of a paged key, which reads `k` and `v` as pools.
    read: Option<Vec<i32>>,
    /// The sink logit of each query head.
    sinks: Option<Vec<f32>>,
    /// The `[B, Lq]` query positions of a causal mask, which the kernel reads in the mask's place.
    positions: Option<Vec<i32>>,
}

impl Operands {
    fn new(spec: &kg::AttentionSpec, mask: Vec<f32>) -> Self {
        let hkv = spec.hq / spec.n_rep;
        let (head_dim, k_shared) = match spec.key {
            kg::KeyLayout::Single | kg::KeyLayout::Paged { .. } => (spec.dk, None),
            kg::KeyLayout::Split { shared_dim } => (
                spec.dk - shared_dim,
                Some(fill(spec.batch * spec.lk * shared_dim, 4)),
            ),
        };
        Self {
            q: fill(spec.batch * spec.hq * spec.lq * spec.dk, 1),
            k: fill(spec.batch * hkv * spec.lk * head_dim, 2),
            k_shared,
            v: fill(spec.batch * hkv * spec.lk * spec.dv, 3),
            mask,
            len: None,
            read: None,
            sinks: None,
            positions: None,
        }
    }

    /// These operands under a causal mask: the kernel reads `positions` and no mask tensor.
    fn with_positions(self, positions: &[i32]) -> Self {
        Self {
            positions: Some(positions.to_vec()),
            ..self
        }
    }

    /// The kernel's fourth parameter: the query positions of a causal mask, else the additive mask.
    fn mask_buffer(&self) -> Buffer {
        match &self.positions {
            Some(positions) => Buffer::from_i32s(positions),
            None => Buffer::from_f32s(&self.mask),
        }
    }

    /// These operands with a sink logit per query head, spread over a range that reaches above and below the
    /// scores so it takes the row maximum for some heads and not for others.
    fn with_sinks(self, hq: usize) -> Self {
        let sinks = (0..hq).map(|h| 3.0 * h as f32 - 2.0).collect();
        Self {
            sinks: Some(sinks),
            ..self
        }
    }

    /// These operands with `k` and `v` as pools of `pool_blocks` blocks of [`BLOCK`] slots, read through the
    /// block table `read`, filled from their own seeds.
    fn with_pools(self, spec: &kg::AttentionSpec, pool_blocks: usize, read: Vec<i32>) -> Self {
        let hkv = spec.hq / spec.n_rep;
        let slots = pool_blocks * BLOCK;
        Self {
            k: fill(slots * hkv * spec.dk, 5),
            v: fill(slots * hkv * spec.dv, 6),
            read: Some(read),
            ..self
        }
    }

    /// These operands with a live length per batch row.
    fn with_len(self, len: &[i32]) -> Self {
        Self {
            len: Some(len.to_vec()),
            ..self
        }
    }

    /// The kernel's operand buffers in parameter order (the output not included), the key and value parts
    /// stored in `spec`'s lane.
    fn buffers(&self, spec: &kg::AttentionSpec) -> Vec<Buffer> {
        let kv = |values: &[f32]| kv_buffer(spec.kv, values);
        let mut buffers = vec![
            Buffer::from_f32s(&self.q),
            kv(&self.k),
            kv(&self.v),
            self.mask_buffer(),
        ];
        buffers.extend(self.k_shared.as_deref().map(kv));
        buffers.extend(self.read.as_deref().map(Buffer::from_i32s));
        buffers.extend(self.len.as_deref().map(Buffer::from_i32s));
        buffers.extend(self.sinks.as_deref().map(Buffer::from_f32s));
        buffers
    }

    /// These operands with the key and value parts rounded to what `lane` stores, so the reference reads the
    /// values the kernel does.
    fn stored(self, lane: kg::KvLane) -> Self {
        let round = |values: Vec<f32>| match lane {
            kg::KvLane::F32 => values,
            kg::KvLane::F16 | kg::KvLane::Bf16 => kv_buffer(lane, &values).to_halves().unwrap(),
        };
        Self {
            k: round(self.k),
            v: round(self.v),
            k_shared: self.k_shared.map(round),
            ..self
        }
    }

    /// The textbook reference over every operand the spec reads.
    fn reference(&self, spec: &kg::AttentionSpec) -> Vec<f32> {
        attention_reference_keys(
            spec,
            &self.q,
            &self.k,
            &self.v,
            &self.mask,
            KeyExtras {
                k_shared: self.k_shared.as_deref(),
                len: self.len.as_deref(),
                read: self.read.as_deref(),
                sinks: self.sinks.as_deref(),
                positions: self.positions.as_deref(),
            },
        )
    }
}

/// A key or value buffer of `lane` holding `values` (rounded to a two-byte lane).
fn kv_buffer(lane: kg::KvLane, values: &[f32]) -> Buffer {
    match lane {
        kg::KvLane::F32 => Buffer::from_f32s(values),
        kg::KvLane::F16 => Buffer::from_halves(poot_kernel_ir::Ty::F16, values),
        kg::KvLane::Bf16 => Buffer::from_halves(poot_kernel_ir::Ty::BF16, values),
    }
}

/// Generate `spec` for `caps` and run it on the interpreter over the launch `generate` returns.
fn interpret(spec: &kg::AttentionSpec, caps: &DeviceCaps, operands: &Operands) -> Vec<f32> {
    interpret_requests(
        spec,
        &[kg::KernelRequest::Attention(spec.clone())],
        caps,
        operands,
    )
}

/// Run `requests`, in order, each over its own generated launch, all writing one output buffer that starts
/// as NaN: a row no request covers stays NaN.
fn interpret_requests(
    spec: &kg::AttentionSpec,
    requests: &[kg::KernelRequest],
    caps: &DeviceCaps,
    operands: &Operands,
) -> Vec<f32> {
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    let lanes = spec.schedule.lanes();
    let out_len = spec.batch * spec.hq * spec.lq * spec.dv;
    let mut buffers = operands.buffers(spec);
    buffers.push(Buffer::from_f32s(&vec![f32::NAN; out_len]));
    for request in requests {
        let generated =
            kg::generate(request, caps, &limits).expect("the attention request generates");
        assert_eq!(generated.body.workgroup_size, [lanes as u32, 1, 1]);
        let grid = generated.launch.grid;
        let workgroups = [grid[0] / lanes as u32, grid[1], grid[2]];
        run(&generated.body, workgroups, &mut buffers).unwrap();
    }
    buffers.last().unwrap().to_f32s().unwrap()
}

fn caps() -> DeviceCaps {
    poot_test_util::device_caps::wgpu_rdna3_igpu()
}

/// `[B, 1, Lq, Lk]` causal mask with `valid[b]` keys visible to every row of batch `b` (decode) or the
/// diagonal-and-below keys (prefill, `valid == None`).
fn additive_mask(batch: usize, lq: usize, lk: usize, valid: Option<&[usize]>) -> Vec<f32> {
    let mut mask = Vec::new();
    for b in 0..batch {
        for i in 0..lq {
            for t in 0..lk {
                let visible = match valid {
                    Some(valid) => t < valid[b],
                    None => t <= i,
                };
                mask.push(if visible { 0.0 } else { -1.0e9 });
            }
        }
    }
    mask
}

/// Decode, one query row against a 13-key cache over 64 lanes, eight clusters of eight: 13 is not a multiple of
/// the eight keys a step takes, so the second step has three clusters past the range, and every cluster that has
/// two keys rescales its running state.
#[test]
fn decode_over_ragged_clusters_matches_the_reference() {
    let spec = with_lanes(spec(1, 4, 2, 1, 13, 8), 64);
    let operands = Operands::new(&spec, additive_mask(1, 1, 13, Some(&[10])));
    let got = interpret(&spec, &caps(), &operands);
    let want = attention_reference(&spec, &operands.q, &operands.k, &operands.v, &operands.mask);
    assert_close(&got, &want, 1e-5);
}

/// Batched decode: four batch rows, each with its own visible length, so the per-batch mask plane and the
/// per-batch key/value rows are all read at the right offsets.
#[test]
fn batched_decode_reads_each_rows_own_operands() {
    let spec = spec(4, 4, 2, 1, 11, 8);
    let operands = Operands::new(&spec, additive_mask(4, 1, 11, Some(&[3, 7, 11, 1])));
    let got = interpret(&spec, &caps(), &operands);
    let want = attention_reference(&spec, &operands.q, &operands.k, &operands.v, &operands.mask);
    assert_close(&got, &want, 1e-5);
}

/// Prefill: every query row attends to the prompt through a causal mask, so rows differ in how many keys
/// they see and the `[Lq, Lk]` mask plane is indexed by row.
#[test]
fn prefill_matches_the_reference() {
    let spec = spec(1, 4, 2, 6, 6, 8);
    let operands = Operands::new(&spec, additive_mask(1, 6, 6, None));
    let got = interpret(&spec, &caps(), &operands);
    let want = attention_reference(&spec, &operands.q, &operands.k, &operands.v, &operands.mask);
    assert_close(&got, &want, 1e-5);
}

/// A prefill split into runs of rows (POOT-1313, the plan a device with a watchdog budget makes of a long
/// one) writes every row exactly once and matches the reference: the runs start mid-head and mid-batch, and
/// the last is shorter than the others. A run that restarts its rows at zero, or reads its place in the run as
/// the row, leaves the later rows NaN or overwrites the first.
///
/// Mutation: in `attention_body`, drop `rows.start` from `gi`; every run recomputes the first rows and the
/// rest of the output stays NaN.
#[test]
fn a_prefill_split_into_runs_of_rows_matches_the_reference() {
    let spec = spec(2, 4, 2, 6, 6, 8);
    let operands = Operands::new(&spec, additive_mask(2, 6, 6, None));
    let groups = spec.row_groups();
    let per_run = 5;
    let runs: Vec<_> = (0..groups.div_ceil(per_run))
        .map(|run| {
            let first = run * per_run;
            kg::KernelRequest::AttentionRows(kg::AttentionRows {
                spec: spec.clone(),
                first,
                count: per_run.min(groups - first),
            })
        })
        .collect();
    assert!(runs.len() > 2, "the split has several runs");
    let got = interpret_requests(&spec, &runs, &caps(), &operands);
    let want = attention_reference(&spec, &operands.q, &operands.k, &operands.v, &operands.mask);
    assert_close(&got, &want, 1e-5);
}

/// A run of rows outside its attention, or an empty one, is a typed error, and a split-KV schedule has no
/// runs: its stage one addresses every workgroup.
#[test]
fn a_run_of_rows_outside_the_grid_is_a_typed_error() {
    let spec = spec(1, 4, 2, 6, 6, 8);
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    let groups = spec.row_groups();
    let split = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        schedule: kg::AttentionSchedule::SplitKv {
            lanes: LANES,
            splits: 2,
            min_units: 1,
            walk: walk_of(spec.dk, spec.dv, spec.dk),
        },
        ..spec.clone()
    };
    for (spec, first, count) in [
        (spec.clone(), groups - 1, 2),
        (spec.clone(), 0, 0),
        (split, 0, 1),
    ] {
        let request = kg::KernelRequest::AttentionRows(kg::AttentionRows { spec, first, count });
        assert!(
            kg::generate(&request, &caps(), &limits).is_err(),
            "first {first} count {count} is refused"
        );
    }
}

/// A sink: one learned logit per head in the row's softmax, no key and no value. Heads whose sink sits below,
/// inside and above the range of their scores all match the reference, on a decode over ragged lanes and on a
/// causal prefill whose rows see different numbers of keys. A kernel that drops the sink, adds its mass but
/// not to the maximum, or gathers a value row for it differs from the reference by far more than the
/// tolerance.
///
/// Mutation: in `attention_body`, skip the sink's initial `(m, l)` state; the output equals the sinkless
/// reference and the heads whose sink is large go red.
#[test]
fn a_sink_joins_the_softmax_and_no_value_sum() {
    let decode = with_lanes(
        kg::AttentionSpec {
            kv: poot_kernelgen::KvLane::F32,
            sink: kg::AttentionSink::PerHead,
            ..spec(2, 4, 2, 1, 13, 8)
        },
        64,
    );
    let operands = Operands::new(&decode, additive_mask(2, 1, 13, Some(&[10, 13]))).with_sinks(4);
    let got = interpret(&decode, &caps(), &operands);
    assert_close(&got, &operands.reference(&decode), 1e-5);
    let sinkless = attention_reference(
        &decode,
        &operands.q,
        &operands.k,
        &operands.v,
        &operands.mask,
    );
    assert!(
        max_abs_error(&got, &sinkless) > 1e-2,
        "the sink must change the output, or the row proves nothing"
    );

    let prefill = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        sink: kg::AttentionSink::PerHead,
        ..spec(1, 4, 2, 6, 6, 8)
    };
    let operands = Operands::new(&prefill, additive_mask(1, 6, 6, None)).with_sinks(4);
    let got = interpret(&prefill, &caps(), &operands);
    assert_close(&got, &operands.reference(&prefill), 1e-5);
}

/// A row with no visible key and a sink divides by the sink's mass alone: zero output, not the NaN a sinkless
/// row gives (`a_row_with_no_visible_key_is_nan_and_its_neighbours_are_not`).
#[test]
fn a_sink_row_with_no_visible_key_is_zero() {
    let sunk = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        sink: kg::AttentionSink::PerHead,
        ..spec(2, 2, 1, 1, 6, 8)
    };
    let operands = Operands::new(&sunk, additive_mask(2, 1, 6, Some(&[0, 6]))).with_sinks(2);
    let got = interpret(&sunk, &caps(), &operands);
    assert_close(&got, &operands.reference(&sunk), 1e-5);
    let row = sunk.hq * sunk.dv;
    assert!(
        got[..row].iter().all(|&x| x == 0.0),
        "the empty row is zero"
    );
}

/// Gemma-style attention-logit softcap on a prefill whose scores reach well past the cap. A kernel that
/// dropped the softcap, or applied it after the mask, differs from the reference by far more than the
/// tolerance.
#[test]
fn softcapped_prefill_matches_the_reference() {
    let softcapped = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        score: kg::ScoreTransform::ScaleSoftcap {
            scale_bits: 1.5f32.to_bits(),
            cap_bits: 2.0f32.to_bits(),
        },
        ..with_lanes(spec(1, 2, 1, 9, 9, 8), LANES)
    };
    let operands = Operands::new(&softcapped, additive_mask(1, 9, 9, None));
    let got = interpret(&softcapped, &caps(), &operands);
    let want = attention_reference(
        &softcapped,
        &operands.q,
        &operands.k,
        &operands.v,
        &operands.mask,
    );
    assert_close(&got, &want, 1e-5);

    let uncapped = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        score: kg::ScoreTransform::Scale {
            scale_bits: 1.5f32.to_bits(),
        },
        ..softcapped
    };
    let without = attention_reference(
        &uncapped,
        &operands.q,
        &operands.k,
        &operands.v,
        &operands.mask,
    );
    assert!(
        max_abs_error(&got, &without) > 1e-2,
        "the fixture must be sensitive to the softcap"
    );
}

/// ALiBi: one mask plane per head, so head `h` scores against its own row. A kernel that read head 0's plane
/// for every head differs from the reference.
#[test]
fn per_head_mask_reads_each_heads_plane() {
    let per_head = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        mask: kg::MaskKind::PerHead,
        ..spec(2, 4, 2, 1, 6, 8)
    };
    let slopes = [0.25f32, 0.0625, 0.015625, 0.00390625];
    let (pos, lk) = (3usize, 6usize);
    let mut mask = Vec::new();
    for b in 0..2 {
        for slope in slopes {
            for t in 0..lk {
                let bias = if t <= pos + b {
                    -slope * ((pos + b) as f32 - t as f32)
                } else {
                    -1.0e9
                };
                mask.push(bias);
            }
        }
    }
    let operands = Operands::new(&per_head, mask);
    let got = interpret(&per_head, &caps(), &operands);
    let want = attention_reference(
        &per_head,
        &operands.q,
        &operands.k,
        &operands.v,
        &operands.mask,
    );
    assert_close(&got, &want, 1e-5);

    // Every head scored against its batch row's first plane instead.
    let mut collapsed = operands.mask.clone();
    for plane in collapsed.chunks_mut(4 * lk) {
        let first = plane[..lk].to_vec();
        for head in plane.chunks_mut(lk) {
            head.copy_from_slice(&first);
        }
    }
    let wrong = attention_reference(&per_head, &operands.q, &operands.k, &operands.v, &collapsed);
    assert!(
        max_abs_error(&got, &wrong) > 1e-4,
        "the per-head fixture must tell a collapsed mask apart"
    );
}

/// A device whose X grid cap is below the workgroup count folds the launch onto Y; the kernel's flat
/// `(batch, head, row)` index must follow the fold and the remainder workgroups must return.
#[test]
fn a_folded_grid_covers_every_row_once() {
    let spec = spec(2, 4, 2, 3, 3, 8);
    let mut narrow = caps();
    narrow.max_grid[0] = 5;
    let operands = Operands::new(&spec, additive_mask(2, 3, 3, None));
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    let launch = kg::generate(
        &kg::KernelRequest::Attention(spec.clone()),
        &narrow,
        &limits,
    )
    .unwrap()
    .launch;
    assert_eq!(
        launch.grid,
        [5 * 32, 3, 1],
        "12 workgroups of 32 lanes (2 batch rows x 2 key/value heads x 3 rows) fold onto 5 across and 3 down"
    );
    let got = interpret(&spec, &narrow, &operands);
    let want = attention_reference(&spec, &operands.q, &operands.k, &operands.v, &operands.mask);
    assert_close(&got, &want, 1e-5);
}

/// A position masked with a true `-inf` while a lane's running maximum is still `-inf` must contribute
/// nothing: `exp(-inf - -inf)` is NaN and would poison the row. Here one lane sees `-inf` first (and
/// then a finite key), and with 8 lanes over 3 keys five lanes see no key at all, so the combine must also
/// fold in lanes whose maximum is `-inf`.
#[test]
fn a_true_inf_mask_and_idle_lanes_do_not_make_nan() {
    let leading = with_lanes(spec(1, 2, 1, 1, 2, 4), LANES);
    let operands = Operands::new(&leading, vec![f32::NEG_INFINITY, 0.0]);
    let got = interpret(&leading, &caps(), &operands);
    let want = attention_reference(
        &leading,
        &operands.q,
        &operands.k,
        &operands.v,
        &operands.mask,
    );
    assert_close(&got, &want, 1e-5);

    let idle = with_lanes(spec(1, 2, 1, 1, 3, 4), LANES);
    let operands = Operands::new(&idle, vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]);
    let got = interpret(&idle, &caps(), &operands);
    let want = attention_reference(&idle, &operands.q, &operands.k, &operands.v, &operands.mask);
    assert_close(&got, &want, 1e-5);
}

/// A row with no visible key is `0 / 0`: NaN, as the oracle's softmax gives, and only that row.
#[test]
fn a_row_with_no_visible_key_is_nan_and_its_neighbours_are_not() {
    let spec = spec(1, 1, 1, 2, 4, 4);
    let mut mask = additive_mask(1, 2, 4, Some(&[4]));
    mask[..4].fill(f32::NEG_INFINITY); // row 0 sees nothing
    let operands = Operands::new(&spec, mask);
    let got = interpret(&spec, &caps(), &operands);
    assert!(
        got[..4].iter().all(|x| x.is_nan()),
        "row 0: {:?}",
        &got[..4]
    );
    let want = attention_reference(&spec, &operands.q, &operands.k, &operands.v, &operands.mask);
    assert_close(&got[4..], &want[4..], 1e-5);
}

/// MLA: a split key (`k_nope` per head, `k_pe` shared by every head) with a value narrower than the key
/// (`dk` 24, `dv` 16, shared part 8). A kernel that read the shared part with the head's stride, dropped it
/// from the dot product, or wrote the output with the key width differs from the reference.
#[test]
fn a_split_key_with_a_narrower_value_matches_the_reference() {
    let mla = walked(kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        dk: 24,
        dv: 16,
        key: kg::KeyLayout::Split { shared_dim: 8 },
        ..with_lanes(spec(2, 4, 1, 1, 11, 8), LANES)
    });
    let operands = Operands::new(&mla, additive_mask(2, 1, 11, Some(&[11, 6])));
    let got = interpret(&mla, &caps(), &operands);
    assert_eq!(got.len(), 2 * 4 * 16, "the output is `dv` wide");
    assert_close(&got, &operands.reference(&mla), 1e-5);

    let without_shared = Operands {
        k_shared: operands.k_shared.as_ref().map(|part| vec![0.0; part.len()]),
        ..Operands::new(&mla, operands.mask.clone())
    };
    assert!(
        max_abs_error(&got, &without_shared.reference(&mla)) > 1e-3,
        "the fixture must be sensitive to the shared part of the key"
    );
}

/// A live bound visits the first `len[b]` keys of each batch row and never reads the rest: NaN in the
/// inactive keys and values of every row leaves the output finite and equal to the reference, which never
/// reads them either. Softcap and a split key ride along.
#[test]
fn a_live_bound_never_reads_the_poisoned_tail() {
    let live = walked(kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        dk: 12,
        dv: 8,
        key: kg::KeyLayout::Split { shared_dim: 4 },
        score: kg::ScoreTransform::ScaleSoftcap {
            scale_bits: 0.5f32.to_bits(),
            cap_bits: 3.0f32.to_bits(),
        },
        bound: kg::KeyBound::Live,
        ..with_lanes(spec(3, 4, 2, 1, 37, 8), LANES)
    });
    let len = [5, 37, 1];
    let mut operands = Operands::new(&live, vec![0.0; 3 * 37]).with_len(&len);
    let (hkv, head_dim, shared_dim) = (2, 8, 4);
    for (b, &live_len) in len.iter().enumerate() {
        for t in live_len as usize..37 {
            for h in 0..hkv {
                let row = (b * hkv + h) * 37 + t;
                operands.k[row * head_dim..(row + 1) * head_dim].fill(f32::NAN);
                operands.v[row * 8..(row + 1) * 8].fill(f32::NAN);
            }
            let row = b * 37 + t;
            operands.k_shared.as_mut().unwrap()[row * shared_dim..(row + 1) * shared_dim]
                .fill(f32::NAN);
        }
    }
    let got = interpret(&live, &caps(), &operands);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    assert_close(&got, &operands.reference(&live), 1e-5);

    // The bound is the live length, not the capacity: a full-capacity bound reads the poison.
    let at_capacity = operands.with_len(&[37, 37, 37]);
    assert!(
        interpret(&live, &caps(), &at_capacity)
            .iter()
            .any(|x| x.is_nan()),
        "the fixture must put NaN where a capacity bound reads"
    );
}

/// A length outside `1..=Lk` reaches no memory past the buffers (the interpreter faults on any out-of-bounds
/// access) and selects what the decomposition selects: no key at all (a NaN row) for a length below one, every
/// key for a length past the capacity. A negative length turned into an unsigned loop bound reads every key
/// instead of none, and an unclamped one reads past the end.
#[test]
fn an_out_of_range_live_length_is_safe_and_selects_what_the_decomposition_selects() {
    let live = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        bound: kg::KeyBound::Live,
        ..with_lanes(spec(5, 2, 1, 1, 9, 4), LANES)
    };
    let len = [-1, 0, 10, i32::MIN, i32::MAX];
    let operands = Operands::new(&live, vec![0.0; 5 * 9]).with_len(&len);
    let got = interpret(&live, &caps(), &operands);
    let want = operands.reference(&live);
    let row = 2 * 4;
    for (b, &l) in len.iter().enumerate() {
        let (got, want) = (&got[b * row..(b + 1) * row], &want[b * row..(b + 1) * row]);
        if l < 1 {
            assert!(got.iter().all(|x| x.is_nan()), "len {l}: {got:?}");
            assert!(want.iter().all(|x| x.is_nan()), "len {l}: {want:?}");
        } else {
            assert_close(got, want, 1e-5);
        }
    }
    // Valid endpoints: one key and every key.
    for len in [[1, 1, 1, 1, 1], [9, 9, 9, 9, 9]] {
        let operands = Operands::new(&live, vec![0.0; 5 * 9]).with_len(&len);
        assert_close(
            &interpret(&live, &caps(), &operands),
            &operands.reference(&live),
            1e-5,
        );
    }
}

// ---- Split-KV decode attention (POOT-1113): the partials stage and the combine stage ----

/// The absolute tolerance of a split-KV row against the reference. Streaming over `lanes` lanes already
/// matches the reference to 1e-5 (the lane combine reorders the f32 sums); split-KV adds one more
/// reassociation (per-split partials, then the cross-split combine), each term of O(1) magnitude with
/// values in `[-1, 1]` and `exp(s - m) <= 1`, so the error stays at that same order. A looser bound would
/// hide a dropped split (its weight is `O(1/splits)` of the row), which the mutation rows below show.
const SPLIT_KV_TOLERANCE: f32 = 1e-5;

/// `spec` under a split-KV schedule of `splits` workgroups per row whose shares may be a single step wide, so a
/// row of `n` visited steps uses `min(splits, n)` of them.
fn with_splits(spec: kg::AttentionSpec, lanes: usize, splits: usize) -> kg::AttentionSpec {
    with_split_floor(spec, lanes, splits, 1)
}

/// [`with_splits`] with shares of at least `min_units` steps.
fn with_split_floor(
    spec: kg::AttentionSpec,
    lanes: usize,
    splits: usize,
    min_units: usize,
) -> kg::AttentionSpec {
    walked(kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        schedule: kg::AttentionSchedule::SplitKv {
            lanes,
            splits,
            min_units,
            walk: walk_of(spec.dk, spec.dv, spec.dk),
        },
        ..spec
    })
}

const LIMITS: kg::BodyLimits = kg::BodyLimits {
    max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
    max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
};

/// Run both stages over a scratch pre-filled with `scratch_fill` (the arena does not zero a slot between
/// occupants) and return the output. `workgroups` is how many workgroups each stage actually launched.
fn interpret_split(
    spec: &kg::AttentionSpec,
    caps: &DeviceCaps,
    operands: &Operands,
    scratch_fill: f32,
) -> (Vec<f32>, [usize; 2]) {
    let run = interpret_split_scratch(spec, caps, operands, scratch_fill);
    (run.out, run.workgroups)
}

/// What both stages of a split-KV attention left behind.
struct SplitRun {
    out: Vec<f32>,
    /// The partial-state scratch after stage one: the partials of each `(row, split)`, then the published
    /// split count of each row.
    scratch: Vec<f32>,
    workgroups: [usize; 2],
}

impl SplitRun {
    /// The `(o, m, l)` words of `split` of `row`.
    fn partial(&self, spec: &kg::AttentionSpec, row: usize, split: usize) -> &[f32] {
        let splits = grid_splits(spec);
        let width = spec.dv + 2;
        let at = (row * splits + split) * width;
        &self.scratch[at..at + width]
    }

    /// How many splits `row`'s stage one published.
    fn published_splits(&self, spec: &kg::AttentionSpec, row: usize) -> f32 {
        let splits = grid_splits(spec);
        self.scratch[spec.rows() * splits * (spec.dv + 2) + row]
    }
}

/// The splits per row a split schedule's grid has.
fn grid_splits(spec: &kg::AttentionSpec) -> usize {
    let kg::AttentionSchedule::SplitKv { splits, .. } = spec.schedule else {
        panic!("a split-KV schedule has partials");
    };
    splits
}

/// [`interpret_split`], keeping the scratch.
fn interpret_split_scratch(
    spec: &kg::AttentionSpec,
    caps: &DeviceCaps,
    operands: &Operands,
    scratch_fill: f32,
) -> SplitRun {
    let partials = kg::generate(&kg::KernelRequest::Attention(spec.clone()), caps, &LIMITS)
        .expect("the split request generates");
    let combine_spec = kg::AttentionCombineSpec::of(spec).expect("a split schedule combines");
    let combine = kg::generate(
        &kg::KernelRequest::AttentionCombine(combine_spec),
        caps,
        &LIMITS,
    )
    .expect("the combine request generates");
    let lanes = spec.schedule.lanes() as u32;
    let groups = |launch: &kg::Launch| [launch.grid[0] / lanes, launch.grid[1], launch.grid[2]];
    let mut buffers = operands.buffers(spec);
    let scratch = buffers.len();
    buffers.push(Buffer::from_f32s(&vec![
        scratch_fill;
        spec.partial_elems().expect(
            "a split schedule has partials"
        )
    ]));
    run(&partials.body, groups(&partials.launch), &mut buffers).unwrap();
    // The combine's params are `[partials, out]`: it runs over its own two buffers.
    let out_len = spec.batch * spec.hq * spec.lq * spec.dv;
    let mut combine_buffers = vec![
        std::mem::replace(&mut buffers[scratch], Buffer::from_f32s(&[])),
        Buffer::from_f32s(&vec![f32::NAN; out_len]),
    ];
    run(&combine.body, groups(&combine.launch), &mut combine_buffers).unwrap();
    let workgroups = |launch: &kg::Launch| {
        let [x, y, z] = groups(launch);
        (x * y * z) as usize
    };
    SplitRun {
        out: combine_buffers[1].to_f32s().unwrap(),
        scratch: combine_buffers[0].to_f32s().unwrap(),
        workgroups: [workgroups(&partials.launch), workgroups(&combine.launch)],
    }
}

/// Decode over 4 splits of a 41-key cache (a chunk of 11 keys, the last split holding 8), lane count 4 so
/// every chunk is itself ragged over the lanes. The two launches match the textbook reference.
#[test]
fn split_kv_decode_matches_the_reference() {
    let base = with_lanes(spec(2, 3, 1, 1, 41, 8), LANES);
    let split = with_splits(base, LANES, 4);
    let operands = Operands::new(&split, additive_mask(2, 1, 41, Some(&[33, 41])));
    let (got, workgroups) = interpret_split(&split, &caps(), &operands, f32::NAN);
    assert_eq!(workgroups, [2 * 3 * 4, 2 * 3], "rows x splits, then rows");
    assert_close(&got, &operands.reference(&split), SPLIT_KV_TOLERANCE);
}

/// A sink under the split schedule is counted once: only the first split's first lane starts from it. Four
/// splits of a 41-key cache, then a live length of 5 that leaves every split but the first without a key.
///
/// Mutation: start every split from the sink; the combine counts its mass four times and the heads whose sink
/// matters go red.
#[test]
fn a_sink_under_split_kv_is_counted_once() {
    let base = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        sink: kg::AttentionSink::PerHead,
        ..with_lanes(spec(2, 3, 1, 1, 41, 8), LANES)
    };
    let split = with_splits(base, LANES, 4);
    let operands = Operands::new(&split, additive_mask(2, 1, 41, Some(&[33, 41]))).with_sinks(3);
    let (got, _) = interpret_split(&split, &caps(), &operands, f32::NAN);
    assert_close(&got, &operands.reference(&split), SPLIT_KV_TOLERANCE);

    let live = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        bound: kg::KeyBound::Live,
        ..split
    };
    let operands = Operands::new(&live, additive_mask(2, 1, 41, None))
        .with_len(&[5, 41])
        .with_sinks(3);
    let (got, _) = interpret_split(&live, &caps(), &operands, f32::NAN);
    assert_close(&got, &operands.reference(&live), SPLIT_KV_TOLERANCE);
}

/// A live length of 5 against 4 splits of a 41-key cache leaves splits past the row's shares with no key to
/// visit. The scratch arrives full of NaN (an arena slot is never zeroed): those splits publish nothing and
/// the combine reads only the published ones, so the output is finite and equals the reference.
///
/// Mutation: in `attention_combine_body`, loop to `splits` instead of the published count; the stale NaN of
/// the unpublished splits reaches the combine and the output row goes NaN.
#[test]
fn a_poisoned_scratch_with_trailing_empty_splits_matches_the_reference() {
    let live = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        bound: kg::KeyBound::Live,
        ..with_lanes(spec(3, 2, 1, 1, 41, 8), LANES)
    };
    let split = with_splits(live, LANES, 4);
    let operands = Operands::new(&split, vec![0.0; 3 * 41]).with_len(&[5, 41, 12]);
    for poison in [f32::NAN, f32::INFINITY, -1.0e30] {
        let (got, _) = interpret_split(&split, &caps(), &operands, poison);
        assert!(
            got.iter().all(|x| x.is_finite()),
            "poison {poison} reached the output: {got:?}"
        );
        assert_close(&got, &operands.reference(&split), SPLIT_KV_TOLERANCE);
    }
}

/// A row whose keys are all masked inside one split (the mask hides the whole first chunk) still
/// matches: that split's `m` stays `-inf` and weighs zero in the combine.
#[test]
fn a_split_whose_keys_are_all_masked_weighs_zero() {
    let split = with_splits(with_lanes(spec(1, 2, 1, 1, 24, 4), LANES), LANES, 4);
    let mut mask = vec![0.0; 24];
    mask[..6].fill(-f32::INFINITY);
    let operands = Operands::new(&split, mask);
    let (got, _) = interpret_split(&split, &caps(), &operands, f32::NAN);
    assert_close(&got, &operands.reference(&split), SPLIT_KV_TOLERANCE);
}

/// A row with no visible key at all is the NaN the oracle's softmax gives, and its neighbour is not.
#[test]
fn a_split_row_with_no_visible_key_is_nan_and_its_neighbour_is_not() {
    let live = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        bound: kg::KeyBound::Live,
        ..with_lanes(spec(2, 1, 1, 1, 16, 4), LANES)
    };
    let split = with_splits(live, LANES, 2);
    let operands = Operands::new(&split, vec![0.0; 2 * 16]).with_len(&[0, 9]);
    let (got, _) = interpret_split(&split, &caps(), &operands, 0.0);
    assert!(got[..4].iter().all(|x| x.is_nan()), "{got:?}");
    assert_close(
        &got[4..],
        &operands.reference(&split)[4..],
        SPLIT_KV_TOLERANCE,
    );
}

/// The launch of both stages stays inside the device's grid: a fold that needs more Y rows than the
/// device allows is a typed refusal from `generate`, not a launch the device would reject (or wrap).
#[test]
fn a_split_launch_past_the_grid_is_a_typed_refusal() {
    let split = with_splits(with_lanes(spec(1, 8, 1, 1, 64, 4), LANES), LANES, 8);
    let narrow = DeviceCaps {
        max_grid: [4, 3, 65_535],
        ..caps()
    };
    // 8 heads x 8 splits = 64 workgroups over X = 4 needs 16 rows, past Y = 3.
    let error = kg::generate(
        &kg::KernelRequest::Attention(split.clone()),
        &narrow,
        &LIMITS,
    )
    .expect_err("the fold passes the Y cap");
    assert!(
        matches!(
            error,
            kg::KernelGenError::ExceedsBound {
                value: 16,
                bound: 3,
                ..
            }
        ),
        "{error:?}"
    );
    // The same request fits the real caps.
    kg::generate(&kg::KernelRequest::Attention(split), &caps(), &LIMITS)
        .expect("the real caps hold it");
}

/// More splits than keys would leave a split with no key by construction: refused as a precondition.
#[test]
fn more_splits_than_keys_is_a_typed_error() {
    let split = with_splits(with_lanes(spec(1, 2, 1, 1, 4, 4), LANES), LANES, 5);
    let error = kg::generate(&kg::KernelRequest::Attention(split), &caps(), &LIMITS)
        .expect_err("5 splits over 4 keys");
    assert!(
        matches!(
            error,
            kg::KernelGenError::ExceedsBound {
                value: 5,
                bound: 4,
                ..
            }
        ),
        "{error:?}"
    );
}

/// Positions per block table entry in the paged fixtures (and slots per pool block).
const BLOCK: usize = 4;

/// A paged key's block table over a pool of `pool_blocks` blocks: batch row `b`'s block `i` that holds a position
/// `visits` names a distinct block, in an order that is neither row-major nor increasing, so a kernel that
/// addressed the pool by position (or by `(b, i)` instead of by the block) reads the wrong rows. A block with no
/// visited position names block 0, as the engine's table does for an entry it has not written.
fn scattered_table(
    batch: usize,
    lk: usize,
    pool_blocks: usize,
    visits: impl Fn(usize, usize) -> bool,
) -> Vec<i32> {
    let blocks = lk / BLOCK;
    let mut table = Vec::with_capacity(batch * blocks);
    for b in 0..batch {
        for i in 0..blocks {
            table.push(if (i * BLOCK..(i + 1) * BLOCK).any(|t| visits(b, t)) {
                // 7 is coprime to every pool size used here, so the visited blocks never repeat.
                ((b * blocks + i) * 7 % (pool_blocks - 1) + 1) as i32
            } else {
                0
            });
        }
    }
    table
}

/// The pool slot of position `t` of batch row `b` under `table`.
fn slot_of(table: &[i32], lk: usize, b: usize, t: usize) -> usize {
    table[b * (lk / BLOCK) + t / BLOCK] as usize * BLOCK + t % BLOCK
}

/// The dense `[B, Hkv, Lk, D]` operand a paged read of `pool` through `table` stands for.
fn gather_pool(
    pool: &[f32],
    table: &[i32],
    (batch, hkv, lk, d): (usize, usize, usize, usize),
) -> Vec<f32> {
    let mut dense = vec![0.0; batch * hkv * lk * d];
    for b in 0..batch {
        for g in 0..hkv {
            for t in 0..lk {
                let from = (slot_of(table, lk, b, t) * hkv + g) * d;
                let to = ((b * hkv + g) * lk + t) * d;
                dense[to..to + d].copy_from_slice(&pool[from..from + d]);
            }
        }
    }
    dense
}

/// `operands`' pools with NaN in every pool row a position `visits` does not name, so a kernel that reads one of
/// them (past a live length, outside a window, in an unwritten block) is caught by the output.
fn poison_unvisited(
    operands: &mut Operands,
    spec: &kg::AttentionSpec,
    visits: impl Fn(usize, usize) -> bool,
) {
    let table = operands.read.clone().expect("a paged key");
    let hkv = spec.hq / spec.n_rep;
    for b in 0..spec.batch {
        for t in (0..spec.lk).filter(|&t| !visits(b, t)) {
            for g in 0..hkv {
                let row = slot_of(&table, spec.lk, b, t) * hkv + g;
                operands.k[row * spec.dk..(row + 1) * spec.dk].fill(f32::NAN);
                operands.v[row * spec.dv..(row + 1) * spec.dv].fill(f32::NAN);
            }
        }
    }
}

/// A paged key reads the pools in place: the kernel over `(pool, table)` equals the dense kernel over the
/// views the block table gathers, and the textbook reference, with a ragged batch of visible lengths, GQA and a
/// softcap riding along. The pool is larger than any one row's keys and the blocks are scattered.
#[test]
fn a_paged_key_equals_the_dense_kernel_over_its_gathered_views() {
    let (batch, hq, n_rep, lk, d, pool_blocks) = (3, 4, 2, 16, 8, 17);
    let visible = [16, 5, 9];
    let paged = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        key: kg::KeyLayout::Paged { block: BLOCK },
        score: kg::ScoreTransform::ScaleSoftcap {
            scale_bits: 0.5f32.to_bits(),
            cap_bits: 3.0f32.to_bits(),
        },
        ..with_lanes(spec(batch, hq, n_rep, 1, lk, d), LANES)
    };
    let mask = additive_mask(batch, 1, lk, Some(&visible));
    let read = scattered_table(batch, lk, pool_blocks, |b, t| t < visible[b]);
    let operands =
        Operands::new(&paged, mask.clone()).with_pools(&paged, pool_blocks, read.clone());
    let got = interpret(&paged, &caps(), &operands);
    assert!(got.iter().all(|x| x.is_finite()), "{got:?}");
    assert_close(&got, &operands.reference(&paged), 1e-5);

    // The same attention as a dense kernel over the gathered views.
    let dense_spec = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        key: kg::KeyLayout::Single,
        ..paged.clone()
    };
    let hkv = hq / n_rep;
    let shape = (batch, hkv, lk, d);
    let mut dense = Operands::new(&dense_spec, mask);
    dense.k = gather_pool(&operands.k, &read, shape);
    dense.v = gather_pool(&operands.v, &read, shape);
    assert_close(&got, &interpret(&dense_spec, &caps(), &dense), 1e-5);
}

/// A paged key under a live bound and the split-KV schedule never reads the pool rows past a row's length:
/// the rows of positions there hold NaN, and the output is finite and equals the reference.
#[test]
fn a_paged_split_kv_with_a_live_bound_never_reads_past_the_length() {
    let (batch, hq, n_rep, lk, d, pool_blocks) = (2, 4, 2, 44, 8, 33);
    let live = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        key: kg::KeyLayout::Paged { block: BLOCK },
        bound: kg::KeyBound::Live,
        ..with_splits(spec(batch, hq, n_rep, 1, lk, d), LANES, 4)
    };
    let len = [9, 41];
    let visits = |b: usize, t: usize| t < len[b] as usize;
    let table = scattered_table(batch, lk, pool_blocks, visits);
    let mut operands = Operands::new(&live, vec![0.0; batch * lk])
        .with_pools(&live, pool_blocks, table)
        .with_len(&len);
    poison_unvisited(&mut operands, &live, visits);
    let (got, _) = interpret_split(&live, &caps(), &operands, f32::NAN);
    assert!(got.iter().all(|x| x.is_finite()), "{got:?}");
    assert_close(&got, &operands.reference(&live), SPLIT_KV_TOLERANCE);
}

/// The `[B, 1, Lq, Lk]` additive mask of the visibility a causal mask states: key `t` is visible to query row
/// `(b, i)` when `t <= max(pos, 0)` and, under a window, `pos - t < window`.
fn mask_of_positions(
    batch: usize,
    lq: usize,
    lk: usize,
    positions: &[i32],
    window: Option<usize>,
) -> Vec<f32> {
    let mut mask = Vec::new();
    for b in 0..batch {
        for i in 0..lq {
            let position = positions[b * lq + i].max(0) as usize;
            for t in 0..lk {
                let visible = t <= position && window.is_none_or(|w| position - t < w);
                mask.push(if visible { 0.0 } else { -1.0e9 });
            }
        }
    }
    mask
}

fn causal(spec: kg::AttentionSpec, window: Option<usize>) -> kg::AttentionSpec {
    kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        mask: kg::MaskKind::Causal { window },
        ..spec
    }
}

/// Key and value rows `t` of batch row `b`, every head (and a split key's shared part), set to NaN: a kernel that
/// reads one has poisoned its output.
fn poison_key(operands: &mut Operands, spec: &kg::AttentionSpec, b: usize, t: usize) {
    let hkv = spec.hq / spec.n_rep;
    let (head_dim, shared_dim) = match spec.key {
        kg::KeyLayout::Single | kg::KeyLayout::Paged { .. } => (spec.dk, 0),
        kg::KeyLayout::Split { shared_dim } => (spec.dk - shared_dim, shared_dim),
    };
    for h in 0..hkv {
        let row = (b * hkv + h) * spec.lk + t;
        operands.k[row * head_dim..(row + 1) * head_dim].fill(f32::NAN);
        operands.v[row * spec.dv..(row + 1) * spec.dv].fill(f32::NAN);
    }
    if let Some(k_shared) = operands.k_shared.as_mut() {
        let row = b * spec.lk + t;
        k_shared[row * shared_dim..(row + 1) * shared_dim].fill(f32::NAN);
    }
}

/// A causal mask stated by its positions equals the textbook attention under the visibility it names and the
/// additive kernel under the mask tensor of that visibility: a decode over ragged lanes, a windowed prefill
/// (every row a different window), and a long decode whose window starts at a key no lane begins on, so each
/// lane's first key moves to its own stride.
///
/// Mutation: in `attention_body`, start every causal row at its lane's first key (drop the window start); the
/// windowed rows read keys the window hides and go red.
#[test]
fn a_causal_mask_is_the_additive_mask_of_its_positions() {
    let rows = [
        (with_lanes(spec(2, 4, 2, 1, 13, 8), 64), vec![4, 12], None),
        (
            with_lanes(spec(1, 4, 2, 9, 9, 8), LANES),
            (0..9).collect(),
            Some(3),
        ),
        (
            with_lanes(spec(2, 2, 1, 1, 40, 8), LANES),
            vec![37, 20],
            Some(5),
        ),
        (
            with_lanes(spec(1, 2, 1, 1, 40, 8), LANES),
            vec![39],
            Some(40),
        ),
    ];
    for (additive, positions, window) in rows {
        let structural = causal(additive.clone(), window);
        let mask = mask_of_positions(additive.batch, additive.lq, additive.lk, &positions, window);
        let by_mask = Operands::new(&additive, mask);
        let by_positions = Operands::new(&structural, Vec::new()).with_positions(&positions);
        let got = interpret(&structural, &caps(), &by_positions);
        assert_close(&got, &by_positions.reference(&structural), 1e-5);
        assert_close(&got, &interpret(&additive, &caps(), &by_mask), 1e-5);
    }
}

/// A causal row never reads a key it cannot see: the keys past the position and before the window carry NaN,
/// and the output is finite and equal to the reference. The additive kernel reads every key and takes the NaN.
///
/// Mutation: in `attention_body`, leave the causal row's `key_end` at the capacity; the poison past the
/// position reaches the row and the output goes NaN.
#[test]
fn a_causal_row_never_reads_a_key_it_cannot_see() {
    let spec = causal(with_lanes(spec(2, 2, 1, 1, 30, 8), LANES), Some(6));
    let positions = [17, 5];
    let mut operands = Operands::new(&spec, Vec::new()).with_positions(&positions);
    for (b, &position) in positions.iter().enumerate() {
        for t in 0..30 {
            if t > position as usize || position as usize - t >= 6 {
                poison_key(&mut operands, &spec, b, t);
            }
        }
    }
    let got = interpret(&spec, &caps(), &operands);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    assert_close(&got, &operands.reference(&spec), 1e-5);

    let additive = kg::AttentionSpec {
        kv: poot_kernelgen::KvLane::F32,
        mask: kg::MaskKind::Broadcast,
        ..spec.clone()
    };
    let by_mask = Operands {
        mask: mask_of_positions(2, 1, 30, &positions, Some(6)),
        positions: None,
        ..operands
    };
    assert!(
        interpret(&additive, &caps(), &by_mask)
            .iter()
            .any(|x| x.is_nan()),
        "the fixture must put NaN where an every-key walk reads"
    );
}

/// A padded query (position `-1`) sees key 0 and no NaN reaches its row or its neighbour's.
#[test]
fn a_causal_row_at_a_negative_position_sees_the_first_key() {
    let spec = causal(with_lanes(spec(2, 2, 1, 1, 9, 8), LANES), None);
    let operands = Operands::new(&spec, Vec::new()).with_positions(&[-1, 6]);
    let got = interpret(&spec, &caps(), &operands);
    assert!(got.iter().all(|x| x.is_finite()), "{got:?}");
    assert_close(&got, &operands.reference(&spec), 1e-5);
    // One visible key: the row is that key's value.
    let row = spec.hq * spec.dv;
    assert_close(&got[..spec.dv], &operands.v[..spec.dv], 1e-5);
    assert_eq!(got.len(), 2 * row);
}

/// A causal row under the split schedule: 4 splits of a 41-key cache with a window, so the row's shares cover
/// only the window, and the keys outside the window are poison.
#[test]
fn a_causal_row_under_split_kv_matches_the_reference() {
    let split = with_splits(
        causal(with_lanes(spec(2, 3, 1, 1, 41, 8), LANES), Some(9)),
        LANES,
        4,
    );
    let positions = [40, 12];
    let mut operands = Operands::new(&split, Vec::new()).with_positions(&positions);
    for (b, &position) in positions.iter().enumerate() {
        for t in 0..41 {
            if t > position as usize || position as usize - t >= 9 {
                poison_key(&mut operands, &split, b, t);
            }
        }
    }
    let (got, _) = interpret_split(&split, &caps(), &operands, f32::NAN);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    assert_close(&got, &operands.reference(&split), SPLIT_KV_TOLERANCE);
}

/// The splits of each row a split-KV attention made partials for, from the live length alone. Six rows of one
/// attention: a grid of 8 splits, shares of at least 16 steps over 4 lanes. Hand-derived: live length 1, 5 and
/// 16 fit one share; 17 needs 2 (balanced to 3 lane steps of 4 keys, 12 keys each, so 2 shares cover it);
/// 100 needs 7 (25 lane steps over 7 shares is 4 each, 16 keys, 7 cover 100); 1024 fills all 8.
const ROW_SHARES: [(i32, usize); 6] = [(1, 1), (5, 1), (16, 1), (17, 2), (100, 7), (1024, 8)];

/// The combine reads only the splits stage one published, and a split past a row's shares writes nothing. The
/// scratch arrives full of NaN and a row of live length `n` uses `ROW_SHARES` of the 8 splits of its grid: the
/// output equals the reference and is finite, the row publishes its count, the other splits' partials are
/// still NaN, and the used ones are finite.
///
/// Mutation: in `attention_combine_body`, loop to `splits` instead of the published count; the combine reads
/// the NaN of the unused splits and every short row goes NaN.
/// Mutation: in `attention_body`, run the workgroups past a row's shares too; they publish neutral partials
/// into slots that must stay untouched and the NaN check goes red.
#[test]
fn the_combine_reads_only_the_splits_stage_one_published() {
    let live = kg::AttentionSpec {
        bound: kg::KeyBound::Live,
        ..with_split_floor(with_lanes(spec(6, 2, 1, 1, 1024, 8), LANES), LANES, 8, 16)
    };
    let lens = ROW_SHARES.map(|(len, _)| len);
    let operands = Operands::new(&live, vec![0.0; 6 * 1024]).with_len(&lens);
    let run = interpret_split_scratch(&live, &caps(), &operands, f32::NAN);
    assert!(run.out.iter().all(|x| x.is_finite()), "{:?}", run.out);
    assert_close(&run.out, &operands.reference(&live), SPLIT_KV_TOLERANCE);
    for (batch, &(len, shares)) in ROW_SHARES.iter().enumerate() {
        for row in [2 * batch, 2 * batch + 1] {
            assert_eq!(
                run.published_splits(&live, row),
                shares as f32,
                "row {row}, live length {len}"
            );
            for split in 0..8 {
                let partial = run.partial(&live, row, split);
                if split < shares {
                    assert!(
                        partial.iter().all(|x| x.is_finite()),
                        "row {row} split {split} has a share of {len} keys: {partial:?}"
                    );
                } else {
                    assert!(
                        partial.iter().all(|x| x.is_nan()),
                        "row {row} split {split} has no share of {len} keys but wrote {partial:?}"
                    );
                }
            }
        }
    }
}

/// The published count is the shares of a balanced cut with none empty. A grid of 8 splits, shares of at least
/// 2 steps over 4 lanes and a cache of 64 keys; hand-derived per live length `n` as
/// `s = min(8, ceil(n / 2))` shares of `ceil(ceil(n / 4) / s)` lane steps (4 keys each), recounted as
/// `ceil(n / keys per share)`: n 1 -> 1, n 2 -> 1, n 3 -> `s` 2, 1 lane step, 4 keys per share, recount 1;
/// n 33 -> `s` 8, 2 lane steps, 8 keys, recount 5; n 64 -> 8 shares of 8 keys. Every published split has a
/// finite maximum, so none is an empty share.
///
/// Mutation: in `split_shares`, publish `shares` and skip the recount; n 3 publishes 2 splits and n 33 publishes
/// 8, with empty shares whose maximum is `-inf`.
#[test]
fn no_published_split_is_an_empty_share() {
    let live = kg::AttentionSpec {
        bound: kg::KeyBound::Live,
        ..with_split_floor(with_lanes(spec(5, 1, 1, 1, 64, 8), LANES), LANES, 8, 2)
    };
    let expected = [(1, 1), (2, 1), (3, 1), (33, 5), (64, 8)];
    let lens = expected.map(|(len, _)| len);
    let operands = Operands::new(&live, vec![0.0; 5 * 64]).with_len(&lens);
    let run = interpret_split_scratch(&live, &caps(), &operands, f32::NAN);
    assert_close(&run.out, &operands.reference(&live), SPLIT_KV_TOLERANCE);
    for (row, &(len, shares)) in expected.iter().enumerate() {
        assert_eq!(
            run.published_splits(&live, row),
            shares as f32,
            "live length {len}"
        );
        for split in 0..shares {
            let [.., m, _l] = run.partial(&live, row, split) else {
                unreachable!("a partial has o, m and l")
            };
            assert!(
                m.is_finite(),
                "live length {len}: split {split} of {shares} saw no key"
            );
        }
    }
}

/// A decode attention over a cache of `lk` keys with a grid of `splits`, its live length a run-time operand.
fn capacity_spec(lk: usize, splits: usize) -> kg::AttentionSpec {
    kg::AttentionSpec {
        bound: kg::KeyBound::Live,
        ..causal(
            with_split_floor(
                with_lanes(spec(1, 2, 1, 1, lk, 8), LANES),
                LANES,
                splits,
                128,
            ),
            None,
        )
    }
}

/// The same live prefix of two caches of different capacity and grid gives the same bits. The prefix is the
/// first `len` keys of a 256-key cache and of a 4096-key cache whose keys past 256 are NaN; the grids are 2
/// and 16 splits with shares of at least 128 steps, so a row of at most 256 keys has at most 2 shares in
/// both. The partition depends on the live length alone, so the rows are bit-identical (and finite: no key
/// past the length is read). Live lengths 5, 128, 129, 200 and 256 cover one share, the edge of the floor, and
/// two shares with the second ragged and full.
///
/// Mutation: in `attention_body`, cut the shares from the capacity (`per = ceil(lk / splits)`, every split
/// published); the 129-key row is two shares of 128 and 1 keys on the 256-key cache and one on the 4096-key
/// cache, and differs in bits.
#[test]
fn the_partition_does_not_depend_on_the_cache_capacity() {
    let (small, big) = (capacity_spec(256, 2), capacity_spec(4096, 16));
    assert_ne!(
        small.workgroups(),
        big.workgroups(),
        "the two capacities plan different grids"
    );
    let (hkv, d) = (2, 8);
    for len in [5, 128, 129, 200, 256] {
        let small_operands = Operands::new(&small, Vec::new())
            .with_positions(&[255])
            .with_len(&[len]);
        let widen = |plane: &[f32]| {
            let mut wide = vec![f32::NAN; hkv * 4096 * d];
            for head in 0..hkv {
                wide[head * 4096 * d..][..256 * d]
                    .copy_from_slice(&plane[head * 256 * d..][..256 * d]);
            }
            wide
        };
        let big_operands = Operands {
            k: widen(&small_operands.k),
            v: widen(&small_operands.v),
            q: small_operands.q.clone(),
            positions: small_operands.positions.clone(),
            len: small_operands.len.clone(),
            ..Operands::new(&big, Vec::new())
        };
        let (from_small, _) = interpret_split(&small, &caps(), &small_operands, f32::NAN);
        let (from_big, _) = interpret_split(&big, &caps(), &big_operands, f32::NAN);
        assert!(from_small.iter().all(|x| x.is_finite()), "len {len}");
        let bits = |row: &[f32]| row.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&from_small), bits(&from_big), "live length {len}");
    }
}

/// A windowed decode partitions its window, not its cache: position 900 of a 1024-key cache under a window of
/// 64 visits 64 keys, which a grid of 8 splits with shares of at least 16 steps cuts into 4. The keys outside
/// the window are NaN, so a share that strayed outside it would poison the output.
///
/// Mutation: in `attention_body`, start a causal row's range at 0 instead of its window's first key; the
/// shares reach the poisoned keys before the window and the output goes NaN (and the row would publish 8
/// splits).
#[test]
fn a_window_is_partitioned_not_the_cache() {
    let windowed = causal(
        with_split_floor(with_lanes(spec(1, 2, 1, 1, 1024, 8), LANES), LANES, 8, 16),
        Some(64),
    );
    let mut operands = Operands::new(&windowed, Vec::new()).with_positions(&[900]);
    for t in (0..1024).filter(|&t| !(837..=900).contains(&t)) {
        poison_key(&mut operands, &windowed, 0, t);
    }
    let run = interpret_split_scratch(&windowed, &caps(), &operands, f32::NAN);
    assert!(run.out.iter().all(|x| x.is_finite()), "{:?}", run.out);
    assert_close(&run.out, &operands.reference(&windowed), SPLIT_KV_TOLERANCE);
    for row in 0..2 {
        assert_eq!(run.published_splits(&windowed, row), 4.0, "row {row}");
    }
}

// ---- The clustered key walk (POOT-1554): the matrix, the butterfly, the resource claim, GQA sharing ----

/// The workgroup width of the matrix rows: two clusters at the widest cluster, eight at the narrowest.
const MATRIX_LANES: usize = 64;

/// The walk the matrix gives a key `dk` wide and a value `dv` wide of `kv` elements: a run of what one 16-byte
/// load covers, grown only while one 32-lane cluster could not cover the wider head, and the cluster that does.
fn matrix_walk(dk: usize, dv: usize, kv: kg::KvLane) -> kg::KeyWalk {
    let mut run = 16 / kv.bytes();
    while dk.max(dv).div_ceil(run) > 32 {
        run *= 2;
    }
    kg::KeyWalk {
        cluster: dk.max(dv).div_ceil(run).next_power_of_two(),
        run,
    }
}

/// One way the matrix varies everything but the shape: the mask, the bound, a sink, a softcap and the key layout.
#[derive(Clone, Copy)]
struct Variant {
    mask: kg::MaskKind,
    live: bool,
    sink: bool,
    softcap: bool,
    paged: bool,
}

/// Seven variants, so that stepping through the matrix by one cell cycles all of them against every other axis
/// (7 is coprime to the 5 head widths x 5 group sizes x 3 lanes x 2 schedules). Between them every mask kind, both
/// bounds, a sink, a softcap and both single-part layouts occur, a causal window shorter and longer than the cache.
const VARIANTS: [Variant; 7] = [
    Variant {
        mask: kg::MaskKind::Broadcast,
        live: false,
        sink: false,
        softcap: false,
        paged: false,
    },
    Variant {
        mask: kg::MaskKind::PerHead,
        live: true,
        sink: true,
        softcap: true,
        paged: false,
    },
    Variant {
        mask: kg::MaskKind::Causal { window: None },
        live: false,
        sink: true,
        softcap: false,
        paged: false,
    },
    Variant {
        mask: kg::MaskKind::Causal { window: Some(11) },
        live: true,
        sink: false,
        softcap: true,
        paged: false,
    },
    Variant {
        mask: kg::MaskKind::Broadcast,
        live: true,
        sink: true,
        softcap: false,
        paged: true,
    },
    Variant {
        mask: kg::MaskKind::PerHead,
        live: false,
        sink: false,
        softcap: true,
        paged: true,
    },
    Variant {
        mask: kg::MaskKind::Causal { window: Some(40) },
        live: false,
        sink: false,
        softcap: false,
        paged: true,
    },
];

/// The `[B, Hq, 1, Lk]` ALiBi-like mask of the matrix: a per-head slope over the distance to a per-row position,
/// and a hard `-1e9` past it.
fn per_head_bias(batch: usize, hq: usize, lk: usize) -> Vec<f32> {
    let mut mask = Vec::new();
    for b in 0..batch {
        for h in 0..hq {
            for t in 0..lk {
                let position = 30 + 7 * b;
                mask.push(if t <= position {
                    -0.02 * (h + 1) as f32 * (position - t) as f32
                } else {
                    -1.0e9
                });
            }
        }
    }
    mask
}

/// One cell of the matrix: the spec and operands of a decode over a 37-key cache of two batch rows and two key/value
/// heads, with the live length, causal position and window of `variant`. Every key a row cannot visit carries NaN
/// (a dense key in its rows, a paged one in the slot its read names), so a walk that strays shows in the output.
fn matrix_cell(
    (dk, dv, shared): (usize, usize, Option<usize>),
    n_rep: usize,
    kv: kg::KvLane,
    split: bool,
    variant: Variant,
) -> (kg::AttentionSpec, Operands) {
    // A split key has no paged form: the variants that page it keep its two parts.
    let paged = variant.paged && shared.is_none();
    // A paged key's block table covers whole blocks.
    let (batch, hkv, lq, lk, pool_blocks) = (
        2usize,
        2usize,
        1usize,
        if paged { 40usize } else { 37 },
        25usize,
    );
    let hq = hkv * n_rep;
    let walk = matrix_walk(dk, dv, kv);
    let schedule = if split {
        kg::AttentionSchedule::SplitKv {
            lanes: MATRIX_LANES,
            splits: 3,
            min_units: 2,
            walk,
        }
    } else {
        kg::AttentionSchedule::Streaming {
            lanes: MATRIX_LANES,
            walk,
        }
    };
    let scale_bits = (1.0 / (dk as f32).sqrt()).to_bits();
    let spec = kg::AttentionSpec {
        kv,
        batch,
        hq,
        n_rep,
        lq,
        lk,
        dk,
        dv,
        key: match shared {
            Some(shared_dim) => kg::KeyLayout::Split { shared_dim },
            None if paged => kg::KeyLayout::Paged { block: BLOCK },
            None => kg::KeyLayout::Single,
        },
        score: if variant.softcap {
            kg::ScoreTransform::ScaleSoftcap {
                scale_bits,
                cap_bits: 4.0f32.to_bits(),
            }
        } else {
            kg::ScoreTransform::Scale { scale_bits }
        },
        mask: variant.mask,
        bound: if variant.live {
            kg::KeyBound::Live
        } else {
            kg::KeyBound::Static
        },
        sink: if variant.sink {
            kg::AttentionSink::PerHead
        } else {
            kg::AttentionSink::Absent
        },
        schedule,
        precision: poot_graph_ir::OperandPrecision::Exact,
    };
    let positions = [30, 12];
    let lens = [29, 37];
    let window = match variant.mask {
        kg::MaskKind::Causal { window } => window,
        _ => None,
    };
    // Whether batch row `b`'s walk visits key `t`: inside the live length and, for a causal row, inside its window.
    let visits = |b: usize, t: usize| {
        let live = !variant.live || t < lens[b] as usize;
        let causal = !matches!(variant.mask, kg::MaskKind::Causal { .. })
            || (t <= positions[b] as usize && window.is_none_or(|w| positions[b] as usize - t < w));
        live && causal
    };
    let mask = match variant.mask {
        kg::MaskKind::Broadcast => additive_mask(batch, lq, lk, Some(&[30, 37])),
        kg::MaskKind::PerHead => per_head_bias(batch, hq, lk),
        kg::MaskKind::Causal { .. } | kg::MaskKind::Blocks { .. } => Vec::new(),
    };
    let mut operands = Operands::new(&spec, mask);
    if paged {
        // The pool rows no visited position names hold NaN: a walk that strays shows in the output.
        let table = scattered_table(batch, lk, pool_blocks, visits);
        operands = operands.with_pools(&spec, pool_blocks, table);
        poison_unvisited(&mut operands, &spec, visits);
    }
    if matches!(variant.mask, kg::MaskKind::Causal { .. }) {
        operands = operands.with_positions(&positions);
    }
    if variant.live {
        operands = operands.with_len(&lens);
    }
    if variant.sink {
        operands = operands.with_sinks(hq);
    }
    if !paged {
        for b in 0..batch {
            for t in (0..lk).filter(|&t| !visits(b, t)) {
                poison_key(&mut operands, &spec, b, t);
            }
        }
    }
    (spec, operands.stored(kv))
}

/// Row 6 of the POOT-1549 design: the clustered body matches the f64 reference across the matrix. Head widths
/// `dk/dv` of 64/64, 96/96 (12 of a cluster of 16 lanes active), 128/128, 256/256 (a run of 8 across a full cluster of
/// 32) and the MLA 192/128 split key (a value narrower than the key, the run inside one part); group sizes `n_rep` of 1, 2, 6, 7 and 8;
/// f32, f16 and bf16 keys and values; the additive, per-head and causal masks (with a window shorter and longer
/// than the cache), static and live bounds, a sink, a softcap, single, split and paged keys; the streaming and the
/// split-KV stage one with its combine. 150 cells, each variant of [`VARIANTS`] against every shape. A key a row
/// does not visit carries NaN.
///
/// Mutation: skip the butterfly's last round (`distance > 1` in the round loop); the lanes of a cluster hold
/// partial scores that differ, and the cells go red. Mutation: start a lane's run at `j * run + 1`; every score
/// reads the wrong channels. Mutation: stride the value rows by `dk` instead of `dv`; the 192/128 cells go red.
#[test]
fn the_clustered_body_matches_the_reference_across_the_matrix() {
    const DIMS: [(usize, usize, Option<usize>); 5] = [
        (64, 64, None),
        (96, 96, None),
        (128, 128, None),
        (256, 256, None),
        (192, 128, Some(64)),
    ];
    const GROUPS: [usize; 5] = [1, 2, 6, 7, 8];
    let mut cell = 0;
    for dims in DIMS {
        for n_rep in GROUPS {
            for kv in [kg::KvLane::F32, kg::KvLane::F16, kg::KvLane::Bf16] {
                for split in [false, true] {
                    let variant = VARIANTS[cell % VARIANTS.len()];
                    cell += 1;
                    let (spec, operands) = matrix_cell(dims, n_rep, kv, split, variant);
                    let label = format!(
                        "cell {cell}: dk/dv {}/{} n_rep {n_rep} {kv:?} split {split} {:?} live {} sink {} softcap {} paged {}",
                        dims.0,
                        dims.1,
                        variant.mask,
                        variant.live,
                        variant.sink,
                        variant.softcap,
                        variant.paged,
                    );
                    let got = if split {
                        interpret_split(&spec, &caps(), &operands, f32::NAN).0
                    } else {
                        interpret(&spec, &caps(), &operands)
                    };
                    assert!(got.iter().all(|x| x.is_finite()), "{label}: {got:?}");
                    let want = operands.reference(&spec);
                    let error = max_abs_error(&got, &want);
                    assert!(
                        error <= SPLIT_KV_TOLERANCE,
                        "{label}: max abs error {error:e} against the reference"
                    );
                }
            }
        }
    }
    assert_eq!(cell, 150);
}

/// Row 7: every lane of a cluster holds the bit-identical score after the butterfly, so the redundant online
/// softmax keeps one `(m, l)` per cluster without a broadcast. The state is not observable from outside (the
/// kernel has no debug output), so it is checked through behaviour: scores spread wide enough that a lane whose
/// maximum or denominator differs from its neighbour's weighs its value run on another scale than the rest of the
/// cluster, so the output row is inconsistent with the reference. Cluster widths 2 to 32 of one head, so every round
/// count of the butterfly (1 to 5) runs; the query is scaled so the softmax is peaked, and the row is checked
/// against the reference and against the same row at every other width: one bit pattern of scores for all of them.
///
/// Mutation: reduce into lane 0 of each cluster only (`part += shuffle` where `j == 0`, the others keep their
/// partial); lanes other than 0 hold partial scores, their `m` and `l` diverge from lane 0's, and the first width
/// goes red on the reference comparison.
#[test]
fn every_lane_of_a_cluster_scores_alike() {
    let (batch, hq, n_rep, lk, d) = (1, 4, 2, 21, 64);
    let mut rows: Vec<(usize, Vec<f32>)> = Vec::new();
    for cluster in [2usize, 4, 8, 16, 32] {
        let walk = kg::KeyWalk {
            cluster,
            run: d / cluster,
        };
        let base = kg::AttentionSpec {
            score: kg::ScoreTransform::Scale {
                scale_bits: 2.0f32.to_bits(),
            },
            schedule: kg::AttentionSchedule::Streaming { lanes: 64, walk },
            ..spec(batch, hq, n_rep, 1, lk, d)
        };
        let operands = Operands::new(&base, additive_mask(batch, 1, lk, Some(&[lk])));
        let got = interpret(&base, &caps(), &operands);
        assert_close(&got, &operands.reference(&base), 1e-5);
        rows.push((cluster, got));
    }
    let (first, rest) = rows.split_first().unwrap();
    for (cluster, row) in rest {
        assert_close(row, &first.1, 1e-5);
        assert!(
            max_abs_error(row, &first.1) <= 1e-5,
            "cluster {cluster} differs from cluster {}",
            first.0
        );
    }
}

/// `reach[a][b]`: whether control can get from block `a` to block `b` in one step or more. A block is on a cycle when
/// it reaches itself.
fn reachability(body: &poot_kernel_ir::Body) -> Vec<Vec<bool>> {
    use poot_kernel_ir::Terminator;
    let successors = |block: usize| -> Vec<usize> {
        match &body.blocks[block].terminator {
            Terminator::Goto { target }
            | Terminator::Barrier { target }
            | Terminator::ThreadIndexCall { target, .. } => vec![target.index as usize],
            Terminator::SwitchInt { targets, .. } => targets
                .branches
                .iter()
                .map(|(_, target)| target.index as usize)
                .chain([targets.otherwise.index as usize])
                .collect(),
            Terminator::Return | Terminator::Trap { .. } => Vec::new(),
        }
    };
    (0..body.blocks.len())
        .map(|from| {
            let mut seen = vec![false; body.blocks.len()];
            let mut work = successors(from);
            while let Some(next) = work.pop() {
                if !std::mem::replace(&mut seen[next], true) {
                    work.extend(successors(next));
                }
            }
            seen
        })
        .collect()
}

/// Row 8: the schedule's resource claim (ADR-0112 section 4). The generated stage one declares workgroup memory of
/// exactly `clusters * n_rep * (dv + 2)` words and holds exactly one barrier, outside every loop, and the key loop
/// (the cycle that holds the butterfly's shuffle) neither reads nor writes workgroup memory: the walk keeps its
/// state in registers. Checked for the streaming and the split-KV schedule, on a key 192 wide and a value 128 wide.
///
/// Mutation: keep `o` in workgroup memory with a per-key read-modify-write (a second array of `lanes * dv` words,
/// read and written in the key loop); the declared bytes and the key-loop claim both go red.
#[test]
fn the_walk_keeps_its_state_in_registers() {
    use poot_kernel_ir::{Rvalue, Statement, Terminator};
    let (dk, dv, n_rep, lanes) = (192usize, 128usize, 6usize, 128usize);
    let walk = kg::KeyWalk {
        cluster: 32,
        run: 8,
    };
    let base = kg::AttentionSpec {
        kv: kg::KvLane::F16,
        key: kg::KeyLayout::Split { shared_dim: 64 },
        dk,
        dv,
        schedule: kg::AttentionSchedule::Streaming { lanes, walk },
        ..spec(2, 4 * n_rep, n_rep, 1, 50, 8)
    };
    let split = kg::AttentionSpec {
        schedule: kg::AttentionSchedule::SplitKv {
            lanes,
            splits: 3,
            min_units: 1,
            walk,
        },
        ..base.clone()
    };
    for spec in [base, split] {
        let body = kg::generate(
            &kg::KernelRequest::Attention(spec.clone()),
            &caps(),
            &LIMITS,
        )
        .expect("the request generates")
        .body;
        let words = (lanes / walk.cluster) * n_rep * (dv + 2);
        assert_eq!(
            kg::BodyUse::measure(&body).unwrap().lds_bytes,
            4 * words as u64,
            "{:?}: the end combine's slab is the whole workgroup memory",
            spec.schedule
        );
        let reach = reachability(&body);
        let barriers: Vec<usize> = (0..body.blocks.len())
            .filter(|&block| matches!(body.blocks[block].terminator, Terminator::Barrier { .. }))
            .collect();
        assert_eq!(barriers.len(), 1, "{:?}: one barrier", spec.schedule);
        assert!(
            !reach[barriers[0]][barriers[0]],
            "{:?}: the barrier is outside every loop",
            spec.schedule
        );
        let shuffles: Vec<usize> = (0..body.blocks.len())
            .filter(|&block| {
                body.blocks[block].statements.iter().any(|statement| {
                    matches!(
                        statement,
                        Statement::Assign(_, Rvalue::SubgroupShuffleXor { .. })
                    )
                })
            })
            .collect();
        assert!(
            !shuffles.is_empty(),
            "{:?}: the walk shuffles",
            spec.schedule
        );
        // The key loop holds every block that shares a cycle with a shuffle.
        let in_key_loop = |block: usize| {
            shuffles.iter().any(|&shuffle| {
                shuffle == block || (reach[shuffle][block] && reach[block][shuffle])
            })
        };
        let touches_workgroup_memory = |statement: &Statement| {
            matches!(
                statement,
                Statement::WorkgroupLocalWrite { .. }
                    | Statement::Assign(_, Rvalue::WorkgroupLocalRead { .. })
            )
        };
        let strays: Vec<usize> = (0..body.blocks.len())
            .filter(|&block| {
                in_key_loop(block)
                    && body.blocks[block]
                        .statements
                        .iter()
                        .any(touches_workgroup_memory)
            })
            .collect();
        assert!(
            strays.is_empty(),
            "{:?}: blocks {strays:?} of the key loop touch workgroup memory",
            spec.schedule
        );
    }
}

/// Row 9: GQA sharing. The grid is one workgroup per `(batch, key/value head, query row[, split])`, and the body
/// reads each key and value element once per query row that visits it, however many query heads share them: the
/// interpreter counts reads per buffer element. A 2-batch, 2-kv-head, 4-heads-per-kv-head, 2-row decode over 50 keys
/// (not a multiple of the 8 clusters of a step, so the last step is ragged) reads every element of `k` and `v`
/// exactly twice (the two query rows), under both schedules.
///
/// Mutation: `heads_per_workgroup` returns 1; the grid assertion (`workgroups() == batch * hkv * lq * splits`) goes
/// red, and a body launched on the old grid reads every element `n_rep` times.
#[test]
fn every_key_is_read_once_per_workgroup_whatever_the_group_size() {
    let (batch, hkv, n_rep, lq, lk, d) = (2usize, 2usize, 4usize, 2usize, 50usize, 64usize);
    let walk = kg::KeyWalk { cluster: 8, run: 8 };
    let streaming = kg::AttentionSpec {
        schedule: kg::AttentionSchedule::Streaming { lanes: 64, walk },
        ..spec(batch, hkv * n_rep, n_rep, lq, lk, d)
    };
    let split = kg::AttentionSpec {
        schedule: kg::AttentionSchedule::SplitKv {
            lanes: 64,
            splits: 3,
            min_units: 1,
            walk,
        },
        ..streaming.clone()
    };
    assert_eq!(streaming.workgroups(), batch * hkv * lq);
    assert_eq!(split.workgroups(), batch * hkv * lq * 3);
    for spec in [streaming, split] {
        let operands = Operands::new(&spec, vec![0.0; batch * lq * lk]);
        let mut buffers = operands.buffers(&spec);
        let partial = spec.partial_elems();
        let scratch = Buffer::from_f32s(&vec![f32::NAN; partial.unwrap_or(spec.rows() * spec.dv)]);
        buffers.push(scratch);
        let generated = kg::generate(
            &kg::KernelRequest::Attention(spec.clone()),
            &caps(),
            &LIMITS,
        )
        .expect("the request generates");
        let grid = generated.launch.grid;
        let workgroups = [grid[0] / 64, grid[1], grid[2]];
        assert_eq!(
            workgroups.iter().product::<u32>() as usize,
            spec.workgroups(),
            "{:?}: the launch is the row groups",
            spec.schedule
        );
        run(&generated.body, workgroups, &mut buffers).unwrap();
        for (name, operand) in [("k", 1), ("v", 2)] {
            let reads = buffers[operand].read_counts();
            assert!(
                reads.iter().all(|&count| count == lq as u32),
                "{:?}: some element of {name} was not read once per query row: {:?}",
                spec.schedule,
                reads
                    .iter()
                    .enumerate()
                    .find(|(_, count)| **count != lq as u32)
            );
        }
    }
}

/// `spec` under the tiled schedule: `br` query positions per workgroup, `bc` keys per block, and the per-lane
/// `(rows, cols)` register tiles of the score and output products over `lanes` lanes.
fn tiled(
    spec: kg::AttentionSpec,
    (br, bc, lanes): (usize, usize, usize),
    score: (usize, usize),
    out: (usize, usize),
) -> kg::AttentionSpec {
    let tile = |(rows, cols)| kg::LaneTile { rows, cols };
    kg::AttentionSpec {
        schedule: kg::AttentionSchedule::Tiled {
            br,
            bc,
            lanes,
            micro: kg::AttentionMicro::Scalar {
                score: tile(score),
                out: tile(out),
            },
        },
        ..spec
    }
}

/// The tiled schedule's flash class: the reassociated online softmax of a different key order and block size.
const TILED_TOLERANCE: f32 = 1e-5;

/// The 8-lane tiling of a block of 4 positions of 2 heads (8 rows), 4 keys and 8 value channels: a 4 x 2 lane
/// grid of 2 x 2 score tiles, and a 4 x 2 grid of 2 x 4 output tiles.
fn small_tiled(spec: kg::AttentionSpec) -> kg::AttentionSpec {
    tiled(spec, (4, 4, 8), (2, 2), (2, 4))
}

/// A tiled causal prefill equals the reference and the streaming kernel: 13 query rows (the last block of 4
/// holds one real row and three padded ones), 13 keys in blocks of 4 (the diagonal blocks are masked, the
/// tail block runs past the capacity), two batch rows and GQA heads packed into the tile rows. The same with the
/// mask stated as an additive tensor, per batch row and per head.
///
/// Mutation: in `tiled_body`, skip the hidden scores of a partial block (`partial` always false); every
/// diagonal block reads keys past its rows' positions and the causal rows go red.
#[test]
fn a_tiled_prefill_matches_the_reference_and_the_streaming_kernel() {
    let (batch, hq, n_rep, l, d) = (2, 4, 2, 13, 8);
    let positions: Vec<i32> = (0..batch).flat_map(|_| 0..l as i32).collect();
    let streaming = causal(spec(batch, hq, n_rep, l, l, d), None);
    let structural = Operands::new(&streaming, Vec::new()).with_positions(&positions);
    let tiled_spec = small_tiled(streaming.clone());
    let got = interpret(&tiled_spec, &caps(), &structural);
    assert!(got.iter().all(|x| x.is_finite()), "{got:?}");
    assert_close(&got, &structural.reference(&tiled_spec), TILED_TOLERANCE);
    assert_close(
        &got,
        &interpret(&streaming, &caps(), &structural),
        TILED_TOLERANCE,
    );

    let additive = spec(batch, hq, n_rep, l, l, d);
    let by_mask = Operands::new(&additive, additive_mask(batch, l, l, None));
    let got = interpret(&small_tiled(additive.clone()), &caps(), &by_mask);
    assert_close(&got, &by_mask.reference(&additive), TILED_TOLERANCE);

    let per_head = kg::AttentionSpec {
        mask: kg::MaskKind::PerHead,
        ..additive
    };
    let mask: Vec<f32> = fill(batch * hq * l * l, 9)
        .into_iter()
        .map(|x| 4.0 * x)
        .collect();
    let operands = Operands::new(&per_head, mask);
    let got = interpret(&small_tiled(per_head.clone()), &caps(), &operands);
    assert_close(&got, &operands.reference(&per_head), TILED_TOLERANCE);
}

/// Other tilings of one causal prefill: one head per key/value head (MHA, 8 positions per block), wide key
/// blocks (16 keys, one partial block per workgroup), lane tiles of one row or one column, and a lane grid
/// whose output columns are fewer than its score columns.
#[test]
fn every_tiling_of_a_prefill_matches_the_reference() {
    let l = 21;
    let rows = [
        // (hq, n_rep, (br, bc, lanes), score, out)
        (2, 1, (8, 4, 8), (2, 2), (1, 8)),
        (4, 2, (4, 16, 8), (2, 8), (2, 4)),
        (4, 2, (4, 4, 16), (1, 2), (1, 4)),
        (8, 4, (2, 8, 4), (4, 4), (2, 8)),
    ];
    for (hq, n_rep, launch, score, out) in rows {
        let streaming = causal(spec(1, hq, n_rep, l, l, 8), None);
        let positions: Vec<i32> = (0..l as i32).collect();
        let operands = Operands::new(&streaming, Vec::new()).with_positions(&positions);
        let spec = tiled(streaming, launch, score, out);
        let got = interpret(&spec, &caps(), &operands);
        assert_close(&got, &operands.reference(&spec), TILED_TOLERANCE);
    }
}

/// A prefill into a cache under a window and a live bound: 6 queries at positions 19..25 of a 40-key cache, a
/// 7-key window and a live length of 25, so the walk starts at the block of key 13 and ends at key 25. Every key
/// no row of the block sees (before the earliest window, past the live length) is NaN, and the output is
/// finite: the walk neither reads nor stages them.
///
/// Mutation: in `key_range`, start the causal walk at key 0 (drop the window's first block); the poisoned keys
/// before the window are staged and the output goes NaN.
#[test]
fn a_tiled_prefill_into_a_cache_never_reads_a_key_no_row_sees() {
    let (lq, lk, window, live) = (6, 40, 7, 25);
    let base = kg::AttentionSpec {
        bound: kg::KeyBound::Live,
        ..causal(spec(1, 4, 2, lq, lk, 8), Some(window))
    };
    let positions: Vec<i32> = (19..19 + lq as i32).collect();
    let mut operands = Operands::new(&base, Vec::new())
        .with_positions(&positions)
        .with_len(&[live]);
    // Row 19's window starts at key 13, whose block of 4 starts at 12; keys 12 and later stay real.
    for t in (0..12).chain(live as usize..lk) {
        poison_key(&mut operands, &base, 0, t);
    }
    let spec = small_tiled(base);
    let got = interpret(&spec, &caps(), &operands);
    assert!(
        got.iter().all(|x| x.is_finite()),
        "poison reached the output: {got:?}"
    );
    assert_close(&got, &operands.reference(&spec), TILED_TOLERANCE);
}

/// A sink and a softcap under the tiled schedule; the sink heads straddle the scores as in the streaming row.
#[test]
fn a_tiled_sink_and_softcap_match_the_reference() {
    let l = 11;
    let base = kg::AttentionSpec {
        sink: kg::AttentionSink::PerHead,
        score: kg::ScoreTransform::ScaleSoftcap {
            scale_bits: 0.7f32.to_bits(),
            cap_bits: 1.5f32.to_bits(),
        },
        ..causal(spec(1, 4, 2, l, l, 8), None)
    };
    let positions: Vec<i32> = (0..l as i32).collect();
    let operands = Operands::new(&base, Vec::new())
        .with_positions(&positions)
        .with_sinks(4);
    let spec = small_tiled(base);
    let got = interpret(&spec, &caps(), &operands);
    assert_close(&got, &operands.reference(&spec), TILED_TOLERANCE);
}

/// A split key (a 4-channel part shared by every head, the value narrower than the key) and a paged key read
/// through a scattered block table, both under the tiled schedule.
#[test]
fn a_tiled_split_or_paged_key_matches_the_reference() {
    let l = 9;
    let positions: Vec<i32> = (0..l as i32).collect();
    let split = kg::AttentionSpec {
        dk: 12,
        key: kg::KeyLayout::Split { shared_dim: 4 },
        ..causal(spec(1, 4, 2, l, l, 8), None)
    };
    let operands = Operands::new(&split, Vec::new()).with_positions(&positions);
    let tiled_split = small_tiled(split);
    assert_close(
        &interpret(&tiled_split, &caps(), &operands),
        &operands.reference(&tiled_split),
        TILED_TOLERANCE,
    );

    // A paged key's block table covers whole blocks.
    let l = 12;
    let paged = kg::AttentionSpec {
        key: kg::KeyLayout::Paged { block: BLOCK },
        ..causal(spec(2, 4, 2, l, l, 8), None)
    };
    let pool_blocks = 17;
    let read = scattered_table(2, l, pool_blocks, |_, _| true);
    let positions: Vec<i32> = (0..2).flat_map(|_| 0..l as i32).collect();
    let operands = Operands::new(&paged, Vec::new())
        .with_positions(&positions)
        .with_pools(&paged, pool_blocks, read);
    let tiled_paged = small_tiled(paged);
    assert_close(
        &interpret(&tiled_paged, &caps(), &operands),
        &operands.reference(&tiled_paged),
        TILED_TOLERANCE,
    );
}

/// A tiled prefill over a two-byte KV cache (f16, the default cache dtype, and bf16) widens every staged key,
/// shared key part and value element on load: it equals the reference over the stored values and the streaming
/// kernel over the same lane.
///
/// Mutation: in `tiled_body`, declare the key and value parameters f32 whatever the lane; the row goes red.
#[test]
fn a_tiled_prefill_reads_its_keys_and_values_through_a_two_byte_lane() {
    let l = 13;
    let positions: Vec<i32> = (0..2).flat_map(|_| 0..l as i32).collect();
    for kv in [kg::KvLane::F16, kg::KvLane::Bf16] {
        let dense = kg::AttentionSpec {
            kv,
            ..causal(spec(2, 4, 2, l, l, 8), None)
        };
        let split = walked(kg::AttentionSpec {
            dk: 12,
            key: kg::KeyLayout::Split { shared_dim: 4 },
            ..dense.clone()
        });
        for streaming in [dense, split] {
            let operands = Operands::new(&streaming, Vec::new())
                .with_positions(&positions)
                .stored(kv);
            let tiled_spec = small_tiled(streaming.clone());
            let got = interpret(&tiled_spec, &caps(), &operands);
            assert_close(&got, &operands.reference(&tiled_spec), TILED_TOLERANCE);
            assert_close(
                &got,
                &interpret(&streaming, &caps(), &operands),
                TILED_TOLERANCE,
            );
        }
    }
}

/// A tiled prefill split into runs of its row groups (blocks of positions) writes every row once; a live length
/// of zero leaves every row with no key, the reference's NaN; a padded query at position -1 sees key 0.
#[test]
fn tiled_runs_empty_rows_and_padded_queries_match_the_reference() {
    let l = 13;
    let base = causal(spec(2, 4, 2, l, l, 8), None);
    let positions: Vec<i32> = (0..2).flat_map(|_| -1..l as i32 - 1).collect();
    let operands = Operands::new(&base, Vec::new()).with_positions(&positions);
    let spec = small_tiled(base.clone());
    let groups = spec.row_groups();
    assert_eq!(
        groups,
        2 * 2 * 4,
        "batch x kv heads x blocks of 4 positions"
    );
    let runs: Vec<_> = (0..groups.div_ceil(3))
        .map(|run| {
            kg::KernelRequest::AttentionRows(kg::AttentionRows {
                spec: spec.clone(),
                first: run * 3,
                count: 3.min(groups - run * 3),
            })
        })
        .collect();
    let got = interpret_requests(&spec, &runs, &caps(), &operands);
    assert!(got.iter().all(|x| x.is_finite()), "{got:?}");
    assert_close(&got, &operands.reference(&spec), TILED_TOLERANCE);

    let empty = small_tiled(kg::AttentionSpec {
        bound: kg::KeyBound::Live,
        ..base
    });
    let operands = Operands::new(&empty, Vec::new())
        .with_positions(&positions)
        .with_len(&[0, 0]);
    assert!(
        interpret(&empty, &caps(), &operands)
            .iter()
            .all(|x| x.is_nan())
    );
}

/// A tiling whose lane tiles do not cover the block with the workgroup's lanes, or a zero block, is a typed
/// error naming the generator.
#[test]
fn a_tiling_that_does_not_fit_the_block_is_a_typed_error() {
    let base = causal(spec(1, 4, 2, 8, 8, 8), None);
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    for spec in [
        tiled(base.clone(), (4, 4, 16), (2, 2), (2, 4)),
        tiled(base.clone(), (4, 4, 8), (3, 2), (2, 4)),
        tiled(base.clone(), (4, 0, 8), (2, 2), (2, 4)),
    ] {
        let request = kg::KernelRequest::Attention(spec.clone());
        assert!(
            matches!(
                kg::generate(&request, &caps(), &limits),
                Err(kg::KernelGenError::NotDivisible { .. }
                    | kg::KernelGenError::BelowMinimum { .. })
            ),
            "{:?} is refused",
            spec.schedule
        );
    }
}

/// The streaming and split-KV bodies multiply exact operands: a request that names another precision is a typed
/// refusal, not an unrounded kernel (POOT-1493; the planner refuses the same, `NoMatrixUnitSchedule`). Mutation:
/// drop the `require_exact` call in `attention_body`; the narrowed streaming and split requests generate and this
/// row goes red.
#[test]
fn a_streaming_or_split_attention_refuses_a_narrowed_precision() {
    use poot_graph_ir::{FragmentDtype, OperandPrecision};
    let streaming = with_lanes(spec(1, 4, 2, 1, 13, 8), 64);
    let split = with_splits(with_lanes(spec(2, 3, 1, 1, 41, 8), LANES), LANES, 4);
    let limits = kg::BodyLimits {
        max_instructions: std::num::NonZeroUsize::new(1 << 22).unwrap(),
        max_locals: std::num::NonZeroUsize::new(1 << 18).unwrap(),
    };
    for spec in [streaming, split] {
        for precision in [
            OperandPrecision::Narrow(FragmentDtype::F16),
            OperandPrecision::Split(FragmentDtype::Bf16),
        ] {
            let narrowed = kg::AttentionSpec {
                precision,
                ..spec.clone()
            };
            let error = kg::generate(&kg::KernelRequest::Attention(narrowed), &caps(), &limits)
                .unwrap_err();
            assert!(
                matches!(
                    error,
                    kg::KernelGenError::Unsupported {
                        request: "attention_body",
                        ..
                    }
                ),
                "{:?} {precision}: {error}",
                spec.schedule
            );
        }
    }
}

/// POOT-1430: a narrowed attention rounds Q, K, V and the probabilities. Only the tiled schedule narrows (the
/// streaming and split-KV bodies multiply exact operands and refuse any other precision, POOT-1493), so under each
/// narrow dtype it lands within the dtype's bound of the textbook attention over operands rounded as declared (what
/// is left is the probabilities' own rounding, half an ulp of the dtype times the value range), and off the exact
/// kernel's output by more than that rounding alone could hide, so the rounding is applied and the row is not
/// inert. Mutation: in `operand_terms` return the value unrounded; the tiled output equals the exact kernel's and
/// this row goes red ("only 0e0 from the exact kernel, the rounding is not applied"). One operand's omission alone
/// stays inside the bound, so the staged-value rows of `operand_precision_interp` pin each operand's rounding.
#[test]
fn a_narrowed_attention_rounds_its_operands() {
    use poot_graph_ir::{FragmentDtype, OperandPrecision};
    let rounded = |dtype, values: &[f32]| -> Vec<f32> {
        values
            .iter()
            .map(|&x| poot_eval::round_to_fragment(dtype, x))
            .collect()
    };
    let distance = |a: &[f32], b: &[f32]| max_abs_error(a, b);

    let prefill = causal(spec(2, 4, 2, 13, 13, 8), None);
    let positions: Vec<i32> = (0..2).flat_map(|_| 0..13).collect();

    struct Case {
        name: &'static str,
        spec: kg::AttentionSpec,
        operands: Operands,
    }
    let cases = [Case {
        name: "tiled prefill",
        operands: Operands::new(&prefill, Vec::new()).with_positions(&positions),
        spec: small_tiled(prefill),
    }];
    let run = |case: &Case, spec: &kg::AttentionSpec| -> Vec<f32> {
        interpret(spec, &caps(), &case.operands)
    };
    for case in &cases {
        let exact = run(case, &case.spec);
        for (dtype, bound, visible) in [
            (FragmentDtype::F16, 3e-3, 5e-5),
            (FragmentDtype::Bf16, 2e-2, 5e-4),
        ] {
            let narrow = kg::AttentionSpec {
                precision: OperandPrecision::Narrow(dtype),
                ..case.spec.clone()
            };
            let got = run(case, &narrow);
            let mut operands = Operands::new(&case.spec, case.operands.mask.clone());
            operands.positions = case.operands.positions.clone();
            operands.len = case.operands.len.clone();
            operands.q = rounded(dtype, &operands.q);
            operands.k = rounded(dtype, &operands.k);
            operands.v = rounded(dtype, &operands.v);
            let want = operands.reference(&case.spec);
            let near = distance(&got, &want);
            let moved = distance(&got, &exact);
            assert!(
                near < bound,
                "{} {dtype:?}: {near:e} from the rounded reference",
                case.name
            );
            assert!(
                moved > visible,
                "{} {dtype:?}: only {moved:e} from the exact kernel, the rounding is not applied",
                case.name
            );
        }
        // A split operand recovers the f32 values: the streaming schedules land far nearer the exact output
        // than the narrow ones. (The tiled schedule stages one tile per operand and takes no split precision.)
        if !matches!(case.spec.schedule, kg::AttentionSchedule::Tiled { .. }) {
            let narrow = kg::AttentionSpec {
                precision: OperandPrecision::Narrow(FragmentDtype::F16),
                ..case.spec.clone()
            };
            let split = kg::AttentionSpec {
                precision: OperandPrecision::Split(FragmentDtype::F16),
                ..case.spec.clone()
            };
            let (narrow_err, split_err) = (
                distance(&run(case, &narrow), &exact),
                distance(&run(case, &split), &exact),
            );
            assert!(
                split_err < narrow_err / 10.0,
                "{}: split {split_err:e} vs narrow {narrow_err:e}",
                case.name
            );
        }
    }
}

/// The tiled schedule stages one tile per operand, so a split precision on it is a typed refusal by name, not a
/// body that drops the `lo` part.
#[test]
fn a_split_precision_has_no_tiled_schedule() {
    use poot_graph_ir::{FragmentDtype, OperandPrecision};
    let base = causal(spec(1, 4, 2, 13, 13, 8), None);
    let tiled = kg::AttentionSpec {
        precision: OperandPrecision::Split(FragmentDtype::F16),
        ..small_tiled(base)
    };
    let error = kg::generate(&kg::KernelRequest::Attention(tiled), &caps(), &LIMITS)
        .expect_err("a split tiled attention has no body");
    assert!(error.to_string().contains("split"), "{error}");
}
