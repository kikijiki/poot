//! The kernel-choice record: which kernel the planner chose for one equation, kept beside its [`Plan`].
//!
//! The plan alone does not say whether a body was generated from a request, imported from a shipped
//! template, or never dispatched at all. [`KernelChoice`] does, and [`Planned`] is the one constructor
//! that builds a [`Plan`] and its choice together, so a plan's key, body and launch are always the ones
//! its choice names. The executor never reads the choice.

use std::fmt;

use poot_kernelgen::{KernelRequest, generate};
use poot_target::{Backend, DeviceCaps};

use crate::*;

/// An equation that plans no kernel launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoDispatch {
    /// The output aliases an input buffer (reshape, an identity cast).
    Alias,
    /// The output is a strided view of an input buffer.
    View,
}

/// The kernel the planner chose for one equation.
// One record per planned equation, matched by value in many places: boxing the request would put a
// `Box` pattern wall between every caller and the request it reads.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug)]
pub enum KernelChoice {
    /// A body `poot-kernelgen` generated from this request.
    Generated(KernelRequest),
    /// A shipped template body, bound with a metadata buffer of `meta_schema` words (zero for none). The
    /// count is read from the metadata the plan carries, never typed in a second place. `index` is the
    /// element type of the runtime index operand the equation binds, for a template that reads one: the
    /// body reads an F32 and an I32 index identically, but they are different bindings and must never
    /// share a plan key (R469-005).
    Imported {
        kernel: ImportedKernel,
        meta_schema: u32,
        index: Option<DType>,
    },
    /// No launch at all.
    NoDispatch(NoDispatch),
}

impl KernelChoice {
    /// A short, stable name for diagnostics and the plan summary: `generated:<family>`,
    /// `imported:<kernel>`, or `nodispatch:<kind>`. It
    /// names the kind of choice, not its parameters; the plan key carries the parameters.
    pub fn label(&self) -> String {
        match self {
            Self::Generated(request) => format!("generated:{}", request.family()),
            Self::Imported { kernel, .. } => format!("imported:{kernel:?}").to_lowercase(),
            Self::NoDispatch(kind) => format!("nodispatch:{kind:?}").to_lowercase(),
        }
    }

    /// The kernel's display label: a generated request names itself
    /// ([`KernelRequest::display_label`]), a shipped template its kernel. Carried by the plan beside its key.
    fn display_label(&self) -> String {
        match self {
            Self::Generated(request) => request.display_label(),
            other => other.label(),
        }
    }

    /// The digest of this choice's canonical encoding for `backend`'s codegen epoch.
    fn digest(&self, backend: Backend) -> ChoiceDigest {
        // The `Debug` form of the plain-data choice is its encoding. That is sound only while no type
        // reachable from a request has a lossy `Debug`: the audit (card 636) found one hand-written
        // impl, `TileSize`, which prints its `u32` edge, and `f32` fields (`FusedInput::Lit`), whose `Debug`
        // is the shortest round-trip text and so distinguishes every value but the NaN payloads. The
        // `every_request_family_has_a_field_that_changes_the_digest` test pins one field per family (and the
        // `TileSize` edge and the Gemv launch shape of the packed family).
        let mut hasher = blake3::Hasher::new();
        hasher.update(poot_codegen::kernel_cache_epoch(codegen_target(backend)).as_bytes());
        hasher.update(b"\n");
        hasher.update(format!("{self:?}").as_bytes());
        ChoiceDigest(*hasher.finalize().as_bytes())
    }
}

/// The blake3 digest of a [`KernelChoice`]'s canonical encoding plus the codegen epoch. It is a plan's
/// `key`: two choices that differ in any parameter get different keys, and the key cannot be spelled
/// anywhere but here.
#[derive(Clone, Copy, PartialEq, Eq)]
struct ChoiceDigest([u8; 32]);

impl fmt::Display for ChoiceDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.iter().try_for_each(|byte| write!(f, "{byte:02x}"))
    }
}

/// One equation's [`Plan`] with the [`KernelChoice`] that produced it.
pub struct Planned {
    pub plan: Plan,
    pub choice: KernelChoice,
}

/// One dispatch of a split plan, with the choice that built it.
pub(crate) struct PlannedDispatch {
    dispatch: Dispatch,
    choice: KernelChoice,
}

/// A generation request, run for `backend`: the body, its launch and the choice that names them.
struct Built {
    body: Body,
    launch: poot_kernelgen::Launch,
    choice: KernelChoice,
    key: String,
    label: String,
}

fn build(
    site: Site<'_>,
    request: KernelRequest,
    backend: Backend,
    caps: &DeviceCaps,
) -> Result<Built, PlanError> {
    let generated = generate(&request, caps, site.limits())
        .map_err(|error| site.refuse(Capability::KernelGen(error)))?;
    let choice = KernelChoice::Generated(request);
    let key = choice.digest(backend).to_string();
    let label = choice.display_label();
    Ok(Built {
        body: generated.body,
        launch: generated.launch,
        choice,
        key,
        label,
    })
}

impl Planned {
    /// Generate `request` and plan the dispatch it describes.
    pub(crate) fn generated(
        site: Site<'_>,
        request: KernelRequest,
        backend: Backend,
        caps: &DeviceCaps,
    ) -> Result<Self, PlanError> {
        let PlannedDispatch { dispatch, choice } =
            Self::generated_dispatch(site, request, backend, caps)?;
        Ok(Self {
            plan: Plan::Dispatches(vec![dispatch]),
            choice,
        })
    }

    /// Generate `request` as one dispatch of a split plan.
    pub(crate) fn generated_dispatch(
        site: Site<'_>,
        request: KernelRequest,
        backend: Backend,
        caps: &DeviceCaps,
    ) -> Result<PlannedDispatch, PlanError> {
        let Built {
            body,
            launch,
            choice,
            key,
            label,
        } = build(site, request, backend, caps)?;
        Ok(PlannedDispatch {
            dispatch: Dispatch {
                body,
                key,
                label,
                launch: Launch {
                    grid: launch.grid,
                    meta: launch.meta.unwrap_or_default(),
                },
            },
            choice,
        })
    }

    /// Plan a shipped template. `meta` is its metadata buffer (`None` for a kernel that takes none).
    pub(crate) fn imported(
        kernel: ImportedKernel,
        backend: Backend,
        meta: Option<Vec<u32>>,
        grid: [u32; 3],
    ) -> Self {
        Self::imported_indexed(kernel, backend, meta, grid, None)
    }

    /// [`Planned::imported`] for a template that reads a runtime index operand of element type `index`.
    pub(crate) fn imported_indexed(
        kernel: ImportedKernel,
        backend: Backend,
        meta: Option<Vec<u32>>,
        grid: [u32; 3],
        index: Option<DType>,
    ) -> Self {
        let PlannedDispatch { dispatch, choice } =
            Self::imported_dispatch(kernel, backend, meta, grid, index);
        Self {
            plan: Plan::Dispatches(vec![dispatch]),
            choice,
        }
    }

    /// One dispatch of a split plan that runs a shipped template; `meta` is its metadata, and the grid is
    /// `groups` workgroups of the template's own workgroup size.
    pub(crate) fn imported_split_dispatch(
        kernel: ImportedKernel,
        backend: Backend,
        meta: Vec<u32>,
        groups: usize,
    ) -> PlannedDispatch {
        let [x, y, z] = kernel.body().workgroup_size;
        let grid = [groups as u32 * x, y, z];
        Self::imported_dispatch(kernel, backend, Some(meta), grid, None)
    }

    fn imported_dispatch(
        kernel: ImportedKernel,
        backend: Backend,
        meta: Option<Vec<u32>>,
        grid: [u32; 3],
        index: Option<DType>,
    ) -> PlannedDispatch {
        let choice = imported_choice(kernel, meta.as_deref(), index);
        let key = choice.digest(backend).to_string();
        let label = choice.display_label();
        PlannedDispatch {
            dispatch: Dispatch {
                body: kernel.body().clone(),
                key,
                label,
                launch: Launch {
                    grid,
                    meta: meta.unwrap_or_default(),
                },
            },
            choice,
        }
    }

    /// Several dispatches over disjoint ranges of one output. The equation's choice is its first
    /// dispatch's: the dispatches are one kernel family and differ only in the range they cover. An empty
    /// split (a zero-element output) has no choice to record and is refused.
    pub(crate) fn split(dispatches: Vec<PlannedDispatch>) -> Result<Self, PlanError> {
        let Some(first) = dispatches.first() else {
            return Err(PlanError::BadShape(
                "a split plan needs at least one dispatch (zero-element output)".into(),
            ));
        };
        let choice = first.choice.clone();
        Ok(Self {
            plan: Plan::Dispatches(
                dispatches
                    .into_iter()
                    .map(|PlannedDispatch { dispatch, .. }| dispatch)
                    .collect(),
            ),
            choice,
        })
    }

    pub(crate) fn alias(src: ValueId) -> Self {
        Self::no_dispatch(Plan::Alias(src), NoDispatch::Alias)
    }

    pub(crate) fn view(src: ValueId, layout: Layout) -> Self {
        let Layout { strides, offset } = layout;
        Self::no_dispatch(
            Plan::View {
                src,
                strides,
                offset,
            },
            NoDispatch::View,
        )
    }

    fn no_dispatch(plan: Plan, kind: NoDispatch) -> Self {
        Self {
            plan,
            choice: KernelChoice::NoDispatch(kind),
        }
    }
}

fn imported_choice(
    kernel: ImportedKernel,
    meta: Option<&[u32]>,
    index: Option<DType>,
) -> KernelChoice {
    KernelChoice::Imported {
        kernel,
        meta_schema: meta.map_or(0, |meta| meta.len() as u32),
        index,
    }
}

#[cfg(test)]
mod tests {
    use poot_graph_ir::AttentionAttrs;
    use poot_graph_ir::{Builder, TensorType};
    use poot_kernelgen::{
        AttentionSchedule, AttentionSpec, KeyBound, KeyLayout, MaskKind, ScoreTransform,
    };
    use poot_tensor::DType;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;
    use crate::{CompileError, CompileOptions, Program, Submission, Target, compile};

    /// A single-sequence decode attention (`d` head dim, `scale` attention scale), traced as the
    /// primitive chain `ops::attention_masked` builds; `compile` forms the flash op from it.
    fn flash_decode_graph(d: usize, scale: f32) -> Graph {
        let (hq, cap) = (2usize, 8usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::new(vec![1, hq, 1, d], DType::F32));
        let k = b.constant("k", TensorType::new(vec![1, hq, cap, d], DType::F32));
        let v = b.constant("v", TensorType::new(vec![1, hq, cap, d], DType::F32));
        let mask = b.constant("mask", TensorType::new(vec![1, 1, 1, cap], DType::F32));
        let out = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, scale, mask);
        b.finish(out)
    }

    const OPTIONS: CompileOptions =
        CompileOptions::new(Submission::Replay, crate::CompileLimits::STANDARD);

    fn target(backend: Backend) -> Target {
        Target {
            backend,
            caps: default_caps_for(backend),
        }
    }

    /// The compiled flash decode equation's plan and choice.
    fn compiled_flash_decode(program: &Program) -> (&Plan, &KernelChoice) {
        let (eqn, plan) = program
            .planned()
            .find(|(eqn, _)| matches!(eqn.op, OpKind::Attention(_)))
            .expect("compile forms the flash decode");
        (plan, program.kernel_choice(eqn))
    }

    fn key_of(plan: &Plan) -> &str {
        match plan {
            Plan::Dispatches(dispatches) if dispatches.len() == 1 => &dispatches[0].key,
            other => panic!("expected a single dispatch, got {other:?}"),
        }
    }

    /// Card 636 SC-003 (dkernel B2): two requests that differ only in one f32 parameter get different
    /// digests, hence different plan keys. The generated attention request takes `scale` as a request parameter stored as its bits.
    /// Mutation: store `scale` as `*scale as u32` (the integer value; both 1/8 and 1/16 truncate to 0):
    /// the requests, digests and keys collide and this row goes red.
    #[test]
    fn requests_differing_only_in_an_f32_parameter_get_different_keys() {
        let compiled = |scale: f32| {
            compile(
                &flash_decode_graph(64, scale),
                &target(Backend::SpirvVulkan),
                &OPTIONS,
            )
            .expect("the decode compiles")
        };
        let (eighth, sixteenth) = (compiled(1.0 / 8.0), compiled(1.0 / 16.0));
        let (eighth, sixteenth) = (
            compiled_flash_decode(&eighth),
            compiled_flash_decode(&sixteenth),
        );
        // The key layer first: this is the assertion the named mutation must turn red.
        assert_ne!(
            key_of(eighth.0),
            key_of(sixteenth.0),
            "flash scale 1/8 and 1/16 must not share a plan key"
        );
        for ((_, choice), scale) in [(eighth, 1.0f32 / 8.0), (sixteenth, 1.0 / 16.0)] {
            let KernelChoice::Generated(KernelRequest::Attention(spec)) = choice else {
                panic!("expected a generated attention request: {choice:?}");
            };
            assert_eq!(
                spec.score,
                ScoreTransform::Scale {
                    scale_bits: scale.to_bits()
                }
            );
        }
    }

    /// A graph holding one `Attention` decode with head dim `d`, staged directly.
    fn flash_decode_composite(d: usize) -> Graph {
        let (hq, cap) = (2usize, 8usize);
        let b = Builder::new();
        let q = b.constant("q", TensorType::new(vec![1, hq, 1, d], DType::F32));
        let k = b.constant("k", TensorType::new(vec![1, hq, cap, d], DType::F32));
        let v = b.constant("v", TensorType::new(vec![1, hq, cap, d], DType::F32));
        let mask = b.constant("mask", TensorType::new(vec![1, 1, 1, cap], DType::F32));
        let mut plan = b.append_plan(0);
        let out = plan
            .equation(
                OpKind::Attention(AttentionAttrs::new(1, 0.5, None)),
                [q, k, v, mask]
                    .map(|operand| poot_graph_ir::Operand::Value(operand.id()))
                    .to_vec(),
            )
            .expect("stage the flash decode");
        plan.declare_result(out).expect("declare the result");
        let mut prepared = b.preflight_append(plan).expect("preflight");
        let committed = b.commit_append(&mut prepared).expect("commit");
        b.finish(committed)
    }

    /// Card 636 SC-004: a request the device cannot hold is a typed refusal out of `compile`, not a panic.
    /// A device whose workgroup memory cannot hold one attention lane at head dim 512 (2056 bytes) is
    /// refused with `AttentionLds` naming the head dim and the device's bytes; the same graph compiles on
    /// the real caps. Mutation: make the planner panic where it refuses; this row panics.
    #[test]
    fn a_device_too_small_for_the_request_compiles_to_a_typed_refusal() {
        let g = flash_decode_composite(512);
        let small = Target {
            backend: Backend::SpirvVulkan,
            caps: poot_target::DeviceCaps {
                lds_bytes: 2048,
                ..default_caps_for(Backend::SpirvVulkan)
            },
        };
        match compile(&g, &small, &OPTIONS) {
            Err(CompileError::Plan(error)) => match *error {
                PlanError::Refused(refusal) => assert_eq!(
                    refusal.missing,
                    Capability::AttentionLds {
                        dv: 512,
                        lds_bytes: 2048,
                    }
                ),
                other => panic!("expected a typed refusal, got {other:?}"),
            },
            other => panic!("expected a refusal, got {:?}", other.map(|_| ())),
        }
        let real = compile(&g, &target(Backend::SpirvVulkan), &OPTIONS)
            .expect("the real caps hold the same request");
        assert!(matches!(
            compiled_flash_decode(&real).1,
            KernelChoice::Generated(KernelRequest::Attention(_))
        ));
    }

    fn digest_of(request: KernelRequest) -> String {
        KernelChoice::Generated(request)
            .digest(Backend::SpirvVulkan)
            .to_string()
    }

    /// Card 636: the digest hashes the `Debug` text of the request, so every field must be visible
    /// in it. This is per family, not per field: one field of each family (the `f32` literal a fused region
    /// carries, and for the packed family the hand-written `TileSize` `Debug` and the Gemv launch shape)
    /// changes the digest. Mutation: make `TileSize`'s `Debug` write "0" (`packed/kernel.rs`); the tiled
    /// packed pair goes red.
    #[test]
    fn every_request_family_has_a_field_that_changes_the_digest() {
        use poot_kernel_ir::{BinOp, Ty};
        use poot_kernelgen::{
            ContractionSpec, ElementwiseSpec, Fold, FusedInput, FusedKernel, FusedScalarOp,
            FusedStep, LeafTy, MatmulShapes, MovementOp, MovementSpec, PackedContraction,
            PackedKernelOp, PackedKernelSpec, PackedRequest, PointwiseForm, PointwiseSpec,
            RopeSpec, RowKernel, RowOp, RowSpec, Schedule, Storage, TileSize, TopKSpec, UnaryOp,
            ViewOperand,
        };
        use poot_quant::format::WeightFormat;

        let fold = |numel| Fold {
            numel,
            two_d: false,
            width: 256,
        };
        let fused = |literal: f32| FusedKernel {
            leaf_tys: vec![LeafTy::F32; 1],
            steps: vec![FusedStep {
                op: FusedScalarOp::Binary(BinOp::Add),
                inputs: vec![FusedInput::Leaf(0), FusedInput::Lit(literal)],
            }],
            output: FusedInput::Step(0),
        };
        let leaf = || ViewOperand {
            shape: vec![4],
            layout: poot_kernelgen::Layout::contiguous(&[4]),
        };
        let pointwise = |literal: f32| {
            KernelRequest::Pointwise(PointwiseSpec {
                out_shape: vec![4],
                numel: 4,
                leaves: vec![leaf()],
                kernel: fused(literal),
                form: PointwiseForm::Float { fold: fold(4) },
            })
        };
        let row = |width: usize| {
            KernelRequest::Row(RowSpec::Fused {
                out_shape: vec![4],
                numel: 4,
                leaves: vec![leaf()],
                kernel: RowKernel {
                    n_leaves: 1,
                    leaf_storage: vec![poot_target::BufferStorage::f32()],
                    n_cols: 4,
                    steps: vec![RowOp::Pointwise {
                        op: FusedScalarOp::Recip,
                        inputs: vec![FusedInput::Leaf(0)],
                    }],
                    output: 1,
                },
                width,
            })
        };
        let serial = |b_cols: usize| {
            KernelRequest::Contraction(ContractionSpec::Serial {
                dt: Ty::F32,
                epilogue: None,
                shapes: MatmulShapes {
                    out_shape: vec![2, 3],
                    a_shape: vec![2, 4],
                    b_shape: vec![4, b_cols],
                },
                fold: fold(6),
            })
        };
        let attention = |scale: f32| {
            KernelRequest::Attention(AttentionSpec {
                batch: 1,
                hq: 2,
                n_rep: 1,
                lq: 1,
                lk: 8,
                dk: 4,
                dv: 4,
                key: KeyLayout::Single,
                score: ScoreTransform::Scale {
                    scale_bits: scale.to_bits(),
                },
                mask: MaskKind::Broadcast,
                bound: KeyBound::Static,
                schedule: AttentionSchedule::Streaming { lanes: 4 },
            })
        };
        let transpose = |perm: Vec<usize>| {
            KernelRequest::Movement(MovementSpec::new(
                Storage::Elements { carrier: Ty::F32 },
                vec![2, 3],
                MovementOp::Transpose {
                    in_shape: vec![3, 2],
                    perm,
                },
            ))
        };
        let packed = |grid_x: u32| {
            KernelRequest::Packed(PackedRequest {
                name: "packed_row_gather",
                spec: PackedKernelSpec {
                    format: WeightFormat::Q8_0,
                    op: PackedKernelOp::RowGather,
                },
                grid: [grid_x, 1, 1],
                meta: vec![32],
            })
        };
        let packed_contraction = |schedule: Schedule| {
            KernelRequest::Contraction(ContractionSpec::Packed(PackedContraction {
                format: WeightFormat::Q8_0,
                rows: poot_kernelgen::PackedRows::Blocks,
                schedule,
                epilogue: None,
                grid: [64, 1, 1],
                meta: vec![32, 4],
            }))
        };
        let tiled = |edge: u32| Schedule::Tiled {
            tile: TileSize::new(edge).expect("a valid tile edge"),
        };
        let gemv = |cols: u32| Schedule::Gemv {
            width: 256,
            cols,
            unroll: 8,
        };
        let unary = |numel: usize| {
            KernelRequest::Elementwise(ElementwiseSpec::Unary {
                op: UnaryOp::Basic(poot_kernel_ir::UnOp::Neg),
                dt: Ty::F32,
                view: None,
                fold: fold(numel),
            })
        };
        let rope = |rot: usize| {
            KernelRequest::Rope(RopeSpec {
                dt: Ty::F32,
                x_shape: vec![1, 4],
                cos_shape: vec![1, 2],
                rot,
                numel: 4,
            })
        };
        let topk = |k: usize| KernelRequest::TopK(TopKSpec { e: 8, k, numel: 2 });

        let pairs = [
            ("elementwise", unary(8), unary(9)),
            ("pointwise f32 literal", pointwise(1.0), pointwise(2.0)),
            (
                "pointwise f32 sign of zero",
                pointwise(0.0),
                pointwise(-0.0),
            ),
            ("row", row(4), row(8)),
            ("contraction", serial(3), serial(5)),
            ("attention", attention(0.5), attention(0.25)),
            ("movement", transpose(vec![1, 0]), transpose(vec![0, 1])),
            ("packed", packed(8), packed(16)),
            ("rope", rope(2), rope(4)),
            ("topk", topk(2), topk(3)),
            (
                "packed tiled tile edge",
                packed_contraction(tiled(8)),
                packed_contraction(tiled(16)),
            ),
            (
                "packed gemv cols",
                packed_contraction(gemv(16)),
                packed_contraction(gemv(8)),
            ),
        ];
        for (family, base, changed) in pairs {
            assert_ne!(
                digest_of(base.clone()),
                digest_of(changed),
                "{family}: a changed field must change the digest"
            );
            assert_eq!(digest_of(base.clone()), digest_of(base), "{family}: stable");
        }
    }

    /// A split plan with no dispatch has no choice to record. `Planned::split` refuses it as a typed
    /// error, and the production splitters reach it with a zero-element output (no chunks to plan).
    /// Mutation: have `split` return `Ok` with a default choice for an empty list; both rows go red.
    #[test]
    fn an_empty_split_is_a_typed_refusal() {
        assert!(matches!(
            Planned::split(Vec::new()),
            Err(PlanError::BadShape(_))
        ));

        let g = flash_decode_graph(8, 1.0);
        let eqn = g.eqns.last().expect("the graph has an equation");
        let backend = Backend::SpirvVulkan;
        let site = Site::new(
            (&g).into(),
            eqn,
            backend,
            poot_test_util::graph_fixtures::roomy_body_limits(),
        );
        let planned = crate::gemv_chunks(
            site,
            backend,
            0,
            |_, _| unreachable!("a zero-element output plans no chunk"),
            &default_caps_for(backend),
        );
        assert!(matches!(planned, Err(PlanError::BadShape(_))));
    }
}
