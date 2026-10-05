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
    /// The output aliases an input buffer (reshape, an identity cast, a collective at world size 1).
    Alias,
    /// The output is a strided view of an input buffer.
    View,
    /// A cross-rank collective the multi-rank executor runs.
    Collective,
}

/// The kernel the planner chose for one equation.
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
    /// One equation planned as several dispatches over disjoint ranges of its output (a watchdog or
    /// miscompile-ceiling split), each with its own choice. Deleted with `Plan::ComputeChunks`.
    Chunked(Vec<KernelChoice>),
}

impl KernelChoice {
    /// A short, stable name for diagnostics and the plan summary: `generated:<family>`,
    /// `imported:<kernel>`, `nodispatch:<kind>` or `chunked:<count>x<choice of the first chunk>`. It
    /// names the kind of choice, not its parameters; the plan key carries the parameters.
    pub fn label(&self) -> String {
        match self {
            Self::Generated(request) => format!("generated:{}", request.family()),
            Self::Imported { kernel, .. } => format!("imported:{kernel:?}").to_lowercase(),
            Self::NoDispatch(kind) => format!("nodispatch:{kind:?}").to_lowercase(),
            Self::Chunked(chunks) => format!(
                "chunked:{}x{}",
                chunks.len(),
                chunks.first().map(Self::label).unwrap_or_default()
            ),
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

/// One dispatch of a [`KernelChoice::Chunked`] plan, with its choice.
pub(crate) struct PlannedChunk {
    chunk: ComputeChunk,
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
    let generated = generate(&request, backend, caps, site.limits())
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
    /// Generate `request` and plan the dispatch it describes: a plain `Plan::Compute`, or a
    /// `Plan::ComputeMeta` when the launch carries a metadata buffer.
    pub(crate) fn generated(
        site: Site<'_>,
        request: KernelRequest,
        backend: Backend,
        caps: &DeviceCaps,
    ) -> Result<Self, PlanError> {
        let Built {
            body,
            launch,
            choice,
            key,
            label,
        } = build(site, request, backend, caps)?;
        let plan = match launch.meta {
            None => Plan::Compute {
                body,
                key,
                label,
                grid: launch.grid,
            },
            Some(meta) => Plan::ComputeMeta {
                body,
                key,
                label,
                meta,
                grid: launch.grid,
            },
        };
        Ok(Self { plan, choice })
    }

    /// Generate `request` as one range of a chunked plan. The chunk's workgroup count is its launch grid
    /// over the body's workgroup size: the request builds its grid as `groups * workgroup`.
    pub(crate) fn generated_chunk(
        site: Site<'_>,
        request: KernelRequest,
        backend: Backend,
        caps: &DeviceCaps,
    ) -> Result<PlannedChunk, PlanError> {
        let Built {
            body,
            launch,
            choice,
            key,
            label,
        } = build(site, request, backend, caps)?;
        let groups = launch.grid[0] as usize / body.workgroup_size[0].max(1) as usize;
        Ok(PlannedChunk {
            chunk: ComputeChunk {
                body,
                key,
                label,
                groups,
                meta: launch.meta.unwrap_or_default(),
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
        let choice = imported_choice(kernel, meta.as_deref(), index);
        let body = kernel.body().clone();
        let key = choice.digest(backend).to_string();
        let label = choice.display_label();
        let plan = match meta {
            None => Plan::Compute {
                body,
                key,
                label,
                grid,
            },
            Some(meta) => Plan::ComputeMeta {
                body,
                key,
                label,
                meta,
                grid,
            },
        };
        Self { plan, choice }
    }

    /// One range of a chunked plan that runs a shipped template; `meta` is the chunk's metadata.
    pub(crate) fn imported_chunk(
        kernel: ImportedKernel,
        backend: Backend,
        meta: Vec<u32>,
        groups: usize,
    ) -> PlannedChunk {
        let choice = imported_choice(kernel, Some(&meta), None);
        PlannedChunk {
            chunk: ComputeChunk {
                body: kernel.body().clone(),
                key: choice.digest(backend).to_string(),
                label: choice.display_label(),
                groups,
                meta,
            },
            choice,
        }
    }

    /// Several dispatches over disjoint ranges of one output.
    pub(crate) fn chunked(chunks: Vec<PlannedChunk>) -> Self {
        let (chunks, choices) = chunks
            .into_iter()
            .map(|PlannedChunk { chunk, choice }| (chunk, choice))
            .unzip();
        Self {
            plan: Plan::ComputeChunks(chunks),
            choice: KernelChoice::Chunked(choices),
        }
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

    pub(crate) fn collective(kind: CollectiveKind, op: RedOp, axis: usize) -> Self {
        Self::no_dispatch(Plan::Collective { kind, op, axis }, NoDispatch::Collective)
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
    use poot_graph_ir::{Builder, TensorType};
    use poot_kernelgen::AttentionSpec;
    use poot_tensor::DType;
    use poot_test_util::device_caps::default_caps_for;

    use super::*;
    use crate::{CompileError, CompileOptions, FusionPolicy, Program, Submission, Target, compile};

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

    const OPTIONS: CompileOptions = CompileOptions {
        execution: Submission::Replay,
        fusion: FusionPolicy::Full,
        limits: crate::CompileLimits::STANDARD,
    };

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
            .find(|(eqn, _)| matches!(eqn.op, OpKind::FlashAttentionDecode { .. }))
            .expect("compile forms the flash decode");
        (plan, program.kernel_choice(eqn))
    }

    fn key_of(plan: &Plan) -> &str {
        match plan {
            Plan::Compute { key, .. } | Plan::ComputeMeta { key, .. } => key,
            other => panic!("expected a single dispatch, got {other:?}"),
        }
    }

    /// Card 636 SC-003 (dkernel B2): two requests that differ only in one f32 parameter get different
    /// digests, hence different plan keys. The synthesized region decode, which the planner chooses for
    /// a flash decode within the LDS cap, takes `scale` as a request parameter stored as its bits.
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
            let KernelChoice::Generated(KernelRequest::Attention(AttentionSpec::RegionDecode {
                scale_bits,
                ..
            })) = choice
            else {
                panic!("expected a generated region decode: {choice:?}");
            };
            assert_eq!(*scale_bits, scale.to_bits());
        }
    }

    /// A graph holding one `FlashAttentionDecode` with head dim `d`, staged directly: the matcher
    /// `compile` runs declines a decode wider than the LDS cap (R480-009), so no traced graph reaches
    /// the planner's D>cap answer, but the planner still answers for any valid equation.
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
                OpKind::FlashAttentionDecode {
                    n_rep: 1,
                    scale: 0.5,
                },
                [q, k, v, mask]
                    .map(|operand| poot_graph_ir::Operand::Value(operand.id))
                    .to_vec(),
            )
            .expect("stage the flash decode");
        plan.declare_result(out).expect("declare the result");
        let mut prepared = b.preflight_append(plan).expect("preflight");
        let id = b.commit_append(&mut prepared).expect("commit");
        b.finish(poot_graph_ir::Traced { id })
    }

    /// Card 636 SC-004: a request the target cannot state is a typed refusal out of `compile`, not a
    /// panic. A head dim above the LDS cap needs the private-array decode, which SPIR-V cannot state;
    /// `generate` returns `Unsupported` and the planner names the missing capability. NVPTX states the
    /// same request. Mutation: make `generate` panic on that arm; this row panics.
    #[test]
    fn an_unstateable_request_compiles_to_a_typed_refusal() {
        let head_dim = poot_target::FLASH_LDS_CAP + 256;
        let g = flash_decode_composite(head_dim);
        match compile(&g, &target(Backend::SpirvVulkan), &OPTIONS) {
            Err(CompileError::Plan(error)) => match *error {
                PlanError::Refused(refusal) => assert_eq!(
                    refusal.missing,
                    Capability::HeadDimExceedsLdsCap {
                        head_dim,
                        lds_cap: poot_target::FLASH_LDS_CAP,
                    }
                ),
                other => panic!("expected a typed refusal, got {other:?}"),
            },
            other => panic!("expected a refusal on SPIR-V, got {:?}", other.map(|_| ())),
        }
        let nvptx =
            compile(&g, &target(Backend::Nvptx), &OPTIONS).expect("NVPTX states the same request");
        assert!(matches!(
            compiled_flash_decode(&nvptx).1,
            KernelChoice::Generated(KernelRequest::Attention(AttentionSpec::DecodeSingle { .. }))
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
            FusedStep, MatmulShapes, MovementSpec, PackedKernelOp, PackedKernelSpec, PackedRequest,
            PointwiseForm, PointwiseSpec, RopeSpec, RowKernel, RowOp, RowSelect, RowSpec, Schedule,
            TileSize, TopKSpec, UnaryOp, ViewOperand,
        };
        use poot_quant::format::WeightFormat;

        let fold = |numel| Fold {
            numel,
            two_d: false,
            width: 256,
        };
        let fused = |literal: f32| FusedKernel {
            n_leaves: 1,
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
                form: PointwiseForm::Float,
            })
        };
        let row = |width: usize| {
            KernelRequest::Row(RowSpec::Fused {
                out_shape: vec![4],
                numel: 4,
                leaves: vec![leaf()],
                kernel: RowKernel {
                    n_leaves: 1,
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
                bias: false,
                shapes: MatmulShapes {
                    out_shape: vec![2, 3],
                    a_shape: vec![2, 4],
                    b_shape: vec![4, b_cols],
                },
                fold: fold(6),
            })
        };
        let attention = |scale: f32| {
            KernelRequest::Attention(AttentionSpec::RegionDecode {
                bsz: 1,
                hq: 2,
                n_rep: 1,
                cap: 8,
                d: 4,
                scale_bits: scale.to_bits(),
                width: 4,
                mask_per_head: false,
            })
        };
        let transpose = |perm: Vec<usize>| {
            KernelRequest::Movement(MovementSpec::Transpose {
                dt: Ty::F32,
                out_shape: vec![2, 3],
                in_shape: vec![3, 2],
                perm,
                numel: 6,
            })
        };
        let packed = |grid_x: u32| {
            KernelRequest::Packed(PackedRequest {
                name: "packed_materialize",
                spec: PackedKernelSpec {
                    format: WeightFormat::Q8_0,
                    op: PackedKernelOp::Materialize,
                },
                grid: [grid_x, 1, 1],
                meta: vec![32],
            })
        };
        let packed_contraction = |schedule: Schedule| {
            KernelRequest::Packed(PackedRequest {
                name: "packed_contraction",
                spec: PackedKernelSpec {
                    format: WeightFormat::Q8_0,
                    op: PackedKernelOp::Contraction {
                        rows: RowSelect::Dense,
                        schedule,
                    },
                },
                grid: [64, 1, 1],
                meta: vec![32, 4],
            })
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
}
