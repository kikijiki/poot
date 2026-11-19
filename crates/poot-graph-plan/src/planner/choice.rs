//! The kernel-choice record: which kernel the planner chose for one equation, kept beside its [`Plan`].
//!
//! The plan alone does not say whether a body was generated from a request, imported from a shipped
//! template, or never dispatched at all. [`KernelChoice`] does, and [`Planned`] is the one constructor
//! that builds a [`Plan`] and its choice together, so a plan's key, body and launch are always the ones
//! its choice names. The executor never reads the choice.

use crate::ViewBinding;
use crate::planner::Lowering;
use crate::planner::kg::Layout;
use crate::refusal::Site;
use crate::types::{Launch, ScratchDecl};
use std::fmt;
use std::hash::{Hash, Hasher};

use poot_kernelgen::{KernelRequest, Workgroup, generate};
use poot_target::{AieTargetId, Backend, DeviceCaps};

use crate::{
    AieAbiRefusal, Binding, Capability, Dispatch, DispatchPlan, ImportedKernel, Plan, PlanError,
};
use poot_graph_ir::ValueId;
use poot_kernel_ir::{Body, IndexWidth};
use poot_tensor::DType;

/// An equation that plans no kernel launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum NoDispatch {
    /// The output aliases an input buffer (reshape, an identity cast).
    Alias,
    /// The output is a strided view of an input buffer.
    View,
    /// The equation is a node of the target's graph artifact (an Intel NPU's vendor-compiled graph).
    Graph,
    /// The output is a side output another equation's dispatch writes (POOT-1456).
    SideOutput,
}

/// The kernel the planner chose for one equation.
// One record per planned equation, matched by value in many places: boxing the request would put a
// `Box` pattern wall between every caller and the request it reads.
#[derive(Clone, Debug, Hash)]
pub enum KernelChoice {
    /// A body `poot-kernelgen` generated from this request, launched as a thread grid.
    Generated(KernelRequest),
    /// The same request under the target's serialized one-core schedule (an AIE core): the ordinary body with
    /// its thread grid turned into loops, one invocation of a one-thread workgroup. `bindings` are what the
    /// generator proved of the body it produced (one per parameter, in binding order); the shape ABI the body
    /// is linked against is derived from them, so the body and its ABI come from one generation.
    Serialized {
        /// The request that was generated.
        request: KernelRequest,
        /// The numerical contract the request's op runs under on the core. It is part of the choice, so
        /// the plan key tells two contracts apart.
        contract: poot_kernelgen::NumericalContract,
        /// What the generator proved of each body parameter, in binding order.
        bindings: Box<[poot_kernelgen::SerializedBinding]>,
    },
    /// A shipped template body, bound with a metadata buffer of `meta_schema` words (zero for none). The
    /// count is read from the metadata the plan carries, never typed in a second place. `index` is the
    /// element type of the runtime index operand the equation binds, for a template that reads one: the
    /// body reads an F32 and an I32 index identically, but they are different bindings and must never
    /// share a plan key (R469-005).
    Imported {
        /// The shipped template.
        kernel: ImportedKernel,
        /// Number of words in the metadata buffer bound with the template; zero for none.
        meta_schema: u32,
        /// Element type of the runtime index operand the equation binds, if the template reads one.
        index: Option<DType>,
        /// The workgroup an elementwise template ([`ImportedKernel::is_elementwise`]) runs at, chosen before
        /// the plan is built from the device's grid cap; `None` for a template that owns its launch.
        workgroup: Option<Workgroup>,
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
            Self::Serialized { request, .. } => format!("serialized:{}", request.family()),
            Self::Imported { kernel, .. } => format!("imported:{kernel:?}").to_lowercase(),
            Self::NoDispatch(kind) => format!("nodispatch:{kind:?}").to_lowercase(),
        }
    }

    /// Whether this choice is a generated contraction whose generator takes a pointwise epilogue
    /// ([`poot_kernelgen::ContractionSpec::accepts_epilogue`]): the one answer to "does this head's own schedule
    /// carry a fused store".
    pub(crate) fn accepts_epilogue(&self) -> bool {
        matches!(
            self,
            Self::Generated(KernelRequest::Contraction(spec)) if spec.accepts_epilogue()
        )
    }

    /// The kernel's display label: a generated request names itself
    /// ([`KernelRequest::display_label`]), a shipped template its kernel. Carried by the plan beside its key.
    fn display_label(&self) -> String {
        match self {
            Self::Generated(request) | Self::Serialized { request, .. } => request.display_label(),
            other => other.label(),
        }
    }

    /// The workgroup X extent this choice declared before its body was generated, for a one-thread-per-element
    /// kernel; `None` for a kernel that owns its launch ([`poot_kernelgen::LaunchOwnership`]).
    pub(crate) fn elementwise_workgroup_x(&self) -> Option<u32> {
        match self {
            Self::Generated(request) | Self::Serialized { request, .. } => match request.launch() {
                poot_kernelgen::LaunchOwnership::Elementwise { workgroup_x } => Some(workgroup_x),
                poot_kernelgen::LaunchOwnership::OwnsLaunch => None,
            },
            Self::Imported { workgroup, .. } => workgroup.map(|workgroup| workgroup.lanes() as u32),
            Self::NoDispatch(_) => None,
        }
    }

    /// The plan key of a dispatch that runs this choice on `target`: the hex of its [`Self::digest`].
    pub(crate) fn key(&self, target: poot_codegen::Target) -> String {
        self.digest(target).to_string()
    }

    /// The digest of this choice's canonical encoding for `target`'s codegen epoch.
    fn digest(&self, target: poot_codegen::Target) -> ChoiceDigest {
        // The epoch, then the target itself, not only the flags the epoch folds in: two targets may share an
        // epoch's flags. The choice follows as its typed fields ([`encoding_digest`]).
        ChoiceDigest(encoding_digest(&(
            poot_codegen::kernel_cache_epoch(target),
            target,
            self,
        )))
    }

    /// The hex digest of this choice's typed fields alone, without the codegen epoch or a target: the identity
    /// of the request, stable across toolchain and codegen edits that change no field. The plan summary
    /// records it per equation.
    #[cfg(any(test, feature = "test-support"))]
    pub fn request_fingerprint(&self) -> String {
        encoding_digest(self)
            .iter()
            .try_fold(String::new(), |mut hex, byte| {
                fmt::Write::write_fmt(&mut hex, format_args!("{byte:02x}")).map(|()| hex)
            })
            .expect("write to a String")
    }
}

/// The canonical encoding of a value, digested: its fields in declaration order through [`Hash`], never its
/// `Debug` text. Every integer is written as fixed-width little-endian bytes, a slice and a string carry their
/// length, an enum carries its variant index and a float field is hashed by its bit pattern in its own `Hash`
/// impl. So a change to how a type prints changes nothing, a change to any field changes the digest, and the
/// encoding does not depend on the platform's word size.
///
/// The encoding follows the order of the fields and variants: reordering them moves every digest, which is the
/// intended signal (the plan digest file says a plan moved), and renaming one moves none.
fn encoding_digest<T: Hash + ?Sized>(value: &T) -> [u8; 32] {
    let mut writer = EncodingWriter(blake3::Hasher::new());
    value.hash(&mut writer);
    *writer.0.finalize().as_bytes()
}

/// A [`Hasher`] whose state is a blake3 hasher: it feeds every written byte through and fixes the byte order of
/// the integer writes, so the digest is the same on every platform. [`Hasher::finish`] is not the digest and is
/// never called; [`encoding_digest`] finalizes the blake3 state.
struct EncodingWriter(blake3::Hasher);

macro_rules! fixed_width_writes {
    ($($write:ident($int:ty)),* $(,)?) => {
        $(fn $write(&mut self, value: $int) {
            self.0.update(&value.to_le_bytes());
        })*
    };
}

impl Hasher for EncodingWriter {
    fn write(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fixed_width_writes!(
        write_u16(u16),
        write_u32(u32),
        write_u64(u64),
        write_u128(u128),
        write_i16(i16),
        write_i32(i32),
        write_i64(i64),
        write_i128(i128),
    );

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_isize(&mut self, value: isize) {
        self.write_i64(value as i64);
    }

    fn finish(&self) -> u64 {
        unreachable!("an EncodingWriter is finalized by `encoding_digest`, not by `Hasher::finish`")
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
    /// The plan the equation lowers to.
    pub plan: Plan,
    /// The kernel choice that produced `plan`.
    pub choice: KernelChoice,
}

/// One dispatch of a split plan. The choice that built it travels inside the [`Dispatch`].
pub(crate) struct PlannedDispatch {
    dispatch: Dispatch,
}

impl PlannedDispatch {
    /// This dispatch bound as `binding` says instead of to the equation's operands and output: a stage of
    /// a plan that passes an intermediate through a scratch buffer.
    pub(crate) fn bound(self, binding: Binding) -> Self {
        Self {
            dispatch: self.dispatch.with_binding(binding),
        }
    }
}

/// The binding of a dispatch over an equation's value operands: the kernel's params are `[operands..,
/// meta?, side writes.., out]`, so the operand count is what is left after the metadata buffer, the request's
/// side writes ([`KernelRequest::side_writes`]) and the output.
fn eqn_binding(body: &Body, launch: &Launch, side_writes: usize) -> Binding {
    let meta = usize::from(!launch.meta.is_empty());
    Binding::eqn_with_side_writes(
        body.params().count().saturating_sub(meta + side_writes + 1),
        side_writes,
    )
}

/// A one-dispatch plan with no scratch. Its binding is the equation's own, so only a kernel whose
/// params break the calling convention fails here.
fn single_plan(dispatch: Dispatch) -> Result<Plan, PlanError> {
    Ok(Plan::Dispatches(DispatchPlan::new(
        vec![dispatch],
        Vec::new(),
    )?))
}

/// A generation request, run for one target: the body, its launch and the choice that names them.
struct Built {
    body: Body,
    launch: poot_kernelgen::Launch,
    choice: KernelChoice,
    key: String,
    label: String,
}

impl Built {
    /// The one-dispatch form of this generation, bound to the equation's own operands.
    fn into_dispatch(self) -> PlannedDispatch {
        let Built {
            body,
            launch,
            choice,
            key,
            label,
        } = self;
        let launch = Launch {
            grid: launch.grid,
            meta: launch.meta.unwrap_or_default(),
        };
        let side_writes = match &choice {
            KernelChoice::Generated(request) | KernelChoice::Serialized { request, .. } => {
                request.side_writes()
            }
            KernelChoice::Imported { .. } | KernelChoice::NoDispatch(_) => 0,
        };
        let binding = eqn_binding(&body, &launch, side_writes);
        PlannedDispatch {
            dispatch: Dispatch::new(body, key, label, launch, binding, choice),
        }
    }
}

/// Generate `request` for `backend`. With `longest`, the longest element count the plan binds to any param of the
/// dispatch, the body runs 32-bit index math ([`IndexWidth::U32`]) where the target lowers it
/// (`poot_codegen::Capability::narrow_index`) and the generator's index bound keeps every index of a binding that
/// long within the 32-bit range ([`poot_kernelgen::index_bound`], POOT-1405): the planner's proof, which the
/// runtime checks again against the lengths it binds. The narrowed body is a different kernel, so its key says so.
fn build(
    site: Site<'_>,
    request: KernelRequest,
    backend: Backend,
    caps: &DeviceCaps,
    longest: Option<u64>,
) -> Result<Built, PlanError> {
    let generated = generate(&request, caps, site.limits())
        .map_err(|error| site.refuse(Capability::KernelGen(error)))?;
    let narrowed = longest
        .zip(poot_kernelgen::index_bound(&request))
        .filter(|&(longest, bound)| {
            poot_codegen::capability(backend.into()).narrow_index && bound.fits_u32(longest)
        })
        .map(|(_, bound)| bound);
    let mut body = generated.body;
    let choice = KernelChoice::Generated(request);
    let mut key = choice.key(backend.into());
    if let Some(bound) = narrowed {
        body.index_width = IndexWidth::U32(bound);
        key.push_str(":u32");
    }
    let label = choice.display_label();
    Ok(Built {
        body,
        launch: generated.launch,
        choice,
        key,
        label,
    })
}

/// The same request under an AIE core's sequential one-core schedule: one invocation of a one-thread
/// workgroup, with the access ranges the generator proved. Only the schedule is the target's; the request is
/// the one every target plans.
fn build_serialized(
    site: Site<'_>,
    request: KernelRequest,
    target: AieTargetId,
) -> Result<Built, PlanError> {
    let sequential = poot_kernelgen::serialized_schedule(&request, site.limits())
        .map_err(|error| site.refuse(Capability::KernelGen(error)))?;
    // The generator proved what the body does to each parameter; the graph says what each buffer is. One
    // binding per operand then the output, or the generator and the equation disagree about the ABI.
    let dims = site.dims();
    if dims.len() != sequential.accesses.len() {
        return Err(site.refuse(Capability::AieAbi(AieAbiRefusal::Bindings {
            eqn: site.eqn_out(),
        })));
    }
    let bindings = sequential
        .accesses
        .iter()
        .zip(dims)
        .map(|(access, dims)| access.over(dims))
        .collect();
    let launch = poot_kernelgen::Launch {
        grid: [1, 1, 1],
        meta: None,
    };
    let choice = KernelChoice::Serialized {
        request,
        contract: sequential.contract,
        bindings,
    };
    let key = choice.key(target.into());
    let label = choice.display_label();
    Ok(Built {
        body: sequential.body,
        launch,
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
        let PlannedDispatch { dispatch } = Self::generated_dispatch(site, request, backend, caps)?;
        let choice = dispatch.choice().clone();
        Ok(Self {
            plan: single_plan(dispatch)?,
            choice,
        })
    }

    /// [`Planned::generated`] for a dispatch whose longest binding the plan site knows: `longest` elements, the
    /// most it binds to any param. That is what lets [`build`] narrow the body's index math.
    pub(crate) fn generated_bounded(
        site: Site<'_>,
        request: KernelRequest,
        backend: Backend,
        caps: &DeviceCaps,
        longest: u64,
    ) -> Result<Self, PlanError> {
        let PlannedDispatch { dispatch } =
            Self::generated_dispatch_bounded(site, request, backend, caps, longest)?;
        let choice = dispatch.choice().clone();
        Ok(Self {
            plan: single_plan(dispatch)?,
            choice,
        })
    }

    /// [`Planned::generated_dispatch`] for a dispatch whose longest binding is `longest` elements
    /// ([`Planned::generated_bounded`]).
    pub(crate) fn generated_dispatch_bounded(
        site: Site<'_>,
        request: KernelRequest,
        backend: Backend,
        caps: &DeviceCaps,
        longest: u64,
    ) -> Result<PlannedDispatch, PlanError> {
        Ok(build(site, request, backend, caps, Some(longest))?.into_dispatch())
    }

    /// Generate `request` under `lowering`'s schedule: a thread grid for a GPU backend, the sequential
    /// one-core loop for an AIE target. The request is the same either way.
    pub(crate) fn lowered(
        site: Site<'_>,
        request: KernelRequest,
        lowering: Lowering<'_>,
    ) -> Result<Self, PlanError> {
        match lowering {
            Lowering::Gpu { backend, caps } => Self::generated(site, request, backend, caps),
            Lowering::Aie(target) => Self::serialized(site, request, target),
        }
    }

    /// Plan `request` for an AIE target: its sequential schedule as one dispatch.
    fn serialized(
        site: Site<'_>,
        request: KernelRequest,
        target: AieTargetId,
    ) -> Result<Self, PlanError> {
        let PlannedDispatch { dispatch } = build_serialized(site, request, target)?.into_dispatch();
        let choice = dispatch.choice().clone();
        Ok(Self {
            plan: single_plan(dispatch)?,
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
        Ok(build(site, request, backend, caps, None)?.into_dispatch())
    }

    /// Plan a shipped template. `meta` is its metadata buffer (`None` for a kernel that takes none). A
    /// one-thread-per-element template ([`ImportedKernel::is_elementwise`]) runs at the narrowest workgroup
    /// that clears the device's grid cap, chosen here and named by the choice. A template whose params break
    /// the dispatch calling convention is a typed error, not a panic.
    pub(crate) fn imported(
        kernel: ImportedKernel,
        backend: Backend,
        caps: &DeviceCaps,
        meta: Option<Vec<u32>>,
        grid: [u32; 3],
    ) -> Result<Self, PlanError> {
        let workgroup = kernel
            .is_elementwise()
            .then(|| Workgroup::for_threads(grid[0] as usize, caps));
        let PlannedDispatch { dispatch } =
            Self::imported_dispatch(kernel, backend, meta, grid, None, workgroup);
        let choice = dispatch.choice().clone();
        Ok(Self {
            plan: single_plan(dispatch)?,
            choice,
        })
    }

    /// One dispatch of a split plan that runs a shipped template; `meta` is its metadata, and the grid is
    /// `groups[0]` workgroups of the template's own workgroup size along X and `groups[1]` along Y.
    pub(crate) fn imported_split_dispatch(
        kernel: ImportedKernel,
        backend: Backend,
        meta: Vec<u32>,
        groups: [usize; 2],
    ) -> PlannedDispatch {
        let [x, y, z] = kernel.body().workgroup_size;
        let grid = [groups[0] as u32 * x, groups[1] as u32 * y, z];
        Self::imported_dispatch(kernel, backend, Some(meta), grid, None, None)
    }

    fn imported_dispatch(
        kernel: ImportedKernel,
        backend: Backend,
        meta: Option<Vec<u32>>,
        grid: [u32; 3],
        index: Option<DType>,
        workgroup: Option<Workgroup>,
    ) -> PlannedDispatch {
        let choice = imported_choice(kernel, meta.as_deref(), index, workgroup);
        let key = choice.key(backend.into());
        let label = choice.display_label();
        let mut body = kernel.body().clone();
        if let Some(workgroup) = workgroup {
            body.workgroup_size[0] = workgroup.lanes() as u32;
        }
        let launch = Launch {
            grid,
            meta: meta.unwrap_or_default(),
        };
        let binding = eqn_binding(&body, &launch, 0);
        PlannedDispatch {
            dispatch: Dispatch::new(body, key, label, launch, binding, choice),
        }
    }

    /// Several dispatches over disjoint ranges of one output, each bound to the equation's operands. Each
    /// dispatch keeps its own choice; the equation's recorded choice is its first dispatch's. An empty
    /// split (a zero-element output) has no choice to record and is refused.
    pub(crate) fn split(dispatches: Vec<PlannedDispatch>) -> Result<Self, PlanError> {
        Self::pipeline(dispatches, Vec::new())
    }

    /// Ordered dispatches over `scratch` buffers: a later dispatch reads what an earlier one wrote
    /// (POOT-ADR-0115). The equation's recorded choice is its first dispatch's. Each dispatch's binding
    /// is the one its [`PlannedDispatch::bound`] gave it; [`DispatchPlan::new`] validates the lot.
    pub(crate) fn pipeline(
        dispatches: Vec<PlannedDispatch>,
        scratch: Vec<ScratchDecl>,
    ) -> Result<Self, PlanError> {
        let Some(first) = dispatches.first() else {
            return Err(PlanError::BadShape(
                "a split plan needs at least one dispatch (zero-element output)".into(),
            ));
        };
        let choice = first.dispatch.choice().clone();
        let dispatches = dispatches
            .into_iter()
            .map(|PlannedDispatch { dispatch }| dispatch)
            .collect();
        Ok(Self {
            plan: Plan::Dispatches(DispatchPlan::new(dispatches, scratch)?),
            choice,
        })
    }

    pub(crate) fn alias(src: ValueId) -> Self {
        Self::no_dispatch(Plan::Alias(src), NoDispatch::Alias)
    }

    pub(crate) fn view(src: ValueId, view: &ViewBinding) -> Self {
        let Layout { strides, offset } = view.layout.clone();
        Self::no_dispatch(
            Plan::View {
                src,
                bound: view.bound,
                strides,
                offset,
            },
            NoDispatch::View,
        )
    }

    /// An equation that lowers to a node of the target's graph artifact.
    pub(crate) fn graph() -> Self {
        Self::no_dispatch(Plan::Graph, NoDispatch::Graph)
    }

    /// A side output of the region whose result is `region`: that region's dispatch writes it (POOT-1456).
    pub(crate) fn side_output(region: ValueId) -> Self {
        Self::no_dispatch(Plan::SideOutput { region }, NoDispatch::SideOutput)
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
    workgroup: Option<Workgroup>,
) -> KernelChoice {
    KernelChoice::Imported {
        kernel,
        meta_schema: meta.map_or(0, |meta| meta.len() as u32),
        index,
        workgroup,
    }
}

#[cfg(test)]
mod tests {
    use crate::planner::{Graph, OpKind};
    use crate::types::DispatchPlanError;
    use poot_graph_ir::{AttentionAttrs, Builder, TensorType};
    use poot_kernelgen::{
        AttentionSchedule, AttentionSink, AttentionSpec, KeyBound, KeyLayout, KeyWalk, MaskKind,
        ScoreTransform,
    };
    use poot_tensor::DType;
    use poot_test_util::device_caps::default_caps_for;
    use std::sync::Arc;

    use super::*;
    use crate::{CompileError, CompileOptions, Program, Target, compile};

    /// A single-sequence decode attention (`d` head dim, `scale` attention scale), traced as the
    /// primitive chain `ops::attention_masked` builds; `compile` forms the flash op from it.
    fn flash_decode_graph(d: usize, scale: f32) -> Graph {
        let (hq, cap) = (2usize, 8usize);
        let b = Builder::for_tests();
        let q = b
            .constant("q", TensorType::new(vec![1, hq, 1, d], DType::F32))
            .unwrap();
        let k = b
            .constant("k", TensorType::new(vec![1, hq, cap, d], DType::F32))
            .unwrap();
        let v = b
            .constant("v", TensorType::new(vec![1, hq, cap, d], DType::F32))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::new(vec![1, 1, 1, cap], DType::F32))
            .unwrap();
        let out = poot_graph_ir::ops::attention_masked(&b, q, k, v, 1, scale, mask).unwrap();
        b.finish(out).unwrap()
    }

    const OPTIONS: CompileOptions = CompileOptions::new(crate::CompileLimits::STANDARD);

    fn target(backend: Backend) -> Target {
        Target::Gpu {
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
            Plan::Dispatches(dispatches) if dispatches.len() == 1 => dispatches[0].key(),
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
        let b = Builder::for_tests();
        let q = b
            .constant("q", TensorType::new(vec![1, hq, 1, d], DType::F32))
            .unwrap();
        let k = b
            .constant("k", TensorType::new(vec![1, hq, cap, d], DType::F32))
            .unwrap();
        let v = b
            .constant("v", TensorType::new(vec![1, hq, cap, d], DType::F32))
            .unwrap();
        let mask = b
            .constant("mask", TensorType::new(vec![1, 1, 1, cap], DType::F32))
            .unwrap();
        let mut plan = b.append_plan(0).unwrap();
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
        b.finish(committed).unwrap()
    }

    /// Card 636 SC-004: a request the device cannot hold is a typed refusal out of `compile`, not a panic.
    /// A device whose workgroup memory cannot hold one attention lane at head dim 512 (2056 bytes) is
    /// refused with `AttentionLds` naming the head dim and the device's bytes; the same graph compiles on
    /// the real caps. Mutation: make the planner panic where it refuses; this row panics.
    #[test]
    fn a_device_too_small_for_the_request_compiles_to_a_typed_refusal() {
        let g = flash_decode_composite(512);
        let small = Target::Gpu {
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
            .digest(Backend::SpirvVulkan.into())
            .to_string()
    }

    /// POOT-1430: every operand precision of a request is a field of its encoding, so the exact, narrowed and split
    /// forms (each fragment dtype) key differently.
    #[test]
    fn every_operand_precision_of_a_request_has_its_own_digest() {
        use poot_graph_ir::{FragmentDtype, OperandPrecision};
        use poot_kernelgen::{
            ContractionSpec, GemvActivation, IndexedContraction, Schedule, WeightLayout,
        };
        let indexed = |precision| {
            KernelChoice::Generated(KernelRequest::Contraction(ContractionSpec::Indexed(
                IndexedContraction {
                    schedule: Schedule::Gemv {
                        width: 64,
                        cols: 1,
                        unroll: 1,
                        activation: GemvActivation::Scalar,
                        rows: poot_kernelgen::GemvRows::ONE,
                        depth: poot_kernelgen::LoadDepth::ONE,
                    },
                    layout: WeightLayout::Nk,
                    weight: poot_target::BufferStorage::f32(),
                    epilogue: None,
                    grid: [64, 1, 1],
                    meta: vec![4, 3, 2, 0, 0, 2],
                    precision,
                },
            )))
        };
        let digests: Vec<_> = [
            OperandPrecision::Exact,
            OperandPrecision::Narrow(FragmentDtype::Bf16),
            OperandPrecision::Narrow(FragmentDtype::F16),
            OperandPrecision::Split(FragmentDtype::Bf16),
            OperandPrecision::Split(FragmentDtype::F16),
        ]
        .into_iter()
        .map(|precision| {
            indexed(precision)
                .digest(Backend::SpirvVulkan.into())
                .to_string()
        })
        .collect();
        for (i, a) in digests.iter().enumerate() {
            for b in &digests[i + 1..] {
                assert_ne!(a, b, "{digests:?}");
            }
        }
    }

    /// Card 636: every field of a request is part of its encoding. This is per family, not per field: one field
    /// of each family (the `f32` literal a fused region carries, and for the packed family the `TileSize` edge and
    /// the Gemv launch shape) changes the digest.
    #[test]
    fn every_request_family_has_a_field_that_changes_the_digest() {
        use poot_kernel_ir::{BinOp, Ty};
        use poot_kernelgen::{
            CastSpec, ContractionSpec, Fold, FusedInput, FusedKernel, FusedScalarOp, FusedStep,
            LeafId, LeafTy, MatmulShapes, MovementOp, MovementSpec, PackedContraction,
            PackedKernelOp, PackedKernelSpec, PackedRequest, PointwiseForm, PointwiseSpec,
            RopeSpec, RowKernel, RowOp, RowSpec, RowValue, ScanKernel, ScanSpec, Schedule, StepId,
            Storage, TopKSpec, ViewOperand,
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
                inputs: vec![FusedInput::Leaf(LeafId::new(0)), FusedInput::Lit(literal)],
            }],
            output: FusedInput::Step(StepId::new(0)),
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
                row_shape: vec![4],
                numel: 4,
                leaves: vec![leaf()],
                kernel: RowKernel {
                    n_leaves: 1,
                    leaf_storage: vec![poot_target::BufferStorage::f32()],
                    n_cols: 4,
                    steps: vec![RowOp::Pointwise {
                        op: FusedScalarOp::Recip,
                        inputs: vec![FusedInput::Leaf(LeafId::new(0))],
                    }],
                    output: RowValue::Step(StepId::new(0)),
                    side_outputs: Vec::new(),
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
                kv: poot_kernelgen::KvLane::F32,
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
                sink: AttentionSink::Absent,
                schedule: AttentionSchedule::Streaming {
                    lanes: 32,
                    walk: KeyWalk { cluster: 4, run: 1 },
                },
                precision: poot_graph_ir::OperandPrecision::Exact,
            })
        };
        let transpose = |perm: Vec<usize>| {
            KernelRequest::Movement(MovementSpec::new(
                Storage::Elements { carrier: Ty::F32 },
                vec![2, 3],
                MovementOp::Copy {
                    source: vec![3, 2],
                    from: poot_kernelgen::Layout::contiguous(&[3, 2]).permuted(&perm),
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
                precision: poot_graph_ir::OperandPrecision::Exact,
            }))
        };
        // The register tilings of a short and a long contraction: two lane grids and register tiles.
        let tiled = |rows: usize| {
            poot_kernelgen::RegisterTiling::choose(
                poot_kernelgen::TiledShape {
                    blocks: 1,
                    m_per_block: rows,
                    out_per_block: 1024,
                    k: 1024,
                    layout: poot_kernelgen::WeightLayout::Nk,
                    rows: poot_kernelgen::TiledRows::Dense,
                    terms: poot_kernelgen::StagedTerms::ONE,
                    weight: poot_kernelgen::StagedWeight::Dense { bits: 32 },
                    activation_bits: 32,
                },
                &DeviceCaps::rocm_default(),
            )
            .schedule()
        };
        let gemv = |cols: u32| Schedule::Gemv {
            width: 256,
            cols,
            unroll: 8,
            activation: poot_kernelgen::GemvActivation::Scalar,
            rows: poot_kernelgen::GemvRows::ONE,
            depth: poot_kernelgen::LoadDepth::ONE,
        };
        let cast =
            |numel: usize| KernelRequest::Cast(CastSpec::F32ToBf16 { numel }, Workgroup::Narrow);
        let rope = |rot: usize| {
            KernelRequest::Rope(RopeSpec {
                dt: Ty::F32,
                x_shape: vec![1, 4],
                cos_shape: vec![1, 2],
                rot,
                numel: 4,
                workgroup: Workgroup::Narrow,
            })
        };
        let topk = |k: usize| {
            KernelRequest::TopK(TopKSpec {
                e: 8,
                k,
                numel: 2,
                workgroup: Workgroup::Narrow,
            })
        };
        // The same workgroup width and sequence, the rows shared by one or two lanes: the schedule field is
        // the only difference between two different bodies.
        let scan = |row_split: usize| {
            KernelRequest::Scan(ScanSpec {
                kernel: ScanKernel::DeltaRule,
                heads: 1,
                len: 8,
                rows: 8,
                width: 8,
                cols: 16,
                row_split,
            })
        };

        let pairs = [
            ("cast", cast(8), cast(9)),
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
            ("scan row split", scan(1), scan(2)),
            (
                "packed tiled lane grid",
                packed_contraction(tiled(16)),
                packed_contraction(tiled(2048)),
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

    /// A pointwise request over one `f32` literal: the smallest request whose field `Debug` prints lossily.
    fn literal_request(literal: f32) -> KernelRequest {
        use poot_kernel_ir::BinOp;
        use poot_kernelgen::{
            Fold, FusedInput, FusedKernel, FusedScalarOp, FusedStep, LeafId, LeafTy, PointwiseForm,
            PointwiseSpec, StepId, ViewOperand,
        };
        KernelRequest::Pointwise(PointwiseSpec {
            out_shape: vec![4],
            numel: 4,
            leaves: vec![ViewOperand {
                shape: vec![4],
                layout: poot_kernelgen::Layout::contiguous(&[4]),
            }],
            kernel: FusedKernel {
                leaf_tys: vec![LeafTy::F32; 1],
                steps: vec![FusedStep {
                    op: FusedScalarOp::Binary(BinOp::Add),
                    inputs: vec![FusedInput::Leaf(LeafId::new(0)), FusedInput::Lit(literal)],
                }],
                output: FusedInput::Step(StepId::new(0)),
            },
            form: PointwiseForm::Float {
                fold: Fold {
                    numel: 4,
                    two_d: false,
                    width: 256,
                },
            },
        })
    }

    /// POOT-1622: two requests that differ in one field have different digests, even when the field prints alike.
    /// Two NaN literals of different payloads print as `NaN` in `Debug`, so a digest of the request's `Debug` text
    /// gives them one key; the encoding of the fields gives each its own, and the same literal the same.
    /// Mutation: have `encoding_digest` hash `format!("{value:?}")` instead of `value.hash`; the two NaN
    /// requests collide and this row goes red.
    #[test]
    fn requests_differing_in_a_field_that_prints_alike_have_different_digests() {
        let (nan_a, nan_b) = (f32::from_bits(0x7fc0_0001), f32::from_bits(0x7fc0_0002));
        assert_eq!(
            format!("{:?}", literal_request(nan_a)),
            format!("{:?}", literal_request(nan_b)),
            "the premise: the two requests print alike"
        );
        assert_ne!(
            digest_of(literal_request(nan_a)),
            digest_of(literal_request(nan_b)),
            "two NaN payloads are two kernels"
        );
        assert_eq!(
            digest_of(literal_request(nan_a)),
            digest_of(literal_request(nan_a))
        );
    }

    /// A type whose `Debug` text is a switch of the test, not a function of its fields.
    #[derive(Hash)]
    struct Printed(u32);

    thread_local! {
        static PRINT_FIELD: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    }

    impl fmt::Debug for Printed {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if PRINT_FIELD.get() {
                write!(f, "Printed({})", self.0)
            } else {
                f.write_str("Printed")
            }
        }
    }

    /// POOT-1622: the digest is a function of the fields, not of how a type prints. The same value digests alike
    /// whether its `Debug` shows its field or hides it, and two values that differ in the field digest apart
    /// while their `Debug` text, hidden, is one string.
    /// Mutation: have `encoding_digest` hash `format!("{value:?}")` instead of `value.hash`; the digest of the
    /// value moves with its `Debug` text and this row goes red.
    #[test]
    fn a_debug_only_change_to_a_type_leaves_its_digest_unchanged() {
        let shown = encoding_digest(&Printed(7));
        PRINT_FIELD.set(false);
        let hidden = encoding_digest(&Printed(7));
        let other = encoding_digest(&Printed(8));
        PRINT_FIELD.set(true);
        assert_eq!(shown, hidden, "a change of `Debug` alone moved the digest");
        assert_ne!(hidden, other, "a changed field must change the digest");
    }

    /// A split plan with no dispatch has no choice to record: `Planned::split` refuses it as a typed
    /// error (a splitter over a zero-element output has no chunks to plan).
    /// Mutation: have `split` return `Ok` with a default choice for an empty list; the row goes red.
    #[test]
    fn an_empty_split_is_a_typed_refusal() {
        assert!(matches!(
            Planned::split(Vec::new()),
            Err(PlanError::BadShape(_))
        ));
    }

    /// POOT-1115: a shipped template whose read param is mutable breaks the dispatch calling convention.
    /// Planning it is a typed [`PlanError::InvalidDispatchPlan`], not a panic: `imported` returns the
    /// error of the same `single_plan` its callers reach with `?`.
    ///
    /// Mutation: have `single_plan` `expect` the validation result; the row panics instead of returning.
    #[test]
    fn a_template_with_a_mutable_read_param_is_a_typed_plan_error() {
        let PlannedDispatch { mut dispatch } = Planned::imported_dispatch(
            ImportedKernel::PackedBf16RowGather,
            Backend::SpirvVulkan,
            Some(vec![4]),
            [8, 1, 1],
            None,
            Some(Workgroup::Narrow),
        );
        let honest = dispatch.clone();
        single_plan(honest).expect("the shipped template follows the calling convention");
        let param = dispatch.body.params().next().expect("a read param");
        let slot = &mut Arc::make_mut(&mut dispatch.body).locals[param.index as usize].ty;
        let poot_kernel_ir::Ty::Ref { mutable, .. } = slot else {
            panic!("the template's read param is a reference: {slot:?}");
        };
        *mutable = true;
        match single_plan(dispatch) {
            Err(PlanError::InvalidDispatchPlan(DispatchPlanError::MutabilityMismatch {
                dispatch: 0,
                param: 0,
                kernel_mutable: true,
            })) => {}
            other => panic!("expected a MutabilityMismatch on param 0, got {other:?}"),
        }
    }
}
