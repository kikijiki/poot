//! The contraction epilogue: a pointwise region applied in registers to the head's result before the one
//! store, so a bias, a residual add or an activation costs no dispatch and no round trip through memory.
//!
//! The region is the kernelgen mirror of `poot_graph_ir`'s `ContractionEpilogue` region. Its local 0 is the
//! head's result **after the head's declared rounding**: a narrow head (BF16, F16) narrows the f32
//! accumulator and widens it again before the first step reads it, which is what the unfused program does by
//! storing the head in its own dtype. Every step then rounds through the same dtype, so a multi-step narrow
//! chain keeps each intermediate rounding the unfused kernels would have stored. A final store cannot stand in
//! for a removed rounding.
//!
//! The steps are plain operations, like the fused pointwise region's: a backend compiler may contract a
//! multiply feeding an add into one fused multiply-add. The no-contraction marker (`BinaryOpNoContract`) is
//! not used here because the SPIR-V postprocess places it by ordinal position in the compiled module, which
//! does not match the source block order once a kernel holds a loop (the decoration lands on the loop's own
//! accumulate); an epilogue after a K loop would silently remove the loop's contraction instead of its own.

use poot_kernel_ir::{Constant, Local, LocalDecl, Operand, Rvalue, Ty};
use poot_target::BufferStorage;

use crate::KernelGenError;
use crate::emit::{Emit, copy};
use crate::fused::{FusedScalarOp, emit_leaf_offset, emit_scalar_op};
use crate::helpers::{
    Layout, WeightLane, elem, ld, row_major_strides, slice_dtype, view_eff_strides,
};

/// An operand of an epilogue step.
#[derive(Clone, Copy, Debug)]
pub enum EpilogueInput {
    /// The head's result, after the head's rounding.
    Head,
    /// Extra input `j` (the request's `leaves`), read at the output element's broadcast position.
    Leaf(usize),
    /// An earlier step's result.
    Step(usize),
    /// An inline literal.
    Lit(f32),
}

/// One pointwise step: `step[s] = op(inputs...)`, rounded through the epilogue's dtype.
#[derive(Clone, Debug)]
pub struct EpilogueStep {
    pub op: FusedScalarOp,
    pub inputs: Vec<EpilogueInput>,
}

/// An extra input of an epilogue: its shape, broadcast against the head's output shape on the right as the graph's
/// pointwise ops do, and how its elements sit in its buffer.
#[derive(Clone, Debug)]
pub struct EpilogueLeaf {
    pub shape: Vec<usize>,
    /// `None` reads the leaf as the epilogue's `dt` elements; `Some(storage)` reads a narrow packed leaf and
    /// widens it to f32, which needs an f32 epilogue.
    pub storage: Option<BufferStorage>,
}

/// The epilogue of a contraction request: part of the kernel's identity, so two epilogues that differ in any
/// step, literal or leaf shape are different kernels.
///
/// `dt` is the head's dtype and the dtype of every step and the result, and of every leaf read as stored: a
/// narrow head with a wider bias is two dtypes `Binary` already refuses, so a region has exactly one. A narrow
/// weight a [`FusedScalarOp::Widen`] step widens (a bias stored BF16 under an f32 epilogue) is the exception: its
/// [`EpilogueLeaf::storage`] names the packed lane it is read through. `out_shape` is the head's output shape,
/// which the region runs over; `leaves[j]` is extra input `j`.
#[derive(Clone, Debug)]
pub struct Epilogue {
    pub dt: Ty,
    pub out_shape: Vec<usize>,
    pub leaves: Vec<EpilogueLeaf>,
    pub steps: Vec<EpilogueStep>,
    pub output: EpilogueInput,
}

impl Epilogue {
    /// Refuses an epilogue whose dtype is not f32, for a generator whose output is f32 (the dense `[N, K]`
    /// bodies): their head result is f32 and has no narrow store to round through.
    pub(crate) fn require_f32(&self, generator: &'static str) -> Result<(), KernelGenError> {
        if self.dt == Ty::F32 {
            Ok(())
        } else {
            Err(KernelGenError::Unsupported {
                request: generator,
                reason: "an epilogue over an f32 head must itself be f32",
            })
        }
    }

    /// The number of extra buffers the kernel takes between its contraction operands and its output.
    pub(crate) fn leaf_count(&self) -> usize {
        self.leaves.len()
    }

    /// The lane each leaf is read through: `None` for a leaf read as `dt` elements.
    pub(crate) fn lanes(
        &self,
        generator: &'static str,
    ) -> Result<Vec<Option<WeightLane>>, KernelGenError> {
        self.leaves
            .iter()
            .map(|leaf| match leaf.storage {
                None => Ok(None),
                Some(_) if self.dt != Ty::F32 => Err(KernelGenError::Unsupported {
                    request: generator,
                    reason: "a widened epilogue leaf needs an f32 epilogue",
                }),
                Some(storage) => WeightLane::of(generator, storage).map(Some),
            })
            .collect()
    }

    /// The parameter declaration of the leaf buffer read through `lane`.
    pub(crate) fn leaf_param(&self, lane: Option<WeightLane>) -> LocalDecl {
        match lane {
            None => ld(slice_dtype(self.dt.clone(), false), false),
            Some(lane) => lane.param(),
        }
    }
}

/// `value` rounded through `dt` and widened back to f32: the register image of storing it in `dt` and
/// reading it again. A no-op for f32.
fn round_through(e: &mut Emit, dt: &Ty, value: Local) -> Local {
    if !matches!(dt, Ty::BF16 | Ty::F16) {
        return value;
    }
    let narrow = e.let_(
        dt.clone(),
        Rvalue::Cast {
            to: dt.clone(),
            operand: copy(value),
        },
    );
    e.let_(
        Ty::F32,
        Rvalue::Cast {
            to: Ty::F32,
            operand: copy(narrow),
        },
    )
}

/// Emit `epilogue` over the head result `acc` (an f32 accumulator) of output element `out_index`, reading
/// leaf `j` from buffer `leaves[j]`; the value to store, widened to f32 (the caller narrows it to the
/// output dtype on store).
pub(crate) fn emit_epilogue(
    e: &mut Emit,
    epilogue: &Epilogue,
    out_index: Local,
    leaves: &[Local],
    lanes: &[Option<WeightLane>],
    acc: Local,
) -> Local {
    debug_assert_eq!(leaves.len(), epilogue.leaves.len());
    debug_assert_eq!(lanes.len(), epilogue.leaves.len());
    let head = round_through(e, &epilogue.dt, acc);
    let out_shape = epilogue.out_shape.as_slice();
    let out_strides = row_major_strides(out_shape);
    let mut leaf_values: Vec<Option<Local>> = vec![None; leaves.len()];
    let mut steps: Vec<Local> = Vec::with_capacity(epilogue.steps.len());
    for step in &epilogue.steps {
        let mut operands = Vec::with_capacity(step.inputs.len());
        for input in &step.inputs {
            operands.push(match *input {
                EpilogueInput::Head => copy(head),
                EpilogueInput::Step(s) => copy(steps[s]),
                EpilogueInput::Lit(value) => Operand::Const(Constant::F32(value)),
                EpilogueInput::Leaf(j) => {
                    let value = *leaf_values[j].get_or_insert_with(|| {
                        read_leaf(e, epilogue, j, lanes[j], &out_strides, out_index, leaves[j])
                    });
                    copy(value)
                }
            });
        }
        let result = {
            let (al, stmts) = e.parts();
            let result = al.add(Ty::F32, false);
            emit_scalar_op(result, step.op, &operands, al, stmts);
            result
        };
        steps.push(round_through(e, &epilogue.dt, result));
    }
    match epilogue.output {
        EpilogueInput::Head => head,
        EpilogueInput::Step(s) => steps[s],
        EpilogueInput::Leaf(j) => *leaf_values[j].get_or_insert_with(|| {
            read_leaf(e, epilogue, j, lanes[j], &out_strides, out_index, leaves[j])
        }),
        EpilogueInput::Lit(value) => {
            e.let_(Ty::F32, Rvalue::Use(Operand::Const(Constant::F32(value))))
        }
    }
}

/// Leaf `j` of `epilogue` (in `buffer`, read through `lane`) at the broadcast position of output element
/// `out_index`, widened to f32.
fn read_leaf(
    e: &mut Emit,
    epilogue: &Epilogue,
    j: usize,
    lane: Option<WeightLane>,
    out_strides: &[usize],
    out_index: Local,
    buffer: Local,
) -> Local {
    let (dt, shape, out_shape) = (&epilogue.dt, &epilogue.leaves[j].shape, &epilogue.out_shape);
    let (eff, base) = view_eff_strides(out_shape, shape, &Layout::contiguous(shape));
    let offset = {
        let (al, stmts) = e.parts();
        let offset = al.add(Ty::Usize, true);
        emit_leaf_offset(
            &eff,
            base,
            out_strides,
            out_shape,
            out_index,
            offset,
            al,
            stmts,
        );
        offset
    };
    if let Some(lane) = lane {
        return lane.emit_elem(e, buffer, offset);
    }
    let raw = e.let_(dt.clone(), Rvalue::Use(Operand::Copy(elem(buffer, offset))));
    if matches!(dt, Ty::BF16 | Ty::F16) {
        e.let_(
            Ty::F32,
            Rvalue::Cast {
                to: Ty::F32,
                operand: copy(raw),
            },
        )
    } else {
        raw
    }
}
