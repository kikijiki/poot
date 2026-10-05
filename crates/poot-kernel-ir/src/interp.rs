//! The reference interpreter for a kernel-IR [`Body`]: the one CPU definition of what a body means, so a test
//! that checks a generated or imported body against an oracle does not carry a private interpreter that covers
//! only the ops of one kernel.
//!
//! [`run`] verifies the body ([`Body::verify`]), then executes every workgroup of a launch in order. Inside a
//! workgroup all lanes run in lockstep between barriers: each lane advances to its next `Barrier` (or `Return`),
//! then every lane moves past the barrier together, so a workgroup-local (LDS) write before a barrier is visible
//! to every lane's read after it. A lane that returns while another waits at a barrier, or lanes that park at
//! different barriers, is an error: the interpreter refuses it as a possible deadlock.
//!
//! Semantics, chosen to match what the codegen backends emit:
//!
//! - Integer arithmetic wraps; division by zero and a shift amount at or past the bit width are errors. `Shr`
//!   is arithmetic on `I32` and logical on `U32` and `Usize`.
//! - `F16` and `BF16` compute in `f32` and round to the narrow format after every operation (ties to even).
//! - Float math (`Exp`, `Sin`, ...) is the exact host function, not a device approximation.
//! - Buffers, private arrays and LDS are bounds-checked, and reading an element that was never written is an
//!   error, so a kernel that relies on uninitialised memory fails here instead of on a device.
//! - Atomics and compare-exchange are sequentially consistent and return the old value. Workgroups run one at a
//!   time, so a kernel that waits on another workgroup does not terminate: it hits the step limit.
//! - Warp-collective `Wmma*` statements are refused by name: their fragment layout is per target, not part of the
//!   backend-neutral IR.

use crate::{
    AtomicOp, BinOp, Body, Constant, Fp8Format, IndexAxis, IntScalarOp, Local, MathOp, Operand,
    Place, ProjectionElem, Rvalue, Site, Statement, Terminator, Ty, UnOp, VerifyError,
};

/// Basic blocks a run may execute, summed over every lane, before it is abandoned as non-terminating.
const STEP_LIMIT: u64 = 1 << 26;

// --- values -------------------------------------------------------------------------------------

/// One scalar of the IR's type lattice. `F16` and `BF16` hold an `f32` that is exactly representable in the
/// narrow format.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    Bool(bool),
    Usize(u64),
    F16(f32),
    BF16(f32),
    F32(f32),
    F64(f64),
    I32(i32),
    U32(u32),
}

impl Scalar {
    fn ty(self) -> Ty {
        match self {
            Scalar::Bool(_) => Ty::Bool,
            Scalar::Usize(_) => Ty::Usize,
            Scalar::F16(_) => Ty::F16,
            Scalar::BF16(_) => Ty::BF16,
            Scalar::F32(_) => Ty::F32,
            Scalar::F64(_) => Ty::F64,
            Scalar::I32(_) => Ty::I32,
            Scalar::U32(_) => Ty::U32,
        }
    }

    fn constant(c: &Constant) -> Scalar {
        match *c {
            Constant::Bool(v) => Scalar::Bool(v),
            Constant::Usize(v) => Scalar::Usize(v),
            Constant::I32(v) => Scalar::I32(v),
            Constant::U32(v) => Scalar::U32(v),
            Constant::F32(v) => Scalar::F32(v),
            Constant::F64(v) => Scalar::F64(v),
        }
    }

    /// The index an integer scalar denotes.
    fn index(self) -> Result<usize, Fault> {
        match self {
            Scalar::Usize(v) => Ok(v as usize),
            Scalar::U32(v) => Ok(v as usize),
            _ => Err(Fault::IllTyped("an index must be an unsigned integer")),
        }
    }
}

/// A register value: a scalar, or the lanes of a `Ty::Vec`.
#[derive(Clone, Debug)]
enum Value {
    Scalar(Scalar),
    Vector(Vec<Scalar>),
}

impl Value {
    fn scalar(self) -> Result<Scalar, Fault> {
        match self {
            Value::Scalar(s) => Ok(s),
            Value::Vector(_) => Err(Fault::IllTyped("expected a scalar, found a vector")),
        }
    }

    fn map(self, f: impl Fn(Scalar) -> Result<Scalar, Fault>) -> Result<Value, Fault> {
        match self {
            Value::Scalar(s) => f(s).map(Value::Scalar),
            Value::Vector(lanes) => lanes
                .into_iter()
                .map(f)
                .collect::<Result<_, _>>()
                .map(Value::Vector),
        }
    }

    fn zip(
        self,
        other: Value,
        f: impl Fn(Scalar, Scalar) -> Result<Scalar, Fault>,
    ) -> Result<Value, Fault> {
        match (self, other) {
            (Value::Scalar(a), Value::Scalar(b)) => f(a, b).map(Value::Scalar),
            (Value::Vector(a), Value::Vector(b)) if a.len() == b.len() => a
                .into_iter()
                .zip(b)
                .map(|(a, b)| f(a, b))
                .collect::<Result<_, _>>()
                .map(Value::Vector),
            _ => Err(Fault::IllTyped("operand shapes differ")),
        }
    }
}

/// A kernel parameter's storage: a typed, bounds-checked array of scalars.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    elem: Ty,
    data: Vec<Scalar>,
}

impl Buffer {
    // Card 671: no production caller anywhere in the workspace builds a `Buffer` from raw values or reads
    // one back as `f32`s - every non-test reference is poot-test-util's fixtures (reached only as a
    // dev-dependency); this crate's own unit tests use them too. Gated so a default build never has them,
    // instead of leaving them unconditionally `pub` with poot-test-util as the only non-test "caller"
    // keeping them off the dead-pub scanner.
    #[cfg(any(test, feature = "test-support"))]
    pub fn from_f32s(values: &[f32]) -> Self {
        Buffer {
            elem: Ty::F32,
            data: values.iter().map(|&v| Scalar::F32(v)).collect(),
        }
    }

    #[cfg(any(test, feature = "test-support"))]
    pub fn from_u32s(values: &[u32]) -> Self {
        Buffer {
            elem: Ty::U32,
            data: values.iter().map(|&v| Scalar::U32(v)).collect(),
        }
    }

    /// The elements of an `f32` buffer, or [`InterpError::NotF32`] for any other element type.
    #[cfg(any(test, feature = "test-support"))]
    pub fn to_f32s(&self) -> Result<Vec<f32>, InterpError> {
        self.data
            .iter()
            .map(|v| match v {
                Scalar::F32(v) => Ok(*v),
                _ => Err(InterpError::NotF32(self.elem.clone())),
            })
            .collect()
    }
}

// --- errors -------------------------------------------------------------------------------------

/// Why one lane could not continue.
#[derive(Clone, PartialEq, Debug, thiserror::Error)]
pub enum Fault {
    #[error("{what} index {index} is out of bounds for length {len}")]
    OutOfBounds {
        what: &'static str,
        index: usize,
        len: usize,
    },
    #[error("{what} element {index} is read before it is written")]
    Uninitialized { what: &'static str, index: usize },
    #[error("integer division by zero")]
    DivideByZero,
    #[error("shift amount {amount} is outside 0..{bits}")]
    ShiftOutOfRange { amount: i128, bits: u32 },
    #[error("cannot reinterpret {from:?} as {to:?}")]
    InvalidBitcast { from: Ty, to: Ty },
    #[error("{0}: the reference interpreter does not model it")]
    Unsupported(&'static str),
    /// A soundness backstop, not a live path: [`Body::verify`]'s static divergence analysis (card 618, a
    /// must-reach post-dominator check, hardened by its review) rejects every body that could produce
    /// this before [`run`] ever executes a lane, so this only fires if that analysis is ever wrong. Kept
    /// as a second, independent check on the interpreter's own lockstep model, the same way a type
    /// checker's callers still get a runtime panic on a real mismatch.
    #[error("a lane returned while another waits at a barrier: the body would deadlock")]
    BarrierDeadlock,
    /// See [`Fault::BarrierDeadlock`]: reachable only if [`Body::verify`]'s static check is unsound.
    #[error("lanes parked at different barriers in one round: bb{}, bb{}", .0, .1)]
    DivergentBarrier(u32, u32),
    #[error("more than {STEP_LIMIT} basic blocks executed: the body does not terminate")]
    StepLimit,
    #[error("ill-typed operation that the verifier accepts: {0}")]
    IllTyped(&'static str),
    #[error("kernel trap: a failed Assert or a reached Unreachable terminator ran (code {0})")]
    Trap(u32),
}

#[derive(Clone, PartialEq, Debug, thiserror::Error)]
pub enum InterpError {
    #[error(transparent)]
    Invalid(#[from] VerifyError),
    #[error("the body has {params} buffer params, {buffers} buffers were given")]
    BufferCount { params: usize, buffers: usize },
    #[error("buffer for param _{param} holds {found:?}, the body declares {expected:?}")]
    BufferType {
        param: usize,
        expected: Ty,
        found: Ty,
    },
    #[error("expected an f32 buffer, found {0:?} elements")]
    NotF32(Ty),
    #[error("workgroup {workgroup:?} lane {lane} at {site}: {fault}")]
    Fault {
        workgroup: [u32; 3],
        lane: u32,
        site: Site,
        fault: Fault,
    },
}

// --- launch -------------------------------------------------------------------------------------

/// Run `body` over a grid of `workgroups` (x, y, z). `buffers[i]` is param `_{i+1}`; stores land in place.
pub fn run(body: &Body, workgroups: [u32; 3], buffers: &mut [Buffer]) -> Result<(), InterpError> {
    body.verify()?;
    check_buffers(body, buffers)?;
    let mut steps = 0u64;
    for gz in 0..workgroups[2] {
        for gy in 0..workgroups[1] {
            for gx in 0..workgroups[0] {
                let group = [gx, gy, gz];
                let mut wg = Workgroup {
                    body,
                    buffers: &mut *buffers,
                    lds: body
                        .workgroup_locals
                        .iter()
                        .map(|decl| vec![None; decl.len as usize])
                        .collect(),
                    group,
                    steps: &mut steps,
                };
                wg.run().map_err(|(lane, site, fault)| InterpError::Fault {
                    workgroup: group,
                    lane,
                    site,
                    fault,
                })?;
            }
        }
    }
    Ok(())
}

fn check_buffers(body: &Body, buffers: &[Buffer]) -> Result<(), InterpError> {
    if buffers.len() != body.param_count as usize {
        return Err(InterpError::BufferCount {
            params: body.param_count as usize,
            buffers: buffers.len(),
        });
    }
    for (buffer, param) in buffers.iter().zip(body.params()) {
        let expected = match body.local_ty(param) {
            Ty::Ref { pointee, .. } => match &**pointee {
                Ty::Slice(elem) => (**elem).clone(),
                _ => unreachable!("verify accepts only slice params"),
            },
            _ => unreachable!("verify accepts only buffer params"),
        };
        if buffer.elem != expected {
            return Err(InterpError::BufferType {
                param: param.index as usize,
                expected,
                found: buffer.elem.clone(),
            });
        }
    }
    Ok(())
}

// --- execution ----------------------------------------------------------------------------------

/// One lane's registers and position.
struct Lane {
    block: usize,
    /// Indexed by local; `None` until assigned.
    locals: Vec<Option<Value>>,
    /// Private arrays, indexed by local; elements are `None` until stored.
    arrays: Vec<Option<Vec<Option<Scalar>>>>,
    local_id: [u32; 3],
}

/// Where a lane stopped.
enum Stop {
    /// Waiting at a barrier whose successor is this block.
    Barrier(usize),
    Return,
}

/// The storage a place addresses.
enum Cell {
    Local(usize),
    Private { local: usize, index: usize },
    Buffer { param: usize, index: usize },
}

type Located<T> = Result<T, (Site, Fault)>;

struct Workgroup<'a> {
    body: &'a Body,
    buffers: &'a mut [Buffer],
    lds: Vec<Vec<Option<Scalar>>>,
    group: [u32; 3],
    steps: &'a mut u64,
}

impl Workgroup<'_> {
    fn lane_count(&self) -> u32 {
        self.body.workgroup_size.iter().product()
    }

    fn new_lane(&self, id: u32) -> Lane {
        let [wx, wy, _] = self.body.workgroup_size;
        Lane {
            block: 0,
            locals: vec![None; self.body.locals.len()],
            arrays: self
                .body
                .locals
                .iter()
                .map(|decl| match &decl.ty {
                    Ty::Array { len, .. } => Some(vec![None; *len as usize]),
                    _ => None,
                })
                .collect(),
            local_id: [id % wx, (id / wx) % wy, id / (wx * wy)],
        }
    }

    /// Run every lane to completion, in lockstep between barriers.
    fn run(&mut self) -> Result<(), (u32, Site, Fault)> {
        let count = self.lane_count();
        let mut lanes: Vec<Lane> = (0..count).map(|id| self.new_lane(id)).collect();
        let mut returned = vec![false; count as usize];
        loop {
            let mut parked: Vec<Option<usize>> = vec![None; count as usize];
            for (id, lane) in lanes.iter_mut().enumerate() {
                if returned[id] {
                    continue;
                }
                match self
                    .step_lane(lane)
                    .map_err(|(site, f)| (id as u32, site, f))?
                {
                    Stop::Barrier(target) => parked[id] = Some(target),
                    Stop::Return => returned[id] = true,
                }
            }
            let mut targets = parked.iter().flatten().copied();
            let Some(first) = targets.next() else {
                return Ok(());
            };
            // The barrier a waiting lane sits at: the block it has not left yet.
            let barrier_site = |lane: usize| {
                let block = crate::BlockId {
                    index: lanes[lane].block as u32,
                };
                Site::Terminator { block }
            };
            // A lane that already returned can never reach this barrier.
            if returned.iter().any(|&done| done) {
                let waiting = parked.iter().position(Option::is_some).unwrap_or(0);
                return Err((
                    waiting as u32,
                    barrier_site(waiting),
                    Fault::BarrierDeadlock,
                ));
            }
            if let Some(other) = targets.find(|&t| t != first) {
                let waiting = parked.iter().position(|t| *t == Some(other)).unwrap_or(0);
                return Err((
                    waiting as u32,
                    barrier_site(waiting),
                    Fault::DivergentBarrier(first as u32, other as u32),
                ));
            }
            for (lane, target) in lanes.iter_mut().zip(&parked) {
                if let Some(target) = target {
                    lane.block = *target;
                }
            }
        }
    }

    /// Execute one lane until it parks at a barrier or returns.
    fn step_lane(&mut self, lane: &mut Lane) -> Located<Stop> {
        let body = self.body;
        loop {
            let block_id = crate::BlockId {
                index: lane.block as u32,
            };
            *self.steps += 1;
            if *self.steps > STEP_LIMIT {
                return Err((Site::Terminator { block: block_id }, Fault::StepLimit));
            }
            let block = &body.blocks[lane.block];
            for (index, statement) in block.statements.iter().enumerate() {
                self.statement(lane, statement).map_err(|fault| {
                    (
                        Site::Statement {
                            block: block_id,
                            index,
                        },
                        fault,
                    )
                })?;
            }
            let flow = self
                .terminator(lane, &block.terminator)
                .map_err(|fault| (Site::Terminator { block: block_id }, fault))?;
            if let Some(stop) = flow {
                return Ok(stop);
            }
        }
    }

    // --- places -----------------------------------------------------------------------------

    fn cell(&self, lane: &Lane, place: &Place) -> Result<Cell, Fault> {
        let index_of = |local: Local| -> Result<usize, Fault> {
            match &lane.locals[local.index as usize] {
                Some(Value::Scalar(s)) => s.index(),
                Some(Value::Vector(_)) => Err(Fault::IllTyped("an index must be a scalar")),
                None => Err(Fault::Uninitialized {
                    what: "index local",
                    index: local.index as usize,
                }),
            }
        };
        match place.projection.as_slice() {
            [] => Ok(Cell::Local(place.local.index as usize)),
            [ProjectionElem::Index(i)] => Ok(Cell::Private {
                local: place.local.index as usize,
                index: index_of(*i)?,
            }),
            [ProjectionElem::Deref, ProjectionElem::Index(i)] => Ok(Cell::Buffer {
                param: place.local.index as usize - 1,
                index: index_of(*i)?,
            }),
            _ => Err(Fault::IllTyped("a place is [], [Index] or [Deref, Index]")),
        }
    }

    fn load(&self, lane: &Lane, cell: &Cell) -> Result<Value, Fault> {
        match *cell {
            Cell::Local(local) => lane.locals[local].clone().ok_or(Fault::Uninitialized {
                what: "local",
                index: local,
            }),
            Cell::Private { local, index } => {
                let array = lane.arrays[local]
                    .as_ref()
                    .ok_or(Fault::IllTyped("indexing a local that is not an array"))?;
                element(array, "private array", index).map(Value::Scalar)
            }
            Cell::Buffer { param, index } => {
                let data = &self.buffers[param].data;
                data.get(index)
                    .copied()
                    .map(Value::Scalar)
                    .ok_or(Fault::OutOfBounds {
                        what: "buffer",
                        index,
                        len: data.len(),
                    })
            }
        }
    }

    fn store(&mut self, lane: &mut Lane, cell: &Cell, value: Value) -> Result<(), Fault> {
        match *cell {
            Cell::Local(local) => {
                lane.locals[local] = Some(value);
                Ok(())
            }
            Cell::Private { local, index } => {
                let array = lane.arrays[local]
                    .as_mut()
                    .ok_or(Fault::IllTyped("indexing a local that is not an array"))?;
                put(array, "private array", index, value.scalar()?)
            }
            Cell::Buffer { param, index } => {
                let data = &mut self.buffers[param].data;
                let len = data.len();
                *data.get_mut(index).ok_or(Fault::OutOfBounds {
                    what: "buffer",
                    index,
                    len,
                })? = value.scalar()?;
                Ok(())
            }
        }
    }

    fn operand(&self, lane: &Lane, operand: &Operand) -> Result<Value, Fault> {
        match operand {
            Operand::Const(c) => Ok(Value::Scalar(Scalar::constant(c))),
            Operand::Copy(place) | Operand::Move(place) => {
                self.load(lane, &self.cell(lane, place)?)
            }
        }
    }

    fn scalar_operand(&self, lane: &Lane, operand: &Operand) -> Result<Scalar, Fault> {
        self.operand(lane, operand)?.scalar()
    }

    fn index_operand(&self, lane: &Lane, operand: &Operand) -> Result<usize, Fault> {
        self.scalar_operand(lane, operand)?.index()
    }

    // --- statements -------------------------------------------------------------------------

    fn statement(&mut self, lane: &mut Lane, statement: &Statement) -> Result<(), Fault> {
        match statement {
            Statement::Assign(place, rvalue) => {
                let dest_ty = place
                    .projection
                    .is_empty()
                    .then(|| self.body.local_ty(place.local));
                let value = self.rvalue(lane, rvalue, dest_ty)?;
                let cell = self.cell(lane, place)?;
                self.store(lane, &cell, value)
            }
            Statement::StorageLive(_) | Statement::StorageDead(_) => Ok(()),
            // Lockstep execution is sequentially consistent: there is nothing to order.
            Statement::Fence { .. } => Ok(()),
            Statement::WorkgroupLocalWrite { idx, value, array } => {
                let index = self.index_operand(lane, idx)?;
                let value = self.scalar_operand(lane, value)?;
                put(
                    &mut self.lds[*array as usize],
                    "workgroup array",
                    index,
                    value,
                )
            }
            Statement::VectorStore { place, value } => {
                let Value::Vector(lanes) = self.operand(lane, value)? else {
                    return Err(Fault::IllTyped("a vector store needs a vector"));
                };
                let Cell::Buffer { param, index } = self.cell(lane, place)? else {
                    return Err(Fault::IllTyped("a vector store targets a buffer"));
                };
                let data = &mut self.buffers[param].data;
                let range = vector_range(index, lanes.len(), data.len())?;
                let target = &mut data[range];
                target.copy_from_slice(&lanes);
                Ok(())
            }
            Statement::WmmaLoad { .. } => Err(Fault::Unsupported("WmmaLoad")),
            Statement::WmmaMma { .. } => Err(Fault::Unsupported("WmmaMma")),
            Statement::WmmaStore { .. } => Err(Fault::Unsupported("WmmaStore")),
            Statement::WmmaZero { .. } => Err(Fault::Unsupported("WmmaZero")),
            Statement::WmmaLoadLds { .. } => Err(Fault::Unsupported("WmmaLoadLds")),
            Statement::WmmaStoreLds { .. } => Err(Fault::Unsupported("WmmaStoreLds")),
        }
    }

    fn rvalue(
        &mut self,
        lane: &mut Lane,
        rvalue: &Rvalue,
        dest: Option<&Ty>,
    ) -> Result<Value, Fault> {
        let lanes_of = |dest: Option<&Ty>| match dest {
            Some(Ty::Vec { lanes, .. }) => Ok(*lanes as usize),
            _ => Err(Fault::IllTyped("a vector rvalue assigns a Ty::Vec local")),
        };
        match rvalue {
            Rvalue::Use(operand) => self.operand(lane, operand),
            Rvalue::BinaryOp(op, a, b) | Rvalue::BinaryOpNoContract(op, a, b) => {
                // The interpreter evaluates plain Rust f32/f64 arithmetic, which never auto-fuses into an
                // FMA (that needs an explicit `mul_add` call this crate never makes), so the no-contraction
                // marker needs no special handling here: both variants are already unfused.
                let (a, b) = (self.operand(lane, a)?, self.operand(lane, b)?);
                a.zip(b, |a, b| binary(*op, a, b))
            }
            Rvalue::UnaryOp(op, a) => self.operand(lane, a)?.map(|a| unary(*op, a)),
            Rvalue::MathUnary(op, a) => self.operand(lane, a)?.map(|a| math(*op, a)),
            Rvalue::IntScalarUnary(op, a) => {
                let a = self.scalar_operand(lane, a)?;
                int_scalar(*op, a).map(Value::Scalar)
            }
            Rvalue::Len(place) => {
                let buffer = &self.buffers[place.local.index as usize - 1];
                Ok(Value::Scalar(Scalar::Usize(buffer.data.len() as u64)))
            }
            Rvalue::Cast { to, operand } => {
                let a = self.scalar_operand(lane, operand)?;
                cast(to, a).map(Value::Scalar)
            }
            Rvalue::Bitcast { to, operand } => {
                let a = self.scalar_operand(lane, operand)?;
                bitcast(to, a).map(Value::Scalar)
            }
            Rvalue::Fp8Decode { format, operand } => {
                let Scalar::U32(carrier) = self.scalar_operand(lane, operand)? else {
                    return Err(Fault::IllTyped("fp8 decode reads a u32 carrier"));
                };
                Ok(Value::Scalar(Scalar::F32(fp8_decode(*format, carrier))))
            }
            Rvalue::Fp8Encode { format, operand } => {
                let Scalar::F32(value) = self.scalar_operand(lane, operand)? else {
                    return Err(Fault::IllTyped("fp8 encode reads an f32"));
                };
                Ok(Value::Scalar(Scalar::U32(fp8_encode(*format, value))))
            }
            Rvalue::WorkgroupLocalRead { idx, array } => {
                let index = self.index_operand(lane, idx)?;
                element(&self.lds[*array as usize], "workgroup array", index).map(Value::Scalar)
            }
            Rvalue::WorkgroupLocalAtomic {
                idx,
                value,
                op,
                array,
            } => {
                let index = self.index_operand(lane, idx)?;
                let operand = self.scalar_operand(lane, value)?;
                let cell = &mut self.lds[*array as usize];
                let old = element(cell, "workgroup array", index)?;
                put(cell, "workgroup array", index, atomic(*op, old, operand)?)?;
                Ok(Value::Scalar(old))
            }
            Rvalue::GlobalAtomic { place, value, op } => {
                let operand = self.scalar_operand(lane, value)?;
                let cell = self.cell(lane, place)?;
                let old = self.load(lane, &cell)?.scalar()?;
                self.store(lane, &cell, Value::Scalar(atomic(*op, old, operand)?))?;
                Ok(Value::Scalar(old))
            }
            Rvalue::WorkgroupLocalCompareExchange {
                idx,
                expected,
                desired,
                array,
            } => {
                let index = self.index_operand(lane, idx)?;
                let expected = self.scalar_operand(lane, expected)?;
                let desired = self.scalar_operand(lane, desired)?;
                let cell = &mut self.lds[*array as usize];
                let old = element(cell, "workgroup array", index)?;
                if compare_exchange_matches(old, expected)? {
                    put(cell, "workgroup array", index, desired)?;
                }
                Ok(Value::Scalar(old))
            }
            Rvalue::GlobalCompareExchange {
                place,
                expected,
                desired,
            } => {
                let expected = self.scalar_operand(lane, expected)?;
                let desired = self.scalar_operand(lane, desired)?;
                let cell = self.cell(lane, place)?;
                let old = self.load(lane, &cell)?.scalar()?;
                if compare_exchange_matches(old, expected)? {
                    self.store(lane, &cell, Value::Scalar(desired))?;
                }
                Ok(Value::Scalar(old))
            }
            Rvalue::VectorLoad { place } => {
                let lanes = lanes_of(dest)?;
                let Cell::Buffer { param, index } = self.cell(lane, place)? else {
                    return Err(Fault::IllTyped("a vector load reads a buffer"));
                };
                let data = &self.buffers[param].data;
                let range = vector_range(index, lanes, data.len())?;
                Ok(Value::Vector(data[range].to_vec()))
            }
            Rvalue::VectorSplat(operand) => {
                let lanes = lanes_of(dest)?;
                let value = self.scalar_operand(lane, operand)?;
                Ok(Value::Vector(vec![value; lanes]))
            }
        }
    }

    // --- terminators ------------------------------------------------------------------------

    /// Advance the lane; `Some` when it stops (a barrier or a return).
    fn terminator(
        &mut self,
        lane: &mut Lane,
        terminator: &Terminator,
    ) -> Result<Option<Stop>, Fault> {
        match terminator {
            Terminator::Goto { target } => lane.block = target.index as usize,
            Terminator::SwitchInt { discr, targets } => {
                let key = switch_key(self.scalar_operand(lane, discr)?)?;
                lane.block = targets
                    .branches
                    .iter()
                    .find(|(value, _)| *value == key)
                    .map_or(targets.otherwise, |(_, target)| *target)
                    .index as usize;
            }
            Terminator::ThreadIndexCall {
                destination,
                dim,
                target,
            } => {
                let id = self.thread_index(lane, *dim);
                let cell = self.cell(lane, destination)?;
                self.store(lane, &cell, Value::Scalar(Scalar::Usize(id)))?;
                lane.block = target.index as usize;
            }
            Terminator::Barrier { target } => {
                return Ok(Some(Stop::Barrier(target.index as usize)));
            }
            Terminator::Return => return Ok(Some(Stop::Return)),
            Terminator::Trap { code } => return Err(Fault::Trap(*code)),
        }
        Ok(None)
    }

    fn thread_index(&self, lane: &Lane, axis: IndexAxis) -> u64 {
        let size = self.body.workgroup_size;
        let global =
            |d: usize| u64::from(self.group[d]) * u64::from(size[d]) + u64::from(lane.local_id[d]);
        match axis {
            IndexAxis::X => global(0),
            IndexAxis::Y => global(1),
            IndexAxis::Z => global(2),
            IndexAxis::LocalX => u64::from(lane.local_id[0]),
            IndexAxis::LocalY => u64::from(lane.local_id[1]),
            IndexAxis::LocalZ => u64::from(lane.local_id[2]),
            IndexAxis::GroupX => u64::from(self.group[0]),
            IndexAxis::GroupY => u64::from(self.group[1]),
            IndexAxis::GroupZ => u64::from(self.group[2]),
        }
    }
}

/// The element range of vector group `index` (`lanes` elements each) in a buffer of `len` elements.
fn vector_range(index: usize, lanes: usize, len: usize) -> Result<std::ops::Range<usize>, Fault> {
    let start = index.checked_mul(lanes);
    match start.and_then(|start| Some(start..start.checked_add(lanes)?)) {
        Some(range) if range.end <= len => Ok(range),
        _ => Err(Fault::OutOfBounds {
            what: "buffer",
            index: index.saturating_mul(lanes).saturating_add(lanes - 1),
            len,
        }),
    }
}

fn element(array: &[Option<Scalar>], what: &'static str, index: usize) -> Result<Scalar, Fault> {
    match array.get(index) {
        Some(Some(value)) => Ok(*value),
        Some(None) => Err(Fault::Uninitialized { what, index }),
        None => Err(Fault::OutOfBounds {
            what,
            index,
            len: array.len(),
        }),
    }
}

fn put(
    array: &mut [Option<Scalar>],
    what: &'static str,
    index: usize,
    value: Scalar,
) -> Result<(), Fault> {
    let len = array.len();
    *array
        .get_mut(index)
        .ok_or(Fault::OutOfBounds { what, index, len })? = Some(value);
    Ok(())
}

/// The value a `SwitchInt` compares: the operand's bit pattern at its own width, as MIR spells it.
fn switch_key(discr: Scalar) -> Result<u128, Fault> {
    match discr {
        Scalar::Bool(v) => Ok(u128::from(v)),
        Scalar::Usize(v) => Ok(u128::from(v)),
        Scalar::U32(v) => Ok(u128::from(v)),
        Scalar::I32(v) => Ok(u128::from(v as u32)),
        _ => Err(Fault::IllTyped(
            "a switch discriminant is a bool or an integer",
        )),
    }
}

// --- scalar operations --------------------------------------------------------------------------

macro_rules! int_binary {
    ($variant:ident, $op:expr, $x:expr, $y:expr) => {{
        let (x, y) = ($x, $y);
        Ok(match $op {
            BinOp::Add => Scalar::$variant(x.wrapping_add(y)),
            BinOp::Sub => Scalar::$variant(x.wrapping_sub(y)),
            BinOp::Mul => Scalar::$variant(x.wrapping_mul(y)),
            BinOp::Div if y == 0 => return Err(Fault::DivideByZero),
            BinOp::Div => Scalar::$variant(x.wrapping_div(y)),
            BinOp::Rem if y == 0 => return Err(Fault::DivideByZero),
            BinOp::Rem => Scalar::$variant(x.wrapping_rem(y)),
            BinOp::BitAnd => Scalar::$variant(x & y),
            BinOp::BitOr => Scalar::$variant(x | y),
            BinOp::BitXor => Scalar::$variant(x ^ y),
            BinOp::Min => Scalar::$variant(x.min(y)),
            BinOp::Max => Scalar::$variant(x.max(y)),
            BinOp::Lt => Scalar::Bool(x < y),
            BinOp::Le => Scalar::Bool(x <= y),
            BinOp::Gt => Scalar::Bool(x > y),
            BinOp::Ge => Scalar::Bool(x >= y),
            BinOp::Eq => Scalar::Bool(x == y),
            BinOp::Ne => Scalar::Bool(x != y),
            BinOp::Shl | BinOp::Shr => {
                unreachable!("shifts are handled before the same-type dispatch")
            }
        })
    }};
}

macro_rules! float_binary {
    ($op:expr, $x:expr, $y:expr, $make:expr) => {{
        let (x, y) = ($x, $y);
        Ok(match $op {
            BinOp::Add => $make(x + y),
            BinOp::Sub => $make(x - y),
            BinOp::Mul => $make(x * y),
            BinOp::Div => $make(x / y),
            BinOp::Rem => $make(x % y),
            BinOp::Min => $make(x.min(y)),
            BinOp::Max => $make(x.max(y)),
            BinOp::Lt => Scalar::Bool(x < y),
            BinOp::Le => Scalar::Bool(x <= y),
            BinOp::Gt => Scalar::Bool(x > y),
            BinOp::Ge => Scalar::Bool(x >= y),
            BinOp::Eq => Scalar::Bool(x == y),
            BinOp::Ne => Scalar::Bool(x != y),
            BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                return Err(Fault::IllTyped("a bit operation on floats"));
            }
        })
    }};
}

fn binary(op: BinOp, a: Scalar, b: Scalar) -> Result<Scalar, Fault> {
    if matches!(op, BinOp::Shl | BinOp::Shr) {
        return shift(op, a, b);
    }
    match (a, b) {
        (Scalar::I32(x), Scalar::I32(y)) => int_binary!(I32, op, x, y),
        (Scalar::U32(x), Scalar::U32(y)) => int_binary!(U32, op, x, y),
        (Scalar::Usize(x), Scalar::Usize(y)) => int_binary!(Usize, op, x, y),
        (Scalar::F32(x), Scalar::F32(y)) => float_binary!(op, x, y, Scalar::F32),
        (Scalar::F64(x), Scalar::F64(y)) => float_binary!(op, x, y, Scalar::F64),
        (Scalar::F16(x), Scalar::F16(y)) => {
            float_binary!(op, x, y, |v| Scalar::F16(round_to_f16(v)))
        }
        (Scalar::BF16(x), Scalar::BF16(y)) => {
            float_binary!(op, x, y, |v| Scalar::BF16(round_to_bf16(v)))
        }
        (Scalar::Bool(x), Scalar::Bool(y)) => Ok(Scalar::Bool(match op {
            BinOp::BitAnd => x & y,
            BinOp::BitOr => x | y,
            BinOp::BitXor | BinOp::Ne => x ^ y,
            BinOp::Eq => x == y,
            _ => return Err(Fault::IllTyped("arithmetic on bools")),
        })),
        _ => Err(Fault::IllTyped("binary operands have different types")),
    }
}

fn shift(op: BinOp, value: Scalar, amount: Scalar) -> Result<Scalar, Fault> {
    let amount: i128 = match amount {
        Scalar::I32(v) => i128::from(v),
        Scalar::U32(v) => i128::from(v),
        Scalar::Usize(v) => i128::from(v),
        _ => return Err(Fault::IllTyped("a shift amount is an integer")),
    };
    // The amount as a shift count for a value of `bits` bits.
    let count = |bits: u32| {
        u32::try_from(amount)
            .ok()
            .filter(|n| *n < bits)
            .ok_or(Fault::ShiftOutOfRange { amount, bits })
    };
    let left = op == BinOp::Shl;
    match value {
        Scalar::I32(v) => count(32).map(|n| Scalar::I32(if left { v << n } else { v >> n })),
        Scalar::U32(v) => count(32).map(|n| Scalar::U32(if left { v << n } else { v >> n })),
        Scalar::Usize(v) => count(64).map(|n| Scalar::Usize(if left { v << n } else { v >> n })),
        _ => Err(Fault::IllTyped("a shifted value is an integer")),
    }
}

fn unary(op: UnOp, a: Scalar) -> Result<Scalar, Fault> {
    Ok(match (op, a) {
        (UnOp::Neg, Scalar::I32(v)) => Scalar::I32(v.wrapping_neg()),
        (UnOp::Neg, Scalar::U32(v)) => Scalar::U32(v.wrapping_neg()),
        (UnOp::Neg, Scalar::Usize(v)) => Scalar::Usize(v.wrapping_neg()),
        (UnOp::Neg, Scalar::F16(v)) => Scalar::F16(-v),
        (UnOp::Neg, Scalar::BF16(v)) => Scalar::BF16(-v),
        (UnOp::Neg, Scalar::F32(v)) => Scalar::F32(-v),
        (UnOp::Neg, Scalar::F64(v)) => Scalar::F64(-v),
        (UnOp::Not, Scalar::Bool(v)) => Scalar::Bool(!v),
        (UnOp::Not, Scalar::I32(v)) => Scalar::I32(!v),
        (UnOp::Not, Scalar::U32(v)) => Scalar::U32(!v),
        (UnOp::Not, Scalar::Usize(v)) => Scalar::Usize(!v),
        _ => return Err(Fault::IllTyped("unary operator on an unsupported type")),
    })
}

fn math_f32(op: MathOp, x: f32) -> f32 {
    match op {
        MathOp::Sqrt => x.sqrt(),
        MathOp::Abs => x.abs(),
        MathOp::Floor => x.floor(),
        MathOp::Ceil => x.ceil(),
        MathOp::Trunc => x.trunc(),
        MathOp::Round => x.round(),
        MathOp::Exp => x.exp(),
        MathOp::Log => x.ln(),
        MathOp::Sin => x.sin(),
        MathOp::Cos => x.cos(),
    }
}

fn math_f64(op: MathOp, x: f64) -> f64 {
    match op {
        MathOp::Sqrt => x.sqrt(),
        MathOp::Abs => x.abs(),
        MathOp::Floor => x.floor(),
        MathOp::Ceil => x.ceil(),
        MathOp::Trunc => x.trunc(),
        MathOp::Round => x.round(),
        MathOp::Exp => x.exp(),
        MathOp::Log => x.ln(),
        MathOp::Sin => x.sin(),
        MathOp::Cos => x.cos(),
    }
}

fn math(op: MathOp, a: Scalar) -> Result<Scalar, Fault> {
    Ok(match a {
        Scalar::F32(v) => Scalar::F32(math_f32(op, v)),
        Scalar::F64(v) => Scalar::F64(math_f64(op, v)),
        Scalar::F16(v) => Scalar::F16(round_to_f16(math_f32(op, v))),
        Scalar::BF16(v) => Scalar::BF16(round_to_bf16(math_f32(op, v))),
        _ => return Err(Fault::IllTyped("float math on a non-float")),
    })
}

fn int_scalar(op: IntScalarOp, a: Scalar) -> Result<Scalar, Fault> {
    let count = |bits: u32| match op {
        IntScalarOp::CountOnes => bits.count_ones(),
        IntScalarOp::LeadingZeros => bits.leading_zeros(),
        IntScalarOp::TrailingZeros => bits.trailing_zeros(),
    };
    match a {
        Scalar::U32(v) => Ok(Scalar::U32(count(v))),
        Scalar::I32(v) => Ok(Scalar::I32(count(v as u32) as i32)),
        _ => Err(Fault::IllTyped(
            "an integer bit operation reads a 32-bit word",
        )),
    }
}

/// A numeric scalar widened to the value it denotes, for casting.
enum Numeric {
    Int(i128),
    Float(f64),
}

fn cast(to: &Ty, a: Scalar) -> Result<Scalar, Fault> {
    let source = match a {
        Scalar::Bool(v) => Numeric::Int(i128::from(v)),
        Scalar::Usize(v) => Numeric::Int(i128::from(v)),
        Scalar::I32(v) => Numeric::Int(i128::from(v)),
        Scalar::U32(v) => Numeric::Int(i128::from(v)),
        Scalar::F16(v) | Scalar::BF16(v) | Scalar::F32(v) => Numeric::Float(f64::from(v)),
        Scalar::F64(v) => Numeric::Float(v),
    };
    // Rust `as` semantics: integers wrap, float to integer saturates and maps NaN to zero.
    Ok(match (to, source) {
        (Ty::I32, Numeric::Int(v)) => Scalar::I32(v as i32),
        (Ty::I32, Numeric::Float(v)) => Scalar::I32(v as i32),
        (Ty::U32, Numeric::Int(v)) => Scalar::U32(v as u32),
        (Ty::U32, Numeric::Float(v)) => Scalar::U32(v as u32),
        (Ty::Usize, Numeric::Int(v)) => Scalar::Usize(v as u64),
        (Ty::Usize, Numeric::Float(v)) => Scalar::Usize(v as u64),
        (Ty::F32, Numeric::Int(v)) => Scalar::F32(v as f32),
        (Ty::F32, Numeric::Float(v)) => Scalar::F32(v as f32),
        (Ty::F64, Numeric::Int(v)) => Scalar::F64(v as f64),
        (Ty::F64, Numeric::Float(v)) => Scalar::F64(v),
        (Ty::F16, Numeric::Int(v)) => Scalar::F16(round_to_f16(v as f32)),
        (Ty::F16, Numeric::Float(v)) => Scalar::F16(round_to_f16(v as f32)),
        (Ty::BF16, Numeric::Int(v)) => Scalar::BF16(round_to_bf16(v as f32)),
        (Ty::BF16, Numeric::Float(v)) => Scalar::BF16(round_to_bf16(v as f32)),
        _ => return Err(Fault::IllTyped("a cast targets a numeric scalar")),
    })
}

fn bitcast(to: &Ty, a: Scalar) -> Result<Scalar, Fault> {
    let bits = match a {
        Scalar::I32(v) => v as u32,
        Scalar::U32(v) => v,
        Scalar::F32(v) => v.to_bits(),
        other => {
            return Err(Fault::InvalidBitcast {
                from: other.ty(),
                to: to.clone(),
            });
        }
    };
    match to {
        Ty::I32 => Ok(Scalar::I32(bits as i32)),
        Ty::U32 => Ok(Scalar::U32(bits)),
        Ty::F32 => Ok(Scalar::F32(f32::from_bits(bits))),
        other => Err(Fault::InvalidBitcast {
            from: a.ty(),
            to: other.clone(),
        }),
    }
}

fn atomic(op: AtomicOp, old: Scalar, operand: Scalar) -> Result<Scalar, Fault> {
    match op {
        AtomicOp::Add => binary(BinOp::Add, old, operand),
        AtomicOp::Min => binary(BinOp::Min, old, operand),
        AtomicOp::Max => binary(BinOp::Max, old, operand),
        AtomicOp::And => binary(BinOp::BitAnd, old, operand),
        AtomicOp::Or => binary(BinOp::BitOr, old, operand),
        AtomicOp::Xor => binary(BinOp::BitXor, old, operand),
        AtomicOp::Exchange => Ok(operand),
    }
}

/// Whether a compare-exchange replaces the cell. Integer-only, like the codegen lowering.
fn compare_exchange_matches(old: Scalar, expected: Scalar) -> Result<bool, Fault> {
    match old {
        Scalar::I32(_) | Scalar::U32(_) | Scalar::Usize(_) => Ok(old == expected),
        _ => Err(Fault::IllTyped("compare-exchange is integer-only")),
    }
}

// --- narrow floats and FP8 ----------------------------------------------------------------------

/// Round to the nearest `f16` (ties to even); overflow becomes infinity.
fn round_to_f16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let magnitude = x.abs();
    let rounded = if magnitude >= 65520.0 {
        f32::INFINITY
    } else if magnitude < f32::from_bits(0x3880_0000) {
        // Subnormal range (below 2^-14): a fixed quantum of 2^-24.
        let quantum = f32::from_bits(0x3380_0000);
        (magnitude / quantum).round_ties_even() * quantum
    } else {
        // Normal range: 10 mantissa bits, so the quantum is 2^(exponent - 10).
        let exponent = (magnitude.to_bits() >> 23) as i32 - 127;
        let quantum = f32::from_bits(((exponent - 10 + 127) as u32) << 23);
        (magnitude / quantum).round_ties_even() * quantum
    };
    rounded.copysign(x)
}

/// Round to the nearest `bfloat16` (ties to even).
fn round_to_bf16(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let bits = x.to_bits();
    let round_up = 0x7fff + ((bits >> 16) & 1);
    f32::from_bits(bits.wrapping_add(round_up) & 0xffff_0000)
}

/// The value of the non-negative E4M3FN code `code` (`0..=0x7e`).
fn e4m3fn_magnitude(code: u8) -> f32 {
    let exponent = i32::from(code >> 3);
    let mantissa = f32::from(code & 7);
    if exponent == 0 {
        mantissa / 512.0
    } else {
        (1.0 + mantissa / 8.0) * 2.0f32.powi(exponent - 7)
    }
}

fn fp8_decode(format: Fp8Format, carrier: u32) -> f32 {
    match format {
        Fp8Format::E4M3Fn => {
            let code = (carrier & 0xff) as u8;
            if code & 0x7f == 0x7f {
                return f32::NAN;
            }
            let magnitude = e4m3fn_magnitude(code & 0x7f);
            if code & 0x80 == 0 {
                magnitude
            } else {
                -magnitude
            }
        }
    }
}

/// Round to the nearest E4M3FN code (ties to even code), saturating at the largest finite value; NaN encodes
/// as `0x7f`.
fn fp8_encode(format: Fp8Format, value: f32) -> u32 {
    match format {
        Fp8Format::E4M3Fn => {
            if value.is_nan() {
                return 0x7f;
            }
            let magnitude = value.abs();
            let mut code = 0u8;
            for upper in 1u8..=0x7e {
                let midpoint = (e4m3fn_magnitude(upper - 1) + e4m3fn_magnitude(upper)) * 0.5;
                let take_upper = if upper % 2 == 0 {
                    magnitude >= midpoint
                } else {
                    magnitude > midpoint
                };
                if take_upper {
                    code = upper;
                }
            }
            let sign = if value.is_sign_negative() { 0x80 } else { 0 };
            u32::from(code | sign)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BasicBlock, BlockId, LocalDecl, SwitchTargets, VerifyErrorKind, WorkgroupLocalDecl,
    };

    fn l(index: u32) -> Local {
        Local { index }
    }

    fn buffer_ty(elem: Ty) -> Ty {
        Ty::Ref {
            mutable: true,
            pointee: Box::new(Ty::Slice(Box::new(elem))),
        }
    }

    fn copy(local: u32) -> Operand {
        Operand::Copy(Place::local(l(local)))
    }

    fn cusize(v: u64) -> Operand {
        Operand::Const(Constant::Usize(v))
    }

    fn cu32(v: u32) -> Operand {
        Operand::Const(Constant::U32(v))
    }

    fn element_of(param: u32, index: u32) -> Place {
        Place {
            local: l(param),
            projection: vec![ProjectionElem::Deref, ProjectionElem::Index(l(index))],
        }
    }

    fn assign(local: u32, rvalue: Rvalue) -> Statement {
        Statement::Assign(Place::local(l(local)), rvalue)
    }

    fn goto(index: u32) -> Terminator {
        Terminator::Goto {
            target: BlockId { index },
        }
    }

    fn thread_index(dest: u32, dim: IndexAxis, target: u32) -> Terminator {
        Terminator::ThreadIndexCall {
            destination: Place::local(l(dest)),
            dim,
            target: BlockId { index: target },
        }
    }

    fn block(statements: Vec<Statement>, terminator: Terminator) -> BasicBlock {
        BasicBlock {
            statements,
            terminator,
        }
    }

    /// `_0: ()`, then the local types (`_1..=param_count` are the params).
    fn body_of(param_count: u32, tys: Vec<Ty>, blocks: Vec<BasicBlock>, wg: [u32; 3]) -> Body {
        let locals = std::iter::once(Ty::Unit)
            .chain(tys)
            .map(|ty| LocalDecl { ty, mutable: true })
            .collect();
        let mut body = Body::new("k", param_count, locals, blocks);
        body.workgroup_size = wg;
        body
    }

    fn u32s(buffer: &Buffer) -> Vec<u32> {
        buffer
            .data
            .iter()
            .map(|v| match v {
                Scalar::U32(v) => *v,
                other => panic!("expected u32, found {other:?}"),
            })
            .collect()
    }

    /// Every lane of `groups` workgroups of `wg` lanes atomically adds 1 to `counter[0]` and records the
    /// value it saw in `seen[global id]`.
    fn atomic_counter_body(wg: u32, op: AtomicOp) -> Body {
        // _1 counter, _2 seen, _3 gid, _4 zero, _5 old
        body_of(
            2,
            vec![
                buffer_ty(Ty::U32),
                buffer_ty(Ty::U32),
                Ty::Usize,
                Ty::Usize,
                Ty::U32,
            ],
            vec![
                block(vec![], thread_index(3, IndexAxis::X, 1)),
                block(
                    vec![
                        assign(4, Rvalue::Use(cusize(0))),
                        assign(
                            5,
                            Rvalue::GlobalAtomic {
                                place: element_of(1, 4),
                                value: cu32(1),
                                op,
                            },
                        ),
                        Statement::Assign(element_of(2, 3), Rvalue::Use(copy(5))),
                    ],
                    Terminator::Return,
                ),
            ],
            [wg, 1, 1],
        )
    }

    #[test]
    fn atomic_add_across_workgroups_counts_every_lane_and_returns_old_values() {
        let (groups, wg) = (3u32, 4u32);
        let body = atomic_counter_body(wg, AtomicOp::Add);
        let mut buffers = [Buffer::from_u32s(&[0]), Buffer::from_u32s(&[u32::MAX; 12])];
        run(&body, [groups, 1, 1], &mut buffers).unwrap();
        assert_eq!(
            u32s(&buffers[0]),
            vec![groups * wg],
            "every lane of every workgroup must land its add"
        );
        // Sequential consistency: the old values seen are exactly 0..lanes, each once.
        let mut seen = u32s(&buffers[1]);
        seen.sort_unstable();
        assert_eq!(seen, (0..groups * wg).collect::<Vec<_>>());
    }

    #[test]
    fn atomic_exchange_max_min_and_bit_ops_update_the_cell() {
        // A single lane applying one op: cell 0b1100, operand 0b1010.
        let cases = [
            (AtomicOp::Exchange, 0b1010),
            (AtomicOp::Max, 0b1100),
            (AtomicOp::Min, 0b1010),
            (AtomicOp::And, 0b1000),
            (AtomicOp::Or, 0b1110),
            (AtomicOp::Xor, 0b0110),
        ];
        for (op, want) in cases {
            // _1 cell, _2 zero, _3 old
            let body = body_of(
                1,
                vec![buffer_ty(Ty::U32), Ty::Usize, Ty::U32],
                vec![block(
                    vec![
                        assign(2, Rvalue::Use(cusize(0))),
                        assign(
                            3,
                            Rvalue::GlobalAtomic {
                                place: element_of(1, 2),
                                value: cu32(0b1010),
                                op,
                            },
                        ),
                    ],
                    Terminator::Return,
                )],
                [1, 1, 1],
            );
            let mut buffers = [Buffer::from_u32s(&[0b1100])];
            run(&body, [1, 1, 1], &mut buffers).unwrap();
            assert_eq!(u32s(&buffers[0]), vec![want], "{op:?}");
        }
    }

    #[test]
    fn compare_exchange_replaces_only_on_a_match() {
        // Lane 0 CASes 5 -> 9 on cell 0 (matches), lane 1 CASes 5 -> 7 (cell now 9, no match).
        // _1 cell, _2 zero, _3 lane, _4 old
        let body = body_of(
            1,
            vec![buffer_ty(Ty::U32), Ty::Usize, Ty::Usize, Ty::U32],
            vec![
                block(vec![], thread_index(3, IndexAxis::LocalX, 1)),
                block(
                    vec![assign(2, Rvalue::Use(cusize(0)))],
                    Terminator::SwitchInt {
                        discr: copy(3),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                block(
                    vec![assign(
                        4,
                        Rvalue::GlobalCompareExchange {
                            place: element_of(1, 2),
                            expected: cu32(5),
                            desired: cu32(9),
                        },
                    )],
                    Terminator::Return,
                ),
                block(
                    vec![assign(
                        4,
                        Rvalue::GlobalCompareExchange {
                            place: element_of(1, 2),
                            expected: cu32(5),
                            desired: cu32(7),
                        },
                    )],
                    Terminator::Return,
                ),
            ],
            [2, 1, 1],
        );
        let mut buffers = [Buffer::from_u32s(&[5])];
        run(&body, [1, 1, 1], &mut buffers).unwrap();
        assert_eq!(u32s(&buffers[0]), vec![9]);
    }

    /// Each lane writes its global id to LDS, barriers, then reads its neighbour's slot and stores it at its
    /// own global id.
    fn neighbour_body(wg: u32) -> Body {
        // _1 out, _2 gid, _3 lid, _4 next, _5 gid as u32, _6 read
        let mut body = body_of(
            1,
            vec![
                buffer_ty(Ty::U32),
                Ty::Usize,
                Ty::Usize,
                Ty::Usize,
                Ty::U32,
                Ty::U32,
            ],
            vec![
                block(vec![], thread_index(2, IndexAxis::X, 1)),
                block(vec![], thread_index(3, IndexAxis::LocalX, 2)),
                block(
                    vec![
                        assign(
                            5,
                            Rvalue::Cast {
                                to: Ty::U32,
                                operand: copy(2),
                            },
                        ),
                        Statement::WorkgroupLocalWrite {
                            idx: copy(3),
                            value: copy(5),
                            array: 0,
                        },
                    ],
                    Terminator::Barrier {
                        target: BlockId { index: 3 },
                    },
                ),
                block(
                    vec![
                        assign(4, Rvalue::BinaryOp(BinOp::Add, copy(3), cusize(1))),
                        assign(
                            4,
                            Rvalue::BinaryOp(BinOp::Rem, copy(4), cusize(u64::from(wg))),
                        ),
                        assign(
                            6,
                            Rvalue::WorkgroupLocalRead {
                                idx: copy(4),
                                array: 0,
                            },
                        ),
                        Statement::Assign(element_of(1, 2), Rvalue::Use(copy(6))),
                    ],
                    Terminator::Return,
                ),
            ],
            [wg, 1, 1],
        );
        body.workgroup_locals = vec![WorkgroupLocalDecl {
            elem_ty: Ty::U32,
            len: wg,
        }];
        body
    }

    #[test]
    fn a_barrier_makes_every_lanes_lds_write_visible() {
        let body = neighbour_body(4);
        let mut buffers = [Buffer::from_u32s(&[u32::MAX; 8])];
        run(&body, [2, 1, 1], &mut buffers).unwrap();
        // Global id g reads the value written by the next lane of its own workgroup.
        assert_eq!(u32s(&buffers[0]), vec![1, 2, 3, 0, 5, 6, 7, 4]);
    }

    /// `run` calls `Body::verify` first (card 618): a body where a lane returns while another waits at a
    /// barrier is now rejected statically, by the same `DivergentBarrierReachability` check that catches
    /// `lanes_parked_at_different_barriers_diverge` below, before the interpreter ever sees a lane. The
    /// dynamic `Fault::BarrierDeadlock` this body used to trigger is now unreachable through `run`; it
    /// stays as a documented soundness backstop (see its doc comment).
    #[test]
    fn a_lane_that_returns_before_a_barrier_deadlocks_the_workgroup() {
        // Lane 0 returns; lane 1 waits at a barrier.
        // _1 out, _2 lane
        let body = body_of(
            1,
            vec![buffer_ty(Ty::U32), Ty::Usize],
            vec![
                block(vec![], thread_index(2, IndexAxis::LocalX, 1)),
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 3 })],
                            otherwise: BlockId { index: 2 },
                        },
                    },
                ),
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 3 },
                    },
                ),
                block(vec![], Terminator::Return),
            ],
            [2, 1, 1],
        );
        let mut buffers = [Buffer::from_u32s(&[0])];
        let err = run(&body, [1, 1, 1], &mut buffers).unwrap_err();
        assert!(
            matches!(
                &err,
                InterpError::Invalid(VerifyError {
                    site: Site::Terminator { block },
                    kind: VerifyErrorKind::DivergentBarrierReachability {
                        branch,
                        barrier,
                        reconverge: Some(reconverge),
                    },
                }) if *block == BlockId { index: 1 }
                    && *branch == BlockId { index: 1 }
                    && *barrier == BlockId { index: 2 }
                    && *reconverge == BlockId { index: 3 }
            ),
            "{err}"
        );
    }

    /// `run` calls `Body::verify` first (card 618): lanes parking at different barriers is now rejected
    /// statically, before the interpreter ever runs a lane. The dynamic `Fault::DivergentBarrier` this
    /// body used to trigger is now unreachable through `run`; it stays as a documented soundness backstop
    /// (see its doc comment).
    #[test]
    fn lanes_parked_at_different_barriers_diverge() {
        // Lane 0 would park at the barrier into bb4 (via bb2), lane 1 at the one into bb5 (via bb3).
        let body = body_of(
            1,
            vec![buffer_ty(Ty::U32), Ty::Usize],
            vec![
                block(vec![], thread_index(2, IndexAxis::LocalX, 1)),
                block(
                    vec![],
                    Terminator::SwitchInt {
                        discr: copy(2),
                        targets: SwitchTargets {
                            branches: vec![(0, BlockId { index: 2 })],
                            otherwise: BlockId { index: 3 },
                        },
                    },
                ),
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 4 },
                    },
                ),
                block(
                    vec![],
                    Terminator::Barrier {
                        target: BlockId { index: 5 },
                    },
                ),
                block(vec![], Terminator::Return),
                block(vec![], Terminator::Return),
            ],
            [2, 1, 1],
        );
        let mut buffers = [Buffer::from_u32s(&[0])];
        let err = run(&body, [1, 1, 1], &mut buffers).unwrap_err();
        assert!(
            matches!(
                &err,
                InterpError::Invalid(VerifyError {
                    site: Site::Terminator { block },
                    kind: VerifyErrorKind::DivergentBarrierReachability {
                        branch,
                        barrier,
                        reconverge: None,
                    },
                }) if *block == BlockId { index: 1 }
                    && *branch == BlockId { index: 1 }
                    && *barrier == BlockId { index: 3 }
            ),
            "{err}"
        );
    }

    /// One lane: `out[0] = lhs <op> rhs`, typed `ty`.
    fn eval_binary(op: BinOp, lhs: Constant, rhs: Constant, ty: Ty) -> Result<Scalar, InterpError> {
        let result_ty = if matches!(
            op,
            BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge | BinOp::Eq | BinOp::Ne
        ) {
            Ty::Bool
        } else {
            ty
        };
        let elem = result_ty.clone();
        // _1 out, _2 zero, _3 result
        let body = body_of(
            1,
            vec![buffer_ty(elem), Ty::Usize, result_ty.clone()],
            vec![block(
                vec![
                    assign(2, Rvalue::Use(cusize(0))),
                    assign(
                        3,
                        Rvalue::BinaryOp(op, Operand::Const(lhs), Operand::Const(rhs)),
                    ),
                    Statement::Assign(element_of(1, 2), Rvalue::Use(copy(3))),
                ],
                Terminator::Return,
            )],
            [1, 1, 1],
        );
        let seed = match result_ty {
            Ty::Bool => Scalar::Bool(false),
            Ty::I32 => Scalar::I32(0),
            Ty::U32 => Scalar::U32(0),
            Ty::F32 => Scalar::F32(0.0),
            other => panic!("unsupported test type {other:?}"),
        };
        let mut buffers = [Buffer {
            elem: result_ty,
            data: vec![seed],
        }];
        run(&body, [1, 1, 1], &mut buffers)?;
        Ok(buffers[0].data[0])
    }

    #[test]
    fn shifts_are_arithmetic_on_i32_and_logical_on_u32() {
        let shr_i32 = eval_binary(BinOp::Shr, Constant::I32(-16), Constant::I32(2), Ty::I32);
        assert_eq!(shr_i32.unwrap(), Scalar::I32(-4));
        let shr_u32 = eval_binary(
            BinOp::Shr,
            Constant::U32(0xffff_fff0),
            Constant::U32(2),
            Ty::U32,
        );
        assert_eq!(shr_u32.unwrap(), Scalar::U32(0x3fff_fffc));
        let shl = eval_binary(
            BinOp::Shl,
            Constant::I32(0x4000_0000),
            Constant::I32(1),
            Ty::I32,
        );
        assert_eq!(
            shl.unwrap(),
            Scalar::I32(i32::MIN),
            "shl wraps into the sign bit"
        );
    }

    #[test]
    fn integer_arithmetic_wraps_and_division_by_zero_is_an_error() {
        let add = eval_binary(
            BinOp::Add,
            Constant::I32(i32::MAX),
            Constant::I32(1),
            Ty::I32,
        );
        assert_eq!(add.unwrap(), Scalar::I32(i32::MIN));
        let div = eval_binary(BinOp::Div, Constant::U32(7), Constant::U32(0), Ty::U32).unwrap_err();
        assert!(
            matches!(
                div,
                InterpError::Fault {
                    fault: Fault::DivideByZero,
                    ..
                }
            ),
            "{div}"
        );
        let shift =
            eval_binary(BinOp::Shl, Constant::U32(1), Constant::U32(32), Ty::U32).unwrap_err();
        assert!(
            matches!(
                shift,
                InterpError::Fault {
                    fault: Fault::ShiftOutOfRange {
                        amount: 32,
                        bits: 32
                    },
                    ..
                }
            ),
            "{shift}"
        );
    }

    #[test]
    fn float_comparisons_follow_ieee_for_nan() {
        let nan = Constant::F32(f32::NAN);
        let one = Constant::F32(1.0);
        for (op, want) in [
            (BinOp::Lt, false),
            (BinOp::Ge, false),
            (BinOp::Eq, false),
            (BinOp::Ne, true),
        ] {
            let got = eval_binary(op, nan.clone(), one.clone(), Ty::F32).unwrap();
            assert_eq!(got, Scalar::Bool(want), "NaN {op:?} 1.0");
        }
    }

    #[test]
    fn narrow_floats_round_to_nearest_even() {
        // f16: 1 + 2^-11 is the tie between 1.0 and 1 + 2^-10; ties go to the even mantissa (1.0).
        assert_eq!(round_to_f16(1.0 + 2.0f32.powi(-11)), 1.0);
        assert_eq!(
            round_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)),
            1.0 + 2.0f32.powi(-9)
        );
        assert_eq!(round_to_f16(65504.0), 65504.0);
        assert_eq!(round_to_f16(65520.0), f32::INFINITY);
        assert_eq!(
            round_to_f16(2.0f32.powi(-24)),
            2.0f32.powi(-24),
            "smallest subnormal"
        );
        assert_eq!(
            round_to_f16(2.0f32.powi(-25)),
            0.0,
            "half the smallest subnormal ties to zero"
        );
        assert_eq!(round_to_f16(-0.1), -0.099975586);
        // bf16: 1 + 2^-8 is the tie between 1.0 and 1 + 2^-7.
        assert_eq!(round_to_bf16(1.0 + 2.0f32.powi(-8)), 1.0);
        assert_eq!(
            round_to_bf16(1.0 + 3.0 * 2.0f32.powi(-8)),
            1.0 + 2.0f32.powi(-6)
        );
        assert_eq!(round_to_bf16(3.140625), 3.140625);
    }

    #[test]
    fn e4m3fn_decode_and_encode_agree_on_every_finite_code() {
        for code in 0u32..=0xff {
            if code & 0x7f == 0x7f {
                assert!(fp8_decode(Fp8Format::E4M3Fn, code).is_nan(), "0x{code:02x}");
                continue;
            }
            let value = fp8_decode(Fp8Format::E4M3Fn, code);
            assert_eq!(
                fp8_encode(Fp8Format::E4M3Fn, value),
                code,
                "0x{code:02x} = {value}"
            );
        }
        // Landmarks.
        assert_eq!(fp8_decode(Fp8Format::E4M3Fn, 0x38), 1.0);
        assert_eq!(fp8_decode(Fp8Format::E4M3Fn, 0x7e), 448.0);
        assert_eq!(fp8_decode(Fp8Format::E4M3Fn, 0x01), 1.0 / 512.0);
        assert_eq!(fp8_decode(Fp8Format::E4M3Fn, 0xb8), -1.0);
        // Saturation, NaN, and the tie between 0x38 (1.0) and 0x39 (1.125) going to the even code.
        assert_eq!(fp8_encode(Fp8Format::E4M3Fn, 1.0e9), 0x7e);
        assert_eq!(fp8_encode(Fp8Format::E4M3Fn, f32::INFINITY), 0x7e);
        assert_eq!(fp8_encode(Fp8Format::E4M3Fn, f32::NAN), 0x7f);
        assert_eq!(fp8_encode(Fp8Format::E4M3Fn, 1.0625), 0x38);
    }

    #[test]
    fn casts_follow_rust_as_semantics() {
        assert_eq!(
            cast(&Ty::U32, Scalar::I32(-1)).unwrap(),
            Scalar::U32(u32::MAX)
        );
        assert_eq!(
            cast(&Ty::Usize, Scalar::I32(-1)).unwrap(),
            Scalar::Usize(u64::MAX)
        );
        assert_eq!(cast(&Ty::I32, Scalar::F32(-3.9)).unwrap(), Scalar::I32(-3));
        assert_eq!(
            cast(&Ty::I32, Scalar::F32(f32::NAN)).unwrap(),
            Scalar::I32(0)
        );
        assert_eq!(cast(&Ty::U32, Scalar::F32(-5.0)).unwrap(), Scalar::U32(0));
        assert_eq!(cast(&Ty::F32, Scalar::U32(7)).unwrap(), Scalar::F32(7.0));
        assert_eq!(
            cast(&Ty::F16, Scalar::F32(1.0 / 3.0)).unwrap(),
            Scalar::F16(0.33325195)
        );
        assert_eq!(
            bitcast(&Ty::F32, Scalar::U32(0x3f80_0000)).unwrap(),
            Scalar::F32(1.0)
        );
    }

    #[test]
    fn out_of_bounds_and_uninitialised_reads_are_errors() {
        // _1 out, _2 index, _3 value
        let body = |index: u64| {
            body_of(
                1,
                vec![buffer_ty(Ty::U32), Ty::Usize, Ty::U32],
                vec![block(
                    vec![
                        assign(2, Rvalue::Use(cusize(index))),
                        assign(3, Rvalue::Use(Operand::Copy(element_of(1, 2)))),
                    ],
                    Terminator::Return,
                )],
                [1, 1, 1],
            )
        };
        let mut buffers = [Buffer::from_u32s(&[1, 2])];
        run(&body(1), [1, 1, 1], &mut buffers).unwrap();
        let err = run(&body(2), [1, 1, 1], &mut buffers).unwrap_err();
        assert!(
            matches!(
                err,
                InterpError::Fault {
                    fault: Fault::OutOfBounds {
                        what: "buffer",
                        index: 2,
                        len: 2
                    },
                    ..
                }
            ),
            "{err}"
        );

        // A private array element read before any store.
        // _1 out, _2 zero, _3 array, _4 value
        let unread = body_of(
            1,
            vec![
                buffer_ty(Ty::U32),
                Ty::Usize,
                Ty::Array {
                    elem: Box::new(Ty::U32),
                    len: 2,
                },
                Ty::U32,
            ],
            vec![block(
                vec![
                    assign(2, Rvalue::Use(cusize(0))),
                    assign(
                        4,
                        Rvalue::Use(Operand::Copy(Place {
                            local: l(3),
                            projection: vec![ProjectionElem::Index(l(2))],
                        })),
                    ),
                ],
                Terminator::Return,
            )],
            [1, 1, 1],
        );
        let err = run(&unread, [1, 1, 1], &mut buffers).unwrap_err();
        assert!(
            matches!(
                err,
                InterpError::Fault {
                    fault: Fault::Uninitialized {
                        what: "private array",
                        index: 0
                    },
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn a_body_that_never_terminates_hits_the_step_limit() {
        let body = body_of(
            1,
            vec![buffer_ty(Ty::U32)],
            vec![block(vec![], goto(0))],
            [1, 1, 1],
        );
        let mut buffers = [Buffer::from_u32s(&[0])];
        let err = run(&body, [1, 1, 1], &mut buffers).unwrap_err();
        assert!(
            matches!(
                err,
                InterpError::Fault {
                    fault: Fault::StepLimit,
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn an_invalid_body_or_wrong_buffers_are_refused_before_running() {
        let body = atomic_counter_body(1, AtomicOp::Add);
        let err = run(&body, [1, 1, 1], &mut [Buffer::from_u32s(&[0])]).unwrap_err();
        assert_eq!(
            err,
            InterpError::BufferCount {
                params: 2,
                buffers: 1
            }
        );
        let err = run(
            &body,
            [1, 1, 1],
            &mut [Buffer::from_u32s(&[0]), Buffer::from_f32s(&[0.0])],
        )
        .unwrap_err();
        assert_eq!(
            err,
            InterpError::BufferType {
                param: 2,
                expected: Ty::U32,
                found: Ty::F32
            }
        );
        let mut invalid = atomic_counter_body(1, AtomicOp::Add);
        invalid.blocks.clear();
        let err = run(&invalid, [1, 1, 1], &mut []).unwrap_err();
        assert!(matches!(err, InterpError::Invalid(_)), "{err}");
    }

    #[test]
    fn wmma_is_refused_by_name() {
        let mut body = atomic_counter_body(1, AtomicOp::Add);
        body.blocks[1].statements.push(Statement::WmmaZero {
            dtype: crate::WmmaDtype::F16,
            shape: crate::WmmaShape::M16N16K16,
            dst: l(5),
        });
        let mut buffers = [Buffer::from_u32s(&[0]), Buffer::from_u32s(&[0])];
        let err = run(&body, [1, 1, 1], &mut buffers).unwrap_err();
        assert!(
            matches!(
                err,
                InterpError::Fault {
                    fault: Fault::Unsupported("WmmaZero"),
                    ..
                }
            ),
            "{err}"
        );
    }

    #[test]
    fn vector_load_splat_and_store_are_lanewise() {
        // out[0..4] = load4(input, 0) * splat(2.0)
        // _1 input, _2 out, _3 zero, _4 v, _5 two, _6 r
        let vec4 = Ty::Vec {
            elem: Box::new(Ty::F32),
            lanes: 4,
        };
        let body = body_of(
            2,
            vec![
                buffer_ty(Ty::F32),
                buffer_ty(Ty::F32),
                Ty::Usize,
                vec4.clone(),
                vec4.clone(),
                vec4,
            ],
            vec![block(
                vec![
                    assign(3, Rvalue::Use(cusize(0))),
                    assign(
                        4,
                        Rvalue::VectorLoad {
                            place: element_of(1, 3),
                        },
                    ),
                    assign(5, Rvalue::VectorSplat(Operand::Const(Constant::F32(2.0)))),
                    assign(6, Rvalue::BinaryOp(BinOp::Mul, copy(4), copy(5))),
                    Statement::VectorStore {
                        place: element_of(2, 3),
                        value: copy(6),
                    },
                ],
                Terminator::Return,
            )],
            [1, 1, 1],
        );
        let mut buffers = [
            Buffer::from_f32s(&[1.0, 2.0, 3.0, 4.0]),
            Buffer::from_f32s(&[0.0; 4]),
        ];
        run(&body, [1, 1, 1], &mut buffers).unwrap();
        assert_eq!(buffers[1].to_f32s().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn a_vector_group_index_that_overflows_is_an_out_of_bounds_fault() {
        assert_eq!(vector_range(1, 4, 8), Ok(4..8));
        for index in [2, usize::MAX / 2, usize::MAX] {
            assert!(
                matches!(
                    vector_range(index, 4, 8),
                    Err(Fault::OutOfBounds {
                        what: "buffer",
                        len: 8,
                        ..
                    })
                ),
                "group index {index}"
            );
        }
    }

    #[test]
    fn a_vector_load_at_a_near_usize_max_group_index_is_a_typed_fault() {
        // _1 input, _2 group index, _3 v
        let vec4 = Ty::Vec {
            elem: Box::new(Ty::F32),
            lanes: 4,
        };
        let body = body_of(
            1,
            vec![buffer_ty(Ty::F32), Ty::Usize, vec4],
            vec![block(
                vec![
                    assign(2, Rvalue::Use(cusize(u64::MAX / 2))),
                    assign(
                        3,
                        Rvalue::VectorLoad {
                            place: element_of(1, 2),
                        },
                    ),
                ],
                Terminator::Return,
            )],
            [1, 1, 1],
        );
        let err = run(&body, [1, 1, 1], &mut [Buffer::from_f32s(&[0.0; 8])]).unwrap_err();
        assert!(
            matches!(
                err,
                InterpError::Fault {
                    fault: Fault::OutOfBounds {
                        what: "buffer",
                        len: 8,
                        ..
                    },
                    ..
                }
            ),
            "{err}"
        );
    }
}
