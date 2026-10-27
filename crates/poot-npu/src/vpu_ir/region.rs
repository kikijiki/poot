//! The region data and its OpenVINO IR serializer.

use std::fmt::Write as _;

use super::{VpuElem, VpuOp, VpuScalar, VpuTensor};

impl VpuElem {
    /// The `element_type` attribute spelling.
    const fn element_type(self) -> &'static str {
        match self {
            VpuElem::F16 => "f16",
            VpuElem::F32 => "f32",
            VpuElem::I32 => "i32",
        }
    }

    /// The port `precision` attribute spelling.
    const fn precision(self) -> &'static str {
        match self {
            VpuElem::F16 => "FP16",
            VpuElem::F32 => "FP32",
            VpuElem::I32 => "I32",
        }
    }
}

impl VpuScalar {
    /// The element type the constant has.
    pub const fn elem(self) -> VpuElem {
        match self {
            VpuScalar::F16(_) => VpuElem::F16,
            VpuScalar::F32(_) => VpuElem::F32,
            VpuScalar::I32(_) => VpuElem::I32,
        }
    }

    /// The constant's little-endian bytes, as the weight blob holds them.
    fn le_bytes(self) -> Vec<u8> {
        match self {
            VpuScalar::F16(bits) => bits.to_le_bytes().to_vec(),
            VpuScalar::F32(bits) => bits.to_le_bytes().to_vec(),
            VpuScalar::I32(value) => value.to_le_bytes().to_vec(),
        }
    }
}

impl VpuTensor {
    fn dims(&self) -> String {
        self.shape.iter().fold(String::new(), |mut s, d| {
            let _ = write!(s, "<dim>{d}</dim>");
            s
        })
    }

    fn shape_attr(&self) -> String {
        join_dims(&self.shape)
    }
}

fn join_dims(shape: &[usize]) -> String {
    shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",")
}

/// A value inside a region: a region input or a node's output. Ids are dense, in definition order.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VpuValue(pub u32);

impl VpuOp {
    /// Whether the op is a nullary `Const`: its layer is an auxiliary constant (`aux_<id>`), not a layer
    /// named after its equation, so the compiler is never asked about it.
    const fn is_constant(&self) -> bool {
        matches!(self, VpuOp::Iota { .. } | VpuOp::Scalar(_))
    }

    /// How many operands the op takes; `None` for a variadic op (`Concat`).
    const fn arity(&self) -> Option<usize> {
        match self {
            VpuOp::Iota { .. } | VpuOp::Scalar(_) => Some(0),
            VpuOp::Exp
            | VpuOp::Negative
            | VpuOp::Sqrt
            | VpuOp::ReduceSum { .. }
            | VpuOp::ReduceMax { .. }
            | VpuOp::Reshape
            | VpuOp::Transpose { .. }
            | VpuOp::Broadcast
            | VpuOp::Slice { .. }
            | VpuOp::Convert => Some(1),
            VpuOp::Add
            | VpuOp::Subtract
            | VpuOp::Multiply
            | VpuOp::Divide
            | VpuOp::Maximum
            | VpuOp::GreaterEqual
            | VpuOp::MatMul
            | VpuOp::Gather { .. } => Some(2),
            VpuOp::DynamicUpdateSlice { .. } => Some(3),
            VpuOp::Concat { .. } => None,
        }
    }
}

/// One lowered equation: an op over earlier values, defining one new value.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VpuNode {
    /// The id of the poot equation this node lowers (its output value's index).
    pub origin: u32,
    /// The op.
    pub op: VpuOp,
    /// The operands, defined earlier in the region.
    pub inputs: Vec<VpuValue>,
    /// The type of the value the node defines.
    pub out: VpuTensor,
}

impl VpuNode {
    /// The IR layer name the node's value goes by: the equation's (`v<origin>`).
    fn layer_name(&self) -> String {
        format!("v{}", self.origin)
    }
}

/// A value a region reads from outside: a runtime argument the device binds (weights, activations, state).
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VpuInput {
    /// The poot value's name.
    pub name: String,
    /// Its type.
    pub tensor: VpuTensor,
}

/// A region: the equations one vendor-compiled graph covers, in program order.
///
/// Values are numbered densely: the inputs first, then each node's output in node order.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VpuRegion {
    /// A name for the model.
    pub name: String,
    /// The runtime arguments, in argument order.
    pub inputs: Vec<VpuInput>,
    /// The equations, in program order.
    pub nodes: Vec<VpuNode>,
    /// The values the region returns, in result order.
    pub outputs: Vec<VpuValue>,
}

impl VpuRegion {
    /// The IR layer name of each argument, in argument order: the names the compiled graph reports.
    pub fn input_names(&self) -> impl Iterator<Item = String> + '_ {
        self.inputs.iter().map(|input| input.name.clone())
    }

    /// The extents of each argument, in argument order.
    pub fn input_shapes(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        self.inputs.iter().map(|input| input.tensor.shape.clone())
    }

    /// The IR layer name and poot equation of every node, in program order. A constant node (a scalar literal
    /// or an iota) is an auxiliary `Const` like an axis or a shape: it has no layer of its own to report.
    pub fn layers(&self) -> impl Iterator<Item = (String, u32)> + '_ {
        self.nodes
            .iter()
            .filter(|node| !node.op.is_constant())
            .map(|node| (node.layer_name(), node.origin))
    }

    /// The node an output value is defined by. Outputs are node values: the emitter numbers inputs first.
    fn output_node(&self, value: VpuValue) -> Option<&VpuNode> {
        self.nodes
            .get((value.0 as usize).checked_sub(self.inputs.len())?)
    }

    /// The IR layer name of each result, in result order: the name of the layer that defines it, which is
    /// what the compiled graph reports for the result.
    pub fn output_names(&self) -> impl Iterator<Item = String> + '_ {
        self.outputs.iter().map(|&value| {
            self.output_node(value)
                .map_or_else(String::new, VpuNode::layer_name)
        })
    }

    /// The extents of each result, in result order.
    pub fn output_shapes(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        self.outputs.iter().map(|&value| {
            self.output_node(value)
                .map_or_else(Vec::new, |node| node.out.shape.clone())
        })
    }
}

/// A region the emitter cannot serialize because it is not well formed.
#[derive(Clone, PartialEq, Eq, Debug, thiserror::Error)]
pub enum VpuEmitError {
    /// A node reads a value that is not defined before it.
    #[error("node for equation v{origin} reads value {value} before it is defined")]
    UndefinedOperand {
        /// The node's equation id.
        origin: u32,
        /// The undefined value's index.
        value: u32,
    },
    /// A node has the wrong number of operands for its op.
    #[error("node for equation v{origin}: {op:?} takes {expected} operands, got {actual}")]
    OperandCount {
        /// The node's equation id.
        origin: u32,
        /// The op.
        op: VpuOp,
        /// Operands the op takes.
        expected: usize,
        /// Operands the node has.
        actual: usize,
    },
    /// A result names a value the region does not define.
    #[error("result {value} is not defined by the region")]
    UndefinedResult {
        /// The undefined value's index.
        value: u32,
    },
    /// The region has no result.
    #[error("a region with no result computes nothing")]
    NoResult,
    /// An op's parameter is outside the operand's shape (an axis, a permutation or a slice range).
    #[error("node for equation v{origin}: {op:?} does not fit its operand of shape {shape:?}")]
    ParameterOutOfRange {
        /// The node's equation id.
        origin: u32,
        /// The op.
        op: VpuOp,
        /// The operand's shape.
        shape: Vec<usize>,
    },
}

/// The serialized region: the OpenVINO IR text and the constant data it references.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct VpuIr {
    /// The IR `.xml` text.
    pub xml: String,
    /// The IR `.bin` contents: the auxiliary constants (axes, permutations, shapes) the text points into.
    pub weights: Vec<u8>,
}

/// A port a layer reads: the producing layer, its output port and the type that flows.
#[derive(Clone)]
struct Port {
    layer: usize,
    port: usize,
    precision: &'static str,
    dims: String,
}

/// The port a layer defines: its shape from `tensor`, its precision as the IR spells it (a comparison's
/// boolean is a precision no region element type has).
struct Output<'a> {
    precision: &'static str,
    tensor: &'a VpuTensor,
}

struct Emitter {
    layers: String,
    edges: String,
    weights: Vec<u8>,
    next_layer: usize,
    ports: Vec<Port>,
    types: Vec<VpuTensor>,
}

impl Emitter {
    fn layer_id(&mut self) -> usize {
        let id = self.next_layer;
        self.next_layer += 1;
        id
    }

    fn edge(&mut self, from: &Port, to_layer: usize, to_port: usize) {
        let _ = write!(
            self.edges,
            r#"<edge from-layer="{}" from-port="{}" to-layer="{to_layer}" to-port="{to_port}"/>"#,
            from.layer, from.port
        );
    }

    /// A `Const` layer holding `bytes`, of the given IR element type and port precision, in the weight blob.
    fn const_layer(
        &mut self,
        element_type: &str,
        precision: &'static str,
        shape: &[usize],
        bytes: &[u8],
    ) -> Port {
        let offset = self.weights.len();
        self.weights.extend_from_slice(bytes);
        let id = self.layer_id();
        let dims = shape.iter().fold(String::new(), |mut s, d| {
            let _ = write!(s, "<dim>{d}</dim>");
            s
        });
        let _ = write!(
            self.layers,
            r#"<layer id="{id}" name="aux_{id}" type="Const" version="opset1"><data element_type="{element_type}" shape="{shape}" offset="{offset}" size="{size}"/><output><port id="0" precision="{precision}">{dims}</port></output></layer>"#,
            shape = join_dims(shape),
            size = bytes.len(),
        );
        Port {
            layer: id,
            port: 0,
            precision,
            dims,
        }
    }

    /// A 1-D (or rank-0 for `scalar`) `i64` constant for an op's second input.
    fn i64_const(&mut self, values: &[i64], scalar: bool) -> Port {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let shape = if scalar { vec![] } else { vec![values.len()] };
        self.const_layer("i64", "I64", &shape, &bytes)
    }

    fn i64_vec(&mut self, values: &[usize]) -> Port {
        let values: Vec<i64> = values.iter().map(|&v| v as i64).collect();
        self.i64_const(&values, false)
    }

    /// Writes one op layer with `inputs` (precision per port from the producing port; `aux` operands are
    /// `i64` constants) and one output of `out`'s type. Returns its output port.
    fn op_layer(
        &mut self,
        name: &str,
        kind: &str,
        version: &str,
        data: &str,
        inputs: &[(Port, &'static str)],
        out: &VpuTensor,
    ) -> Port {
        let result = Output {
            precision: out.elem.precision(),
            tensor: out,
        };
        self.op_layer_as(name, kind, version, data, inputs, &result)
    }

    /// [`Emitter::op_layer`] with an output port precision other than `out`'s element type's: the boolean
    /// result of a comparison, which no region value carries.
    fn op_layer_as(
        &mut self,
        name: &str,
        kind: &str,
        version: &str,
        data: &str,
        inputs: &[(Port, &'static str)],
        out: &Output<'_>,
    ) -> Port {
        let id = self.layer_id();
        let mut in_ports = String::new();
        for (index, (port, precision)) in inputs.iter().enumerate() {
            let _ = write!(
                in_ports,
                r#"<port id="{index}" precision="{precision}">{}</port>"#,
                port.dims
            );
        }
        let out_port = inputs.len();
        let input = if inputs.is_empty() {
            String::new()
        } else {
            format!("<input>{in_ports}</input>")
        };
        let _ = write!(
            self.layers,
            r#"<layer id="{id}" name="{name}" type="{kind}" version="{version}">{data}{input}<output><port id="{out_port}" precision="{p}" names="{name}">{dims}</port></output></layer>"#,
            p = out.precision,
            dims = out.tensor.dims(),
        );
        for (index, (port, _)) in inputs.iter().enumerate() {
            self.edge(port, id, index);
        }
        Port {
            layer: id,
            port: out_port,
            precision: out.precision,
            dims: out.tensor.dims(),
        }
    }
}

/// Serializes `region` to OpenVINO IR (version 11).
///
/// # Errors
///
/// [`VpuEmitError`] when the region is not well formed: an operand used before it is defined, a wrong
/// operand count, an out-of-range op parameter, or no result.
pub fn emit_vpu(region: &VpuRegion) -> Result<VpuIr, VpuEmitError> {
    if region.outputs.is_empty() {
        return Err(VpuEmitError::NoResult);
    }
    let mut e = Emitter {
        layers: String::new(),
        edges: String::new(),
        weights: Vec::new(),
        next_layer: 0,
        ports: Vec::new(),
        types: Vec::new(),
    };
    for input in &region.inputs {
        let id = e.layer_id();
        let t = &input.tensor;
        let _ = write!(
            e.layers,
            r#"<layer id="{id}" name="{name}" type="Parameter" version="opset1"><data shape="{shape}" element_type="{et}"/><output><port id="0" precision="{p}" names="{name}">{dims}</port></output></layer>"#,
            name = input.name,
            shape = t.shape_attr(),
            et = t.elem.element_type(),
            p = t.elem.precision(),
            dims = t.dims(),
        );
        e.ports.push(Port {
            layer: id,
            port: 0,
            precision: t.elem.precision(),
            dims: t.dims(),
        });
        e.types.push(t.clone());
    }
    for node in &region.nodes {
        lower_node(&mut e, node)?;
    }
    for (index, value) in region.outputs.iter().enumerate() {
        let port = e
            .ports
            .get(value.0 as usize)
            .cloned()
            .ok_or(VpuEmitError::UndefinedResult { value: value.0 })?;
        let id = e.layer_id();
        let _ = write!(
            e.layers,
            r#"<layer id="{id}" name="result_{index}" type="Result" version="opset1"><input><port id="0" precision="{p}">{dims}</port></input></layer>"#,
            p = port.precision,
            dims = port.dims,
        );
        e.edge(&port, id, 0);
    }
    let xml = format!(
        r#"<?xml version="1.0"?><net name="{name}" version="11"><layers>{layers}</layers><edges>{edges}</edges></net>"#,
        name = region.name,
        layers = e.layers,
        edges = e.edges,
    );
    Ok(VpuIr {
        xml,
        weights: e.weights,
    })
}

fn lower_node(e: &mut Emitter, node: &VpuNode) -> Result<(), VpuEmitError> {
    let origin = node.origin;
    if let Some(expected) = node.op.arity()
        && node.inputs.len() != expected
    {
        return Err(VpuEmitError::OperandCount {
            origin,
            op: node.op.clone(),
            expected,
            actual: node.inputs.len(),
        });
    }
    let mut operands = Vec::with_capacity(node.inputs.len());
    let mut shapes = Vec::with_capacity(node.inputs.len());
    let mut elems = Vec::with_capacity(node.inputs.len());
    for value in &node.inputs {
        let index = value.0 as usize;
        let (Some(port), Some(ty)) = (e.ports.get(index), e.types.get(index)) else {
            return Err(VpuEmitError::UndefinedOperand {
                origin,
                value: value.0,
            });
        };
        operands.push(port.clone());
        shapes.push(ty.shape.clone());
        elems.push(ty.elem);
    }
    let out_of_range = || VpuEmitError::ParameterOutOfRange {
        origin,
        op: node.op.clone(),
        shape: shapes.first().cloned().unwrap_or_default(),
    };
    let name = node.layer_name();
    let prec = |p: &Port| p.precision;
    let with = |operands: &[Port]| -> Vec<(Port, &'static str)> {
        operands.iter().map(|p| (p.clone(), prec(p))).collect()
    };
    let binary = |e: &mut Emitter, name: &str, kind: &str| {
        e.op_layer(
            name,
            kind,
            "opset1",
            r#"<data auto_broadcast="numpy"/>"#,
            &with(&operands),
            &node.out,
        )
    };
    let out = match &node.op {
        VpuOp::Add => binary(e, &name, "Add"),
        VpuOp::Subtract => binary(e, &name, "Subtract"),
        VpuOp::Multiply => binary(e, &name, "Multiply"),
        VpuOp::Divide => binary(e, &name, "Divide"),
        VpuOp::Maximum => binary(e, &name, "Maximum"),
        VpuOp::Exp => e.op_layer(&name, "Exp", "opset1", "", &with(&operands), &node.out),
        VpuOp::Negative => e.op_layer(&name, "Negative", "opset1", "", &with(&operands), &node.out),
        VpuOp::Sqrt => e.op_layer(&name, "Sqrt", "opset1", "", &with(&operands), &node.out),
        VpuOp::GreaterEqual => {
            // The comparison is boolean; poot's `Ge` is the operands' element type holding 1 or 0.
            let flags = e.op_layer_as(
                &format!("{name}_flags"),
                "GreaterEqual",
                "opset1",
                r#"<data auto_broadcast="numpy"/>"#,
                &with(&operands),
                &Output {
                    precision: "BOOL",
                    tensor: &node.out,
                },
            );
            e.op_layer(
                &name,
                "Convert",
                "opset1",
                &format!(
                    r#"<data destination_type="{}"/>"#,
                    node.out.elem.element_type()
                ),
                &[(flags, "BOOL")],
                &node.out,
            )
        }
        VpuOp::Scalar(scalar) => {
            let elem = scalar.elem();
            e.const_layer(
                elem.element_type(),
                elem.precision(),
                &node.out.shape,
                &scalar.le_bytes(),
            )
        }
        VpuOp::Iota { len } => {
            let bytes: Vec<u8> = match node.out.elem {
                VpuElem::I32 => (0..*len as i32).flat_map(i32::to_le_bytes).collect(),
                VpuElem::F32 => (0..*len).flat_map(|i| (i as f32).to_le_bytes()).collect(),
                VpuElem::F16 => return Err(out_of_range()),
            };
            e.const_layer(
                node.out.elem.element_type(),
                node.out.elem.precision(),
                &node.out.shape,
                &bytes,
            )
        }
        VpuOp::MatMul => e.op_layer(
            &name,
            "MatMul",
            "opset1",
            r#"<data transpose_a="false" transpose_b="false"/>"#,
            &with(&operands),
            &node.out,
        ),
        VpuOp::Convert => e.op_layer(
            &name,
            "Convert",
            "opset1",
            &format!(
                r#"<data destination_type="{}"/>"#,
                node.out.elem.element_type()
            ),
            &with(&operands),
            &node.out,
        ),
        VpuOp::Concat { axis } => {
            if shapes.iter().any(|s| *axis >= s.len()) {
                return Err(out_of_range());
            }
            e.op_layer(
                &name,
                "Concat",
                "opset1",
                &format!(r#"<data axis="{axis}"/>"#),
                &with(&operands),
                &node.out,
            )
        }
        VpuOp::ReduceSum { axis, keepdim } | VpuOp::ReduceMax { axis, keepdim } => {
            if *axis >= shapes[0].len() {
                return Err(out_of_range());
            }
            let kind = if matches!(node.op, VpuOp::ReduceSum { .. }) {
                "ReduceSum"
            } else {
                "ReduceMax"
            };
            let axes = e.i64_vec(&[*axis]);
            let mut ins = with(&operands);
            ins.push((axes, "I64"));
            e.op_layer(
                &name,
                kind,
                "opset1",
                &format!(r#"<data keep_dims="{keepdim}"/>"#),
                &ins,
                &node.out,
            )
        }
        VpuOp::Reshape => {
            let shape = e.i64_vec(&node.out.shape);
            let mut ins = with(&operands);
            ins.push((shape, "I64"));
            e.op_layer(
                &name,
                "Reshape",
                "opset1",
                r#"<data special_zero="false"/>"#,
                &ins,
                &node.out,
            )
        }
        VpuOp::Broadcast => {
            let shape = e.i64_vec(&node.out.shape);
            let mut ins = with(&operands);
            ins.push((shape, "I64"));
            e.op_layer(
                &name,
                "Broadcast",
                "opset3",
                r#"<data mode="numpy"/>"#,
                &ins,
                &node.out,
            )
        }
        VpuOp::Transpose { perm } => {
            let mut sorted = perm.clone();
            sorted.sort_unstable();
            if sorted.iter().enumerate().any(|(i, &p)| i != p) || perm.len() != shapes[0].len() {
                return Err(out_of_range());
            }
            let order = e.i64_vec(perm);
            let mut ins = with(&operands);
            ins.push((order, "I64"));
            e.op_layer(&name, "Transpose", "opset1", "", &ins, &node.out)
        }
        VpuOp::Slice { axis, start, end } => {
            if *axis >= shapes[0].len() || start > end || *end > shapes[0][*axis] {
                return Err(out_of_range());
            }
            let mut ins = with(&operands);
            for aux in [
                e.i64_vec(&[*start]),
                e.i64_vec(&[*end]),
                e.i64_vec(&[1]),
                e.i64_vec(&[*axis]),
            ] {
                ins.push((aux, "I64"));
            }
            e.op_layer(&name, "Slice", "opset8", "", &ins, &node.out)
        }
        VpuOp::DynamicUpdateSlice { axis } => {
            let fits = *axis < shapes[0].len()
                && shapes[1].len() == shapes[0].len()
                && shapes[1][*axis] <= shapes[0][*axis]
                && shapes[2].is_empty();
            if !fits {
                return Err(out_of_range());
            }
            lower_update_slice(e, node, &name, &operands, &shapes, *axis)
        }
        VpuOp::Gather { axis } => {
            if *axis >= shapes[0].len() {
                return Err(out_of_range());
            }
            lower_gather(e, node, &name, &operands, &shapes, elems[1], *axis)
        }
    };
    e.ports.push(out);
    e.types.push(node.out.clone());
    Ok(())
}

/// `Gather`. The NPU faults on a rank-0 index (a rank-1 index is fine; measured on Lunar Lake, POOT-040),
/// so a scalar-index gather is `Reshape(index -> [1]) -> Gather -> Reshape(-> the equation's output
/// shape)`, which computes the same value.
fn lower_gather(
    e: &mut Emitter,
    node: &VpuNode,
    name: &str,
    operands: &[Port],
    shapes: &[Vec<usize>],
    index_elem: VpuElem,
    axis: usize,
) -> Port {
    let axis_const = e.i64_const(&[axis as i64], true);
    let index_rank = shapes[1].len();
    if index_rank > 0 {
        let ins = [
            (operands[0].clone(), operands[0].precision),
            (operands[1].clone(), operands[1].precision),
            (axis_const, "I64"),
        ];
        return e.op_layer(
            name,
            "Gather",
            "opset8",
            r#"<data batch_dims="0"/>"#,
            &ins,
            &node.out,
        );
    }
    let index_1d = VpuTensor {
        elem: index_elem,
        shape: vec![1],
    };
    let one = e.i64_vec(&[1]);
    let index = e.op_layer(
        &format!("{name}_index"),
        "Reshape",
        "opset1",
        r#"<data special_zero="false"/>"#,
        &[(operands[1].clone(), operands[1].precision), (one, "I64")],
        &index_1d,
    );
    let mut kept = shapes[0].clone();
    kept[axis] = 1;
    let gathered_ty = VpuTensor {
        elem: node.out.elem,
        shape: kept,
    };
    let gathered = e.op_layer(
        &format!("{name}_rows"),
        "Gather",
        "opset8",
        r#"<data batch_dims="0"/>"#,
        &[
            (operands[0].clone(), operands[0].precision),
            (index, operands[1].precision),
            (axis_const, "I64"),
        ],
        &gathered_ty,
    );
    let shape = e.i64_vec(&node.out.shape);
    e.op_layer(
        name,
        "Reshape",
        "opset1",
        r#"<data special_zero="false"/>"#,
        &[(gathered, node.out.elem.precision()), (shape, "I64")],
        &node.out,
    )
}

/// `DynamicUpdateSlice` as a `ScatterUpdate` of the update's rows at the indices `index..index + extent`:
/// the indices are the constant range `0..extent` plus the scalar index, so a literal index is a constant
/// the vendor compiler folds and a runtime index is read from its value.
fn lower_update_slice(
    e: &mut Emitter,
    node: &VpuNode,
    name: &str,
    operands: &[Port],
    shapes: &[Vec<usize>],
    axis: usize,
) -> Port {
    let extent = shapes[1][axis];
    let range: Vec<u8> = (0..extent as i32).flat_map(i32::to_le_bytes).collect();
    let range = e.const_layer("i32", "I32", &[extent], &range);
    let indices_ty = VpuTensor {
        elem: VpuElem::I32,
        shape: vec![extent],
    };
    let indices = e.op_layer(
        &format!("{name}_indices"),
        "Add",
        "opset1",
        r#"<data auto_broadcast="numpy"/>"#,
        &[(range, "I32"), (operands[2].clone(), operands[2].precision)],
        &indices_ty,
    );
    let axis_const = e.i64_const(&[axis as i64], true);
    e.op_layer(
        name,
        "ScatterUpdate",
        "opset3",
        "",
        &[
            (operands[0].clone(), operands[0].precision),
            (indices, "I32"),
            (operands[1].clone(), operands[1].precision),
            (axis_const, "I64"),
        ],
        &node.out,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(elem: VpuElem, shape: &[usize]) -> VpuTensor {
        VpuTensor {
            elem,
            shape: shape.to_vec(),
        }
    }

    fn region(nodes: Vec<VpuNode>, outputs: Vec<VpuValue>) -> VpuRegion {
        VpuRegion {
            name: "r".into(),
            inputs: vec![
                VpuInput {
                    name: "a".into(),
                    tensor: tensor(VpuElem::F16, &[2, 3]),
                },
                VpuInput {
                    name: "b".into(),
                    tensor: tensor(VpuElem::F16, &[3, 4]),
                },
            ],
            nodes,
            outputs,
        }
    }

    fn node(op: VpuOp, inputs: &[u32], out: VpuTensor) -> VpuNode {
        VpuNode {
            origin: 9,
            op,
            inputs: inputs.iter().map(|&i| VpuValue(i)).collect(),
            out,
        }
    }

    /// Every op emits its one opset layer type with the version and attributes the op takes.
    /// Mutation: emit `MatMul` with `transpose_b="true"`; this row fails on the attribute.
    #[test]
    fn each_op_emits_its_opset_layer() {
        let cases = [
            (
                node(VpuOp::MatMul, &[0, 1], tensor(VpuElem::F16, &[2, 4])),
                r#"type="MatMul" version="opset1"><data transpose_a="false" transpose_b="false"/>"#,
            ),
            (
                node(VpuOp::Exp, &[0], tensor(VpuElem::F16, &[2, 3])),
                r#"type="Exp" version="opset1">"#,
            ),
            (
                node(VpuOp::Broadcast, &[0], tensor(VpuElem::F16, &[5, 2, 3])),
                r#"type="Broadcast" version="opset3"><data mode="numpy"/>"#,
            ),
            (
                node(
                    VpuOp::Concat { axis: 1 },
                    &[0, 0],
                    tensor(VpuElem::F16, &[2, 6]),
                ),
                r#"type="Concat" version="opset1"><data axis="1"/>"#,
            ),
            (
                node(VpuOp::Convert, &[0], tensor(VpuElem::F32, &[2, 3])),
                r#"type="Convert" version="opset1"><data destination_type="f32"/>"#,
            ),
            (
                node(VpuOp::Subtract, &[0, 0], tensor(VpuElem::F16, &[2, 3])),
                r#"type="Subtract" version="opset1"><data auto_broadcast="numpy"/>"#,
            ),
            (
                node(VpuOp::Divide, &[0, 0], tensor(VpuElem::F16, &[2, 3])),
                r#"type="Divide" version="opset1"><data auto_broadcast="numpy"/>"#,
            ),
            (
                node(VpuOp::Maximum, &[0, 0], tensor(VpuElem::F16, &[2, 3])),
                r#"type="Maximum" version="opset1"><data auto_broadcast="numpy"/>"#,
            ),
            (
                node(VpuOp::Negative, &[0], tensor(VpuElem::F16, &[2, 3])),
                r#"type="Negative" version="opset1">"#,
            ),
            (
                node(VpuOp::Sqrt, &[0], tensor(VpuElem::F16, &[2, 3])),
                r#"type="Sqrt" version="opset1">"#,
            ),
            (
                node(
                    VpuOp::ReduceMax {
                        axis: 1,
                        keepdim: false,
                    },
                    &[0],
                    tensor(VpuElem::F16, &[2]),
                ),
                r#"type="ReduceMax" version="opset1"><data keep_dims="false"/>"#,
            ),
            (
                node(VpuOp::Reshape, &[0], tensor(VpuElem::F16, &[6])),
                r#"type="Reshape" version="opset1"><data special_zero="false"/>"#,
            ),
        ];
        for (node, fragment) in cases {
            let ir = emit_vpu(&region(vec![node], vec![VpuValue(2)])).unwrap();
            assert!(ir.xml.contains(fragment), "{fragment} not in {}", ir.xml);
        }
    }

    /// A malformed region is a typed error, not text the vendor compiler then rejects.
    /// Mutation: drop the defined-before-use check in `lower_node`; the first row indexes an absent port.
    #[test]
    fn a_malformed_region_is_a_typed_error() {
        let add = |inputs: &[u32]| node(VpuOp::Add, inputs, tensor(VpuElem::F16, &[2, 3]));
        let result = |nodes, outputs| emit_vpu(&region(nodes, outputs)).unwrap_err();
        assert_eq!(
            result(vec![add(&[0, 7])], vec![VpuValue(2)]),
            VpuEmitError::UndefinedOperand {
                origin: 9,
                value: 7
            }
        );
        assert_eq!(
            result(vec![add(&[0])], vec![VpuValue(2)]),
            VpuEmitError::OperandCount {
                origin: 9,
                op: VpuOp::Add,
                expected: 2,
                actual: 1
            }
        );
        assert_eq!(
            result(vec![add(&[0, 0])], vec![VpuValue(5)]),
            VpuEmitError::UndefinedResult { value: 5 }
        );
        assert_eq!(result(vec![add(&[0, 0])], vec![]), VpuEmitError::NoResult);
    }

    /// An axis, permutation or slice range outside the operand's shape is refused before any text exists.
    #[test]
    fn op_parameters_must_fit_the_operand() {
        let out = tensor(VpuElem::F16, &[2, 3]);
        for op in [
            VpuOp::ReduceSum {
                axis: 2,
                keepdim: false,
            },
            VpuOp::Transpose { perm: vec![0, 0] },
            VpuOp::Transpose { perm: vec![1] },
            VpuOp::Slice {
                axis: 0,
                start: 1,
                end: 3,
            },
            VpuOp::Slice {
                axis: 0,
                start: 2,
                end: 1,
            },
            VpuOp::Concat { axis: 2 },
            VpuOp::ReduceMax {
                axis: 2,
                keepdim: true,
            },
            VpuOp::DynamicUpdateSlice { axis: 2 },
        ] {
            let operands = vec![0; op.arity().unwrap_or(2)];
            let error = emit_vpu(&region(
                vec![node(op.clone(), &operands, out.clone())],
                vec![VpuValue(2)],
            ));
            assert!(
                matches!(error, Err(VpuEmitError::ParameterOutOfRange { .. })),
                "{op:?}: {error:?}"
            );
        }
    }

    /// Every layer name `layers()` reports is a layer of the emitted text, for a region holding a scalar
    /// literal, an iota and a `Ge` (the vendor compiler is asked about exactly these names).
    /// Mutation: filter only `Scalar` nodes in `layers()`; the iota's `v9` is reported but never emitted.
    #[test]
    fn every_reported_layer_is_in_the_emitted_text() {
        let r = region(
            vec![
                node(VpuOp::Iota { len: 3 }, &[], tensor(VpuElem::I32, &[3])),
                node(
                    VpuOp::Scalar(VpuScalar::F16(0x3c00)),
                    &[],
                    tensor(VpuElem::F16, &[]),
                ),
                node(VpuOp::GreaterEqual, &[0, 3], tensor(VpuElem::F16, &[2, 3])),
            ],
            vec![VpuValue(4)],
        );
        let xml = emit_vpu(&r).unwrap().xml;
        let reported: Vec<_> = r.layers().collect();
        assert_eq!(reported.len(), 1, "only the Ge: {reported:?}");
        for (name, _) in reported {
            assert!(
                xml.contains(&format!(r#"name="{name}""#)),
                "{name} in {xml}"
            );
        }
    }

    /// The same region always yields the same text and blob: the vendor blob is cached under it.
    #[test]
    fn emission_is_deterministic() {
        let r = region(
            vec![node(VpuOp::Exp, &[0], tensor(VpuElem::F16, &[2, 3]))],
            vec![VpuValue(2)],
        );
        assert_eq!(emit_vpu(&r), emit_vpu(&r));
    }
}
