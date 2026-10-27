//! The data-movement generator over every storage, on the KIR interpreter (tier 1, no device): each op
//! against an independent coordinate-walking reference, for 32-bit elements (f32 and i32 carriers) and for
//! 8- and 16-bit packed rows, at row widths that do and do not fill a word.
//!
//! The reference states the index mappings by walking multi-dimensional coordinates, where the generator
//! works in flat strides, so a stride bug in the generator cannot be mirrored in the oracle. Every
//! comparison is on logical lane bits: a value is a `u32` holding the lane (the whole word for 32-bit
//! storage), padding lanes of a packed input are filled with a nonzero pattern and padding lanes of the
//! output start nonzero, so a movement that read or required padding fails.

use poot_kernel_ir::Ty;
use poot_kernel_ir::interp::{Buffer, run};
use poot_kernel_ir::test_support::workgroups_covering;
use poot_kernelgen::{KernelGenError, MovementOp, MovementSpec, Storage, UnitWindow, UpdateIndex};

/// A row-major tensor of lane values.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Tensor {
    shape: Vec<usize>,
    lanes: Vec<u32>,
}

impl Tensor {
    fn numel(shape: &[usize]) -> usize {
        shape.iter().product()
    }

    /// Lane values `seed, seed + 1, ...` folded into `width` bits, so no two neighbours are equal.
    fn counting(shape: &[usize], seed: u32, width: u32) -> Self {
        let mask = if width == 32 {
            u32::MAX
        } else {
            (1 << width) - 1
        };
        let lanes = (0..Self::numel(shape) as u32)
            .map(|i| (seed.wrapping_add(i).wrapping_mul(0x9e37_79b1) >> 3 ^ i) & mask)
            .collect();
        Self {
            shape: shape.to_vec(),
            lanes,
        }
    }

    fn at(&self, coord: &[usize]) -> u32 {
        let mut flat = 0;
        for (&c, &extent) in coord.iter().zip(&self.shape) {
            assert!(c < extent, "coordinate {coord:?} outside {:?}", self.shape);
            flat = flat * extent + c;
        }
        self.lanes[flat]
    }
}

/// Every coordinate of `shape` in row-major order.
fn coords(shape: &[usize]) -> Vec<Vec<usize>> {
    let mut all = vec![vec![]];
    for &extent in shape {
        all = all
            .into_iter()
            .flat_map(|prefix| {
                (0..extent).map(move |c| {
                    let mut next = prefix.clone();
                    next.push(c);
                    next
                })
            })
            .collect();
    }
    all
}

fn build(shape: &[usize], mut value: impl FnMut(&[usize]) -> u32) -> Tensor {
    Tensor {
        shape: shape.to_vec(),
        lanes: coords(shape).iter().map(|c| value(c)).collect(),
    }
}

/// The three storages the bank covers, with how a tensor crosses into and out of an interpreter buffer.
#[derive(Clone, Copy, Debug)]
enum Kind {
    F32,
    I32,
    Packed(u32),
}

impl Kind {
    /// The kind whose storage is `storage`.
    fn of(storage: &Storage) -> Kind {
        match storage {
            Storage::Elements { carrier: Ty::F32 } => Kind::F32,
            Storage::Elements { carrier: Ty::I32 } => Kind::I32,
            Storage::PackedRows { lane_bits } => Kind::Packed(*lane_bits),
            Storage::Elements { carrier } => panic!("the bank has no {carrier:?} carrier"),
        }
    }

    fn storage(self) -> Storage {
        match self {
            Kind::F32 => Storage::Elements { carrier: Ty::F32 },
            Kind::I32 => Storage::Elements { carrier: Ty::I32 },
            Kind::Packed(lane_bits) => Storage::PackedRows { lane_bits },
        }
    }

    fn width(self) -> u32 {
        match self {
            Kind::F32 | Kind::I32 => 32,
            Kind::Packed(lane_bits) => lane_bits,
        }
    }

    /// The words `tensor`'s buffer holds in this storage: one word per 32-bit element, or the packed
    /// rows, each padded to a whole word with all-ones lanes.
    fn words(self, tensor: &Tensor) -> Vec<u32> {
        match self {
            Kind::F32 | Kind::I32 => tensor.lanes.clone(),
            Kind::Packed(lane_bits) => {
                let lanes = (32 / lane_bits) as usize;
                let row_len = tensor.shape.last().copied().unwrap_or(1);
                let mask = (1u32 << lane_bits) - 1;
                let mut words = Vec::new();
                for row in tensor.lanes.chunks(row_len) {
                    for word in row.chunks(lanes) {
                        let mut value = 0u32;
                        for lane in 0..lanes {
                            let lane_value = word.get(lane).copied().unwrap_or(u32::MAX);
                            value |= (lane_value & mask) << (lane as u32 * lane_bits);
                        }
                        words.push(value);
                    }
                }
                words
            }
        }
    }

    /// An output buffer of `shape` whose every bit starts set, so an unwritten lane or word shows.
    fn blank(self, shape: &[usize]) -> Vec<u32> {
        vec![u32::MAX; self.storage().units(shape)]
    }

    /// The logical lanes of `words` as a tensor of `shape`; padding lanes are not read.
    fn tensor(self, words: &[u32], shape: &[usize]) -> Tensor {
        let lanes = match self {
            Kind::F32 | Kind::I32 => words.to_vec(),
            Kind::Packed(lane_bits) => {
                let lane_count = (32 / lane_bits) as usize;
                let row_len = shape.last().copied().unwrap_or(1);
                let words_per_row = row_len.div_ceil(lane_count);
                let mask = (1u32 << lane_bits) - 1;
                let mut lanes = Vec::new();
                for row in 0..Tensor::numel(shape) / row_len {
                    for col in 0..row_len {
                        let word = words[row * words_per_row + col / lane_count];
                        lanes.push((word >> ((col % lane_count) as u32 * lane_bits)) & mask);
                    }
                }
                lanes
            }
        };
        Tensor {
            shape: shape.to_vec(),
            lanes,
        }
    }

    /// `words` as the interpreter buffer of this storage.
    fn interp_buffer(self, words: &[u32]) -> Buffer {
        match self {
            Kind::F32 => {
                Buffer::from_f32s(&words.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>())
            }
            Kind::I32 => Buffer::from_i32s(&words.iter().map(|&w| w as i32).collect::<Vec<_>>()),
            Kind::Packed(_) => Buffer::from_u32s(words),
        }
    }

    /// The words an interpreter buffer of this storage holds.
    fn interp_words(self, buffer: &Buffer) -> Vec<u32> {
        match self {
            Kind::F32 => buffer
                .to_f32s()
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect(),
            Kind::I32 => buffer
                .to_i32s()
                .unwrap()
                .iter()
                .map(|&v| v as u32)
                .collect(),
            Kind::Packed(_) => buffer.to_u32s().unwrap(),
        }
    }
}

/// One movement case: the op, its data inputs, an optional index buffer, and the reference output.
struct Case {
    name: &'static str,
    out_shape: Vec<usize>,
    op: MovementOp,
    inputs: Vec<Tensor>,
    index: Option<Vec<i32>>,
    expected: Tensor,
}

/// One movement launch, as words: the spec, the data inputs, the `I32` index buffer and the output buffer's
/// initial words.
struct Launch {
    spec: MovementSpec,
    data: Vec<Vec<u32>>,
    index: Option<Vec<u32>>,
    out: Vec<u32>,
}

/// A backend that runs one movement and returns the output buffer's words.
type Device<'a> = dyn FnMut(&Launch) -> Vec<u32> + 'a;

/// The launch of `spec` over `inputs` and `index`, with an all-ones output buffer. A row scatter and an
/// in-place update write in place: the first input is the base, which becomes the output buffer's initial
/// words (the kernel never reads it as an input), and only the rest are the kernel's data inputs.
fn launch(kind: Kind, spec: &MovementSpec, inputs: &[Tensor], index: Option<&[i32]>) -> Launch {
    let in_place = spec.op.writes_in_place();
    let (seed, data) = match inputs.split_first() {
        Some((base, rest)) if in_place => (Some(base), rest),
        _ => (None, inputs),
    };
    Launch {
        spec: spec.clone(),
        data: data.iter().map(|t| kind.words(t)).collect(),
        index: index.map(|words| words.iter().map(|&w| w as u32).collect()),
        out: match seed {
            Some(base) => kind.words(base),
            None => kind.blank(&spec.out_shape),
        },
    }
}

/// The KIR interpreter as a [`Device`]: the word-level contract the real backends implement.
fn interpret(launch: &Launch) -> Vec<u32> {
    let spec = &launch.spec;
    let kind = Kind::of(&spec.storage);
    let body = poot_kernelgen::movement_body("movement", spec, None).expect("a valid request");
    body.verify().expect("the generated body verifies");
    let mut buffers: Vec<Buffer> = launch.data.iter().map(|w| kind.interp_buffer(w)).collect();
    if let Some(words) = &launch.index {
        buffers.push(Buffer::from_i32s(
            &words.iter().map(|&w| w as i32).collect::<Vec<_>>(),
        ));
    }
    buffers.push(kind.interp_buffer(&launch.out));
    run(
        &body,
        workgroups_covering(&body, spec.threads()),
        &mut buffers,
    )
    .expect("the body runs");
    kind.interp_words(&buffers.pop().unwrap())
}

/// `case` on `kind`'s storage through `device`, compared to the reference.
fn run_case(kind: Kind, case: &Case, device: &mut Device<'_>) {
    let spec = MovementSpec::new(kind.storage(), case.out_shape.clone(), case.op.clone());
    let out = device(&launch(kind, &spec, &case.inputs, case.index.as_deref()));
    let got = kind.tensor(&out, &case.out_shape);
    assert_eq!(got, case.expected, "{} on {kind:?}", case.name);
}

// --- the references ------------------------------------------------------------------------------------

fn transpose(input: &Tensor, perm: &[usize]) -> Tensor {
    let shape: Vec<usize> = perm.iter().map(|&axis| input.shape[axis]).collect();
    build(&shape, |c| {
        let mut source = vec![0; c.len()];
        for (d, &axis) in perm.iter().enumerate() {
            source[axis] = c[d];
        }
        input.at(&source)
    })
}

fn slice(input: &Tensor, axis: usize, start: usize, len: usize) -> Tensor {
    let mut shape = input.shape.clone();
    shape[axis] = len;
    build(&shape, |c| {
        let mut source = c.to_vec();
        source[axis] += start;
        input.at(&source)
    })
}

fn broadcast(input: &Tensor, shape: &[usize]) -> Tensor {
    let lead = shape.len() - input.shape.len();
    build(shape, |c| {
        let source: Vec<usize> = input
            .shape
            .iter()
            .enumerate()
            .map(|(i, &extent)| if extent == 1 { 0 } else { c[lead + i] })
            .collect();
        input.at(&source)
    })
}

fn gather(input: &Tensor, axis: usize, index_shape: &[usize], index: &[usize]) -> Tensor {
    let mut shape = input.shape[..axis].to_vec();
    shape.extend_from_slice(index_shape);
    shape.extend_from_slice(&input.shape[axis + 1..]);
    let index_rank = index_shape.len();
    build(&shape, |c| {
        let position = c[axis..axis + index_rank]
            .iter()
            .zip(index_shape)
            .fold(0, |flat, (&coord, &extent)| flat * extent + coord);
        let mut source = c[..axis].to_vec();
        source.push(index[position]);
        source.extend_from_slice(&c[axis + index_rank..]);
        input.at(&source)
    })
}

fn concat(parts: &[Tensor], axis: usize) -> Tensor {
    let mut shape = parts[0].shape.clone();
    shape[axis] = parts.iter().map(|t| t.shape[axis]).sum();
    build(&shape, |c| {
        let mut at = c[axis];
        for part in parts {
            if at < part.shape[axis] {
                let mut source = c.to_vec();
                source[axis] = at;
                return part.at(&source);
            }
            at -= part.shape[axis];
        }
        unreachable!("the coordinate lies inside the concatenated extent")
    })
}

fn update(operand: &Tensor, update: &Tensor, axis: usize, start: usize) -> Tensor {
    build(&operand.shape, |c| {
        if (start..start + update.shape[axis]).contains(&c[axis]) {
            let mut source = c.to_vec();
            source[axis] -= start;
            update.at(&source)
        } else {
            operand.at(c)
        }
    })
}

/// `base` with source row `t` written at pool slot `slots[t]` (`-1`: the row writes nothing).
fn scatter_rows(base: &Tensor, src: &Tensor, slots: &[i32]) -> Tensor {
    build(&base.shape, |c| {
        match slots.iter().position(|&slot| slot == c[0] as i32) {
            Some(row) => {
                let mut source = c.to_vec();
                source[0] = row;
                src.at(&source)
            }
            None => base.at(c),
        }
    })
}

// --- the bank ------------------------------------------------------------------------------------------

/// Every case of the bank, with lane values `width` bits wide.
fn bank(width: u32) -> Vec<Case> {
    let t = |shape: &[usize], seed| Tensor::counting(shape, seed, width);
    let mut cases = Vec::new();

    for (name, in_shape, perm) in [
        ("transpose 3d, odd last axis", vec![2, 3, 5], vec![2, 0, 1]),
        ("transpose 2d", vec![3, 5], vec![1, 0]),
        ("transpose identity", vec![2, 7], vec![0, 1]),
    ] {
        let input = t(&in_shape, 1);
        let expected = transpose(&input, &perm);
        cases.push(Case {
            name,
            out_shape: expected.shape.clone(),
            op: MovementOp::Transpose { in_shape, perm },
            inputs: vec![input],
            index: None,
            expected,
        });
    }

    for (name, in_shape, axis, start, len) in [
        ("slice middle axis", vec![2, 5, 3], 1, 1, 3),
        ("slice last axis to an odd row", vec![3, 7], 1, 2, 3),
        ("slice leading axis", vec![5, 3], 0, 2, 2),
        ("slice to one element", vec![4, 5], 1, 4, 1),
    ] {
        let input = t(&in_shape, 2);
        let expected = slice(&input, axis, start, len);
        cases.push(Case {
            name,
            out_shape: expected.shape.clone(),
            op: MovementOp::Slice {
                in_shape,
                axis,
                start,
            },
            inputs: vec![input],
            index: None,
            expected,
        });
    }

    for (name, in_shape, out_shape) in [
        ("broadcast a row", vec![3], vec![2, 3]),
        ("broadcast a unit axis", vec![2, 1, 5], vec![2, 4, 5]),
        ("broadcast a scalar", vec![], vec![2, 3]),
        ("broadcast a column", vec![4, 1], vec![4, 5]),
    ] {
        let input = t(&in_shape, 3);
        let expected = broadcast(&input, &out_shape);
        cases.push(Case {
            name,
            out_shape,
            op: MovementOp::Broadcast { in_shape },
            inputs: vec![input],
            index: None,
            expected,
        });
    }

    for (name, in_shape, out_shape) in [
        ("reshape across row widths", vec![2, 6], vec![3, 4]),
        ("reshape to odd rows", vec![3, 5], vec![5, 3]),
        ("reshape flatten", vec![3, 5], vec![15]),
    ] {
        let input = t(&in_shape, 4);
        let expected = Tensor {
            shape: out_shape.clone(),
            lanes: input.lanes.clone(),
        };
        cases.push(Case {
            name,
            out_shape,
            op: MovementOp::Reshape { in_shape },
            inputs: vec![input],
            index: None,
            expected,
        });
    }

    for (name, in_shape, axis, index_shape, index) in [
        ("gather rows", vec![4, 5], 0, vec![3], vec![2usize, 0, 3]),
        ("gather a scalar index", vec![4, 5], 0, vec![], vec![2]),
        (
            "gather a matrix of indices",
            vec![4, 3],
            0,
            vec![2, 2],
            vec![3, 3, 0, 1],
        ),
        ("gather middle axis", vec![2, 4, 3], 1, vec![2], vec![3, 1]),
        ("gather last axis", vec![3, 5], 1, vec![2], vec![4, 0]),
    ] {
        let input = t(&in_shape, 5);
        let expected = gather(&input, axis, &index_shape, &index);
        let index_numel = index.len();
        cases.push(Case {
            name,
            out_shape: expected.shape.clone(),
            op: MovementOp::Gather {
                in_shape,
                axis,
                index_numel,
            },
            inputs: vec![input],
            index: Some(index.iter().map(|&i| i as i32).collect()),
            expected,
        });
    }

    for (name, shapes, axis) in [
        (
            "concat middle axis, three inputs",
            vec![vec![2, 2, 3], vec![2, 1, 3], vec![2, 3, 3]],
            1,
        ),
        ("concat leading axis", vec![vec![1, 5], vec![2, 5]], 0),
        ("concat of one input is a copy", vec![vec![2, 5]], 1),
        (
            "concat last axis to an odd row",
            vec![vec![3, 2], vec![3, 3]],
            1,
        ),
        (
            "concat last axis, four inputs",
            vec![vec![2, 1], vec![2, 2], vec![2, 1], vec![2, 3]],
            1,
        ),
    ] {
        let inputs: Vec<Tensor> = shapes
            .iter()
            .enumerate()
            .map(|(i, s)| t(s, 10 + i as u32))
            .collect();
        let expected = concat(&inputs, axis);
        cases.push(Case {
            name,
            out_shape: expected.shape.clone(),
            op: MovementOp::Concat {
                axis,
                in_shapes: shapes,
            },
            inputs,
            index: None,
            expected,
        });
    }

    for (name, operand_shape, update_shape, axis, start) in [
        ("update middle axis", vec![2, 4, 5], vec![2, 2, 5], 1, 1),
        ("update last axis, odd rows", vec![3, 7], vec![3, 2], 1, 3),
        ("update one slot", vec![2, 4, 3], vec![2, 1, 3], 1, 2),
        ("update whole axis", vec![2, 3], vec![2, 3], 1, 0),
        ("update first axis", vec![4, 3], vec![2, 3], 0, 1),
        (
            "update last axis inside one word",
            vec![2, 9],
            vec![2, 1],
            1,
            6,
        ),
        (
            "update last axis, a word's tail and head",
            vec![2, 11],
            vec![2, 6],
            1,
            3,
        ),
    ] {
        let operand = t(&operand_shape, 20);
        let new = t(&update_shape, 21);
        let expected = update(&operand, &new, axis, start);
        // The copy form and the in-place form (whose output buffer already holds the operand) of one update.
        type Form = fn(Vec<usize>, usize, UpdateIndex) -> MovementOp;
        let forms: [Form; 2] = [
            |update_shape, axis, index| MovementOp::UpdateSlice {
                update_shape,
                axis,
                index,
            },
            |update_shape, axis, index| MovementOp::WriteSlice {
                update_shape,
                axis,
                index,
            },
        ];
        for form in forms {
            cases.push(Case {
                name,
                out_shape: operand_shape.clone(),
                op: form(update_shape.clone(), axis, UpdateIndex::Static(start)),
                inputs: vec![operand.clone(), new.clone()],
                index: None,
                expected: expected.clone(),
            });
            cases.push(Case {
                name,
                out_shape: operand_shape.clone(),
                op: form(update_shape.clone(), axis, UpdateIndex::Runtime),
                inputs: vec![operand.clone(), new.clone()],
                index: Some(vec![start as i32]),
                expected: expected.clone(),
            });
        }
    }

    // A runtime index too large for the window: both forms write only the positions inside the axis and keep
    // the operand everywhere else, so a too-large index never writes past its row.
    for (name, operand_shape, update_shape, axis, start) in [
        ("update past the end, clipped", vec![2, 5], vec![2, 3], 1, 4),
        ("update wholly past the end", vec![2, 5], vec![2, 3], 1, 7),
        (
            "update past the end of an outer axis",
            vec![3, 2, 3],
            vec![3, 2, 3],
            1,
            1,
        ),
    ] {
        let operand = t(&operand_shape, 22);
        let new = t(&update_shape, 23);
        let expected = update(&operand, &new, axis, start);
        for op in [
            MovementOp::UpdateSlice {
                update_shape: update_shape.clone(),
                axis,
                index: UpdateIndex::Runtime,
            },
            MovementOp::WriteSlice {
                update_shape: update_shape.clone(),
                axis,
                index: UpdateIndex::Runtime,
            },
        ] {
            cases.push(Case {
                name,
                out_shape: operand_shape.clone(),
                op,
                inputs: vec![operand.clone(), new.clone()],
                index: Some(vec![start as i32]),
                expected: expected.clone(),
            });
        }
    }

    for (name, base_shape, src_shape, slots) in [
        ("scatter rows", vec![4, 3], vec![2, 3], vec![2, 0]),
        (
            "scatter rows, odd width",
            vec![4, 5],
            vec![3, 5],
            vec![2, 3, 1],
        ),
        (
            "scatter rows, a padding row",
            vec![4, 5],
            vec![3, 5],
            vec![3, -1, 0],
        ),
        (
            "scatter of only padding keeps every row",
            vec![3, 2],
            vec![1, 2],
            vec![-1],
        ),
    ] {
        let base = t(&base_shape, 30);
        let src = t(&src_shape, 31);
        let expected = scatter_rows(&base, &src, &slots);
        cases.push(Case {
            name,
            out_shape: base_shape.clone(),
            op: MovementOp::ScatterRows {
                src_shape: src_shape.clone(),
                window: None,
            },
            inputs: vec![base.clone(), src.clone()],
            index: Some(slots.clone()),
            expected: expected.clone(),
        });
    }
    cases
}

fn every_kind() -> [Kind; 4] {
    [Kind::F32, Kind::I32, Kind::Packed(8), Kind::Packed(16)]
}

/// The whole bank on `device`: every op on every storage.
fn every_op_on_every_storage(device: &mut Device<'_>) {
    for kind in every_kind() {
        for case in bank(kind.width()) {
            run_case(kind, &case, device);
        }
    }
}

#[test]
fn every_op_matches_the_reference_on_every_storage() {
    for kind in every_kind() {
        for case in bank(kind.width()) {
            run_case(kind, &case, &mut interpret);
        }
    }
}

/// The specials of an 8-bit float: signed zero, the finite extremes of E4M3FN, the smallest subnormals, a few
/// ordinary values and both NaN codes.
const FP8_SPECIALS: [u32; 12] = [
    0x00, 0x80, 0x7e, 0xfe, 0x7f, 0xff, 0x01, 0x81, 0x38, 0xb8, 0x77, 0x08,
];

/// One raw-bit fixture: the op, its output shape, its index buffer, its data inputs and the reference.
type RawBitCase = (
    MovementOp,
    Vec<usize>,
    Option<Vec<i32>>,
    Vec<Tensor>,
    Tensor,
);

/// The raw-bit fixture: a `[3, 5]` table of the specials (odd width, so every row ends in padding lanes), and
/// a gather, a slice and a concat over it, each with its reference.
fn fp8_raw_bit_cases() -> Vec<RawBitCase> {
    let table = Tensor {
        shape: vec![3, 5],
        lanes: FP8_SPECIALS.iter().cycle().take(15).copied().collect(),
    };
    vec![
        (
            MovementOp::Gather {
                in_shape: vec![3, 5],
                axis: 0,
                index_numel: 4,
            },
            vec![4, 5],
            Some(vec![2, 0, 1, 2]),
            vec![table.clone()],
            gather(&table, 0, &[4], &[2, 0, 1, 2]),
        ),
        (
            MovementOp::Slice {
                in_shape: vec![3, 5],
                axis: 1,
                start: 1,
            },
            vec![3, 3],
            None,
            vec![table.clone()],
            slice(&table, 1, 1, 3),
        ),
        (
            MovementOp::Concat {
                axis: 1,
                in_shapes: vec![vec![3, 5], vec![3, 5]],
            },
            vec![3, 10],
            None,
            vec![table.clone(), table.clone()],
            concat(&[table.clone(), table], 1),
        ),
    ]
}

/// The special encodings of an 8-bit float move as raw bits through a gather, a slice and a concat on
/// `device`, and the decoded finite lanes equal the same movement over the decoded f32s (the dense movement
/// reference). Padding lanes of the inputs are all ones and those of the output start all ones, so neither
/// is required to be zero or read.
fn fp8_raw_bits_on(device: &mut Device<'_>) {
    use poot_quant::scalar::e4m3fn_to_f32;
    let decoded = |t: &Tensor| Tensor {
        shape: t.shape.clone(),
        lanes: t
            .lanes
            .iter()
            .map(|&b| e4m3fn_to_f32(b as u8).to_bits())
            .collect(),
    };
    let (byte_kind, dense_kind) = (Kind::Packed(8), Kind::F32);
    for (op, out_shape, index, inputs, reference) in fp8_raw_bit_cases() {
        let spec = |storage| MovementSpec::new(storage, out_shape.clone(), op.clone());
        let moved_words = device(&launch(
            byte_kind,
            &spec(byte_kind.storage()),
            &inputs,
            index.as_deref(),
        ));
        let moved = byte_kind.tensor(&moved_words, &out_shape);
        assert_eq!(moved, reference, "raw bits through {op:?}");

        // The same movement over the decoded f32s: decoded finite lanes are equal, NaN lanes stay NaN.
        let dense_inputs: Vec<Tensor> = inputs.iter().map(decoded).collect();
        let dense_words = device(&launch(
            dense_kind,
            &spec(dense_kind.storage()),
            &dense_inputs,
            index.as_deref(),
        ));
        let dense = dense_kind.tensor(&dense_words, &out_shape);
        let moved_decoded = decoded(&moved);
        for ((&byte, got), want) in moved
            .lanes
            .iter()
            .zip(&moved_decoded.lanes)
            .zip(&dense.lanes)
        {
            if byte & 0x7f == 0x7f {
                assert!(f32::from_bits(*got).is_nan() && f32::from_bits(*want).is_nan());
            } else {
                assert_eq!(got, want, "decoded finite lane {byte:#04x} through {op:?}");
            }
        }
    }
}

#[test]
fn fp8_special_encodings_move_as_raw_bits_and_decode_like_dense_movement() {
    fp8_raw_bits_on(&mut interpret);
}

/// A scatter split into windows writes the same output as the unsplit scatter: each window writes its own
/// range of the source units into the one pool buffer in turn, and the windows together cover every source
/// unit.
fn windowed_scatter_on(kind: Kind, device: &mut Device<'_>) {
    let base = Tensor::counting(&[6, 5], 40, kind.width());
    let src = Tensor::counting(&[3, 5], 41, kind.width());
    let slots = [4, -1, 1];
    let expected = scatter_rows(&base, &src, &slots);
    let units = kind.storage().units(&[3, 5]);
    let cut = units / 2 + 1;
    let mut pool = base;
    for (offset, threads) in [(0, cut), (cut, units - cut)] {
        let spec = MovementSpec::new(
            kind.storage(),
            vec![6, 5],
            MovementOp::ScatterRows {
                src_shape: vec![3, 5],
                window: Some(UnitWindow { offset, threads }),
            },
        );
        assert_eq!(spec.threads(), threads);
        let out = device(&launch(
            kind,
            &spec,
            &[pool.clone(), src.clone()],
            Some(&slots),
        ));
        pool = kind.tensor(&out, &[6, 5]);
    }
    assert_eq!(pool, expected, "windowed scatter on {kind:?}");
}

#[test]
fn a_windowed_scatter_composes_to_the_whole_output() {
    for kind in every_kind() {
        windowed_scatter_on(kind, &mut interpret);
    }
}

/// The generator refuses the requests it cannot state, with a typed error naming the contract.
#[test]
fn malformed_requests_are_typed_refusals() {
    let element = Storage::Elements { carrier: Ty::F32 };
    let refuse = |storage: Storage, out_shape: Vec<usize>, op: MovementOp| {
        poot_kernelgen::movement_body("m", &MovementSpec::new(storage, out_shape, op), None)
            .expect_err("the request is malformed")
    };
    assert!(matches!(
        refuse(
            element.clone(),
            vec![0, 3],
            MovementOp::Broadcast { in_shape: vec![3] }
        ),
        KernelGenError::BelowMinimum { .. }
    ));
    assert!(matches!(
        refuse(
            element.clone(),
            vec![2, 3],
            MovementOp::Slice {
                in_shape: vec![2, 5],
                axis: 1,
                start: 4
            }
        ),
        KernelGenError::ExceedsBound { .. }
    ));
    assert!(matches!(
        refuse(
            element.clone(),
            vec![2, 3],
            MovementOp::Transpose {
                in_shape: vec![3, 2],
                perm: vec![1, 1]
            }
        ),
        KernelGenError::Unsupported { .. }
    ));
    assert!(matches!(
        refuse(
            element.clone(),
            vec![2, 3],
            MovementOp::Gather {
                in_shape: vec![4, 3],
                axis: 2,
                index_numel: 2
            }
        ),
        KernelGenError::AxisOutOfRange { .. }
    ));
    assert!(matches!(
        refuse(
            element.clone(),
            vec![2, 4],
            MovementOp::Concat {
                axis: 1,
                in_shapes: vec![vec![2, 3], vec![2, 3]]
            }
        ),
        KernelGenError::NotEqualTo { .. }
    ));
    assert!(matches!(
        refuse(
            Storage::PackedRows { lane_bits: 4 },
            vec![2, 3],
            MovementOp::Reshape {
                in_shape: vec![3, 2]
            }
        ),
        KernelGenError::Unsupported { .. }
    ));
    assert!(matches!(
        refuse(
            element,
            vec![4, 3],
            MovementOp::UpdateSlice {
                update_shape: vec![2, 3],
                axis: 0,
                index: UpdateIndex::Static(3)
            }
        ),
        KernelGenError::ExceedsBound { .. }
    ));
}

/// The whole movement bank on a device: every op on every storage, the FP8 raw-bit fixture, the windowed
/// scatter on every storage.
fn device_bank(device: &mut Device<'_>) {
    every_op_on_every_storage(device);
    fp8_raw_bits_on(device);
    for kind in every_kind() {
        windowed_scatter_on(kind, device);
    }
}

/// wgpu (SPIR-V through the Vulkan runtime): skips with no adapter. Run serial under `/tmp/poot-gpu.lock`.
#[test]
fn the_movement_bank_matches_the_reference_on_wgpu() {
    use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
    let Ok(ctx) = poot_wgpu_runtime::Context::new() else {
        poot_test_util::skip_report("poot-kernelgen", &"no wgpu context");
        return;
    };
    let dir = std::env::temp_dir().join("poot-kernelgen-movement-wgpu");
    std::fs::create_dir_all(&dir).unwrap();
    let mut count = 0usize;
    device_bank(&mut |launch| {
        count += 1;
        let name = format!("movement_{count}");
        let body = poot_kernelgen::movement_body(&name, &launch.spec, None).unwrap();
        let path = artifact_path(&dir, &name, Target::SpirvVulkan);
        let artifact = compile_kernel(&body, poot_target::Backend::SpirvVulkan, &path)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernel = kernel_handle(&body, &artifact).unwrap();
        // Resident buffers, so the output starts as `launch.out` (an in-place scatter updates the pool
        // it is handed; every other op overwrites every word).
        let mut owned: Vec<_> = launch.data.iter().map(|w| ctx.upload_u32(w)).collect();
        if let Some(words) = &launch.index {
            owned.push(ctx.upload_u32(words));
        }
        let refs: Vec<_> = owned.iter().collect();
        let out = ctx.upload_u32(&launch.out);
        ctx.dispatch_dev(
            &name,
            &kernel,
            body.workgroup_size,
            [launch.spec.threads() as u32, 1, 1],
            &refs,
            &out,
        )
        .unwrap_or_else(|e| panic!("{name}: dispatch: {e}"));
        let mut bytes = vec![0u8; launch.out.len() * 4];
        ctx.read_bytes(&out, &mut bytes).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect()
    });
}

/// ROCm/HSA (AMDGCN): skips with no HSA library or GPU agent. Run serial under `/tmp/poot-gpu.lock`.
#[test]
fn the_movement_bank_matches_the_reference_on_rocm() {
    use poot_rocm_runtime::{RocmContext, RocmError};
    use poot_runtime_common::BufferRole;
    use poot_target::ElementKind;
    let ctx = match RocmContext::new() {
        Ok(ctx) => ctx,
        Err(RocmError::LibraryNotFound(_) | RocmError::NoGpuAgent) => {
            poot_test_util::skip_report("poot-kernelgen", &"no ROCm device");
            return;
        }
        Err(error) => panic!("{error}"),
    };
    let arch = poot_target::AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
        .expect("classify the device arch");
    let target = poot_codegen::Target::AmdGcn(arch);
    let dir = std::env::temp_dir().join("poot-kernelgen-movement-rocm");
    std::fs::create_dir_all(&dir).unwrap();
    let mut count = 0usize;
    device_bank(&mut |launch| {
        count += 1;
        let name = format!("movement_{count}");
        let body = poot_kernelgen::movement_body(&name, &launch.spec, None).unwrap();
        let path = poot_codegen::artifact_path(&dir, &name, target);
        let artifact =
            poot_codegen::compile_kernel(&body, poot_target::Backend::AmdGcn(arch), &path)
                .unwrap_or_else(|e| panic!("{name}: {e}"));
        let compiled = poot_codegen::kernel_handle(&body, &artifact).unwrap();
        let module = ctx
            .load_hsaco(&compiled)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernel = ctx.lookup_kernel(&module, compiled.entry_point()).unwrap();
        // Every buffer travels as 4-byte words; the f32 path is a lossless bit copy.
        let upload = |words: &[u32]| {
            let as_f32: Vec<f32> = words.iter().map(|&w| f32::from_bits(w)).collect();
            ctx.upload_f32(&as_f32, BufferRole::Input).unwrap()
        };
        let mut owned: Vec<_> = launch.data.iter().map(|w| upload(w)).collect();
        if let Some(words) = &launch.index {
            owned.push(upload(words));
        }
        let inputs = owned.len();
        // The output starts as `launch.out`: an in-place scatter updates the pool it is handed.
        let seed: Vec<f32> = launch.out.iter().map(|&w| f32::from_bits(w)).collect();
        owned.push(ctx.upload_f32(&seed, BufferRole::Output).unwrap());
        let mut elements = vec![ElementKind::I32; inputs];
        elements.push(ElementKind::F32);
        let slots: Vec<_> = owned
            .iter()
            .map(|buffer| (buffer, buffer.elem_count() as u64))
            .collect();
        let kernarg = ctx.bind_kernarg(&kernel, &slots).unwrap();
        let block_x = body.workgroup_size[0];
        let grid_x = (launch.spec.threads() as u32).div_ceil(block_x) * block_x;
        ctx.dispatch(
            &kernel,
            &kernarg,
            &elements,
            (grid_x, 1, 1),
            (block_x, 1, 1),
        )
        .unwrap_or_else(|e| panic!("{name}: dispatch: {e}"));
        let mut got = vec![0.0f32; launch.out.len()];
        ctx.download_f32(owned.last().unwrap(), &mut got).unwrap();
        got.iter().map(|v| v.to_bits()).collect()
    });
}

/// NVPTX: skips with no NVIDIA device (the PTX pod batch runs it).
#[test]
fn the_movement_bank_matches_the_reference_on_ptx() {
    use poot_codegen::{Target, artifact_path, compile_kernel, kernel_handle};
    use poot_ptx_runtime::PtxContext;
    let Ok(ctx) = PtxContext::new() else {
        poot_test_util::skip_report("poot-kernelgen", &"no PTX device");
        return;
    };
    let dir = std::env::temp_dir().join("poot-kernelgen-movement-ptx");
    std::fs::create_dir_all(&dir).unwrap();
    let mut count = 0usize;
    device_bank(&mut |launch| {
        count += 1;
        let name = format!("movement_{count}");
        let body = poot_kernelgen::movement_body(&name, &launch.spec, None).unwrap();
        let path = artifact_path(&dir, &name, Target::Nvptx);
        let artifact = compile_kernel(&body, poot_target::Backend::Nvptx, &path)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let kernel = kernel_handle(&body, &artifact).unwrap();
        let upload = |words: &[u32]| {
            let as_i32: Vec<i32> = words.iter().map(|&w| w as i32).collect();
            ctx.upload_i32(&as_i32).unwrap()
        };
        let mut owned: Vec<_> = launch.data.iter().map(|w| upload(w)).collect();
        if let Some(words) = &launch.index {
            owned.push(upload(words));
        }
        let lengths: Vec<usize> = owned.iter().map(|b| b.elem_count()).collect();
        let refs: Vec<_> = owned.iter().collect();
        // The output starts as `launch.out`: an in-place scatter updates the pool it is handed.
        let seed: Vec<f32> = launch.out.iter().map(|&w| f32::from_bits(w)).collect();
        let out = ctx.upload_f32(&seed).unwrap();
        ctx.dispatch_dev(
            &name,
            &kernel,
            body.workgroup_size,
            [launch.spec.threads() as u32, 1, 1],
            &refs,
            &lengths,
            &out,
            out.elem_count(),
        )
        .unwrap_or_else(|e| panic!("{name}: dispatch: {e}"));
        ctx.download_f32(&out)
            .unwrap()
            .iter()
            .map(|v| v.to_bits())
            .collect()
    });
}
