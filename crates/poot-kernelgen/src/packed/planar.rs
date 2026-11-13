//! `PlanarPlan`: the planar (safetensors) storage front end for the descriptor-driven packed decode
//! emitter (dquant.md 3.3, card 542c). A planar format has one source tensor per
//! [`PlanarRole`] (`PackedWeight::sources()`'s order), addressed directly by the logical
//! coordinate `[o, k]` through each operand's `grid`/`packing`/`major` (`poot_quant::decode`'s
//! `read_planar`/`stored_shape` is the host reference this mirrors bit for bit). Block storage has
//! its own front end ([`super::decode_plan::DecodePlan`]); the two share `emit_read_bits` and
//! `emit_float_decode` (the descriptor's per-format constants are the only difference between them,
//! not a second decoder).
//!
//! Unlike a block, a planar operand carries no `SubScale`/`Min`/`SubMin`: every format in this
//! front end (E4M3 per-channel/128x128, E2M1 row-32, GPTQ, AWQ) has at most `Scale` and `Zero`
//! (`format.rs` module docs' operand table), so [`Factors::min`] is always `None` here.

use poot_kernel_ir::{BinOp, Local, Rvalue, Ty};
use poot_quant::format::{
    Axis, Extent, FieldEncoding, FormatDescriptor, GroupMap, LaneOrder, Major, PlanarOperand,
    PlanarRole, Storage, ValueMap, WeightFormat,
};

use super::decode_plan::Factors;
use super::float_decode::{FloatDecode, emit_float_decode};
use super::kernel::PackedRunSharing;
use super::source::Source;
use crate::KernelGenError;
use crate::emit::{Emit, bin, c32, copy, cu};
use poot_target::RoundingConversions;

fn reject(format: WeightFormat, reason: &'static str) -> KernelGenError {
    KernelGenError::UnsupportedDescriptor { format, reason }
}

/// The `K` length over which a planar scale is one value.
enum ScaleRun {
    /// The whole row (or there is no scale).
    Whole,
    /// Aligned runs of this many values.
    Values(usize),
    /// Every `K` index names its own (act-order groups).
    PerElement,
}

pub(crate) struct PlanarPlan {
    format: WeightFormat,
    /// Operands in `PackedWeight::sources()` order: one kernel parameter per entry.
    operands: Vec<PlanarOperand>,
    value: ValueMap,
    /// How the `Codes` of a `ValueMap::Float` format decode; `None` for any other value map.
    float: Option<FloatDecode>,
    codes: usize,
    scale: Option<usize>,
    zero: Option<usize>,
    group_index: Option<usize>,
}

impl PlanarPlan {
    pub(crate) fn of(descriptor: &FormatDescriptor) -> Result<Self, KernelGenError> {
        let format = descriptor.format;
        let Storage::Planar(layout) = descriptor.storage else {
            return Err(reject(format, "block storage (card 542a/542b's front end)"));
        };
        let operands = layout.operands.to_vec();
        let codes = operands
            .iter()
            .position(|operand| operand.role == PlanarRole::Codes)
            .ok_or(reject(format, "no Codes operand"))?;
        let scale = operands
            .iter()
            .position(|operand| operand.role == PlanarRole::Scale);
        let zero = operands
            .iter()
            .position(|operand| operand.role == PlanarRole::Zero);
        let group_index = operands
            .iter()
            .position(|operand| operand.role == PlanarRole::GroupIndex);
        let float = match (descriptor.value, operands[codes].encoding) {
            (ValueMap::Float, FieldEncoding::Float(format)) => Some(FloatDecode::exact(format)),
            (ValueMap::Float, _) => {
                return Err(reject(
                    format,
                    "a float value map over a Codes operand that is not float-encoded",
                ));
            }
            _ => None,
        };
        Ok(Self {
            format,
            operands,
            value: descriptor.value,
            float,
            codes,
            scale,
            zero,
            group_index,
        })
    }

    /// This plan decoding its values through what the device keeps ([`FloatDecode::shifted_on`]): a float code
    /// that meets a scale takes the cheaper decode, and [`Self::emit_factors`] folds the power of two it leaves
    /// into the scale, once per factor run.
    pub(crate) fn on(mut self, conversions: RoundingConversions) -> Self {
        if self.scale.is_some() {
            self.float = self.float.map(|decode| decode.shifted_on(conversions));
        }
        self
    }

    /// One kernel word-slice parameter per entry, in `PackedWeight::sources()` order.
    pub(crate) fn source_count(&self) -> usize {
        self.operands.len()
    }

    /// The `Scale` factor at logical `(o, k)`, or `Factors { scale: None, min: None }` when the
    /// format has no `Scale` operand (never true for the four families this front end serves, kept
    /// for symmetry with the block front end's `Factors`). Planar formats never have `Min`.
    pub(crate) fn emit_factors(
        &self,
        e: &mut Emit,
        src: &[Source],
        o: Local,
        k: Local,
        k_total: Local,
        out_total: Local,
    ) -> Factors {
        let Some(scale_pos) = self.scale else {
            return Factors {
                scale: None,
                min: None,
            };
        };
        let operand = self.operands[scale_pos];
        let bits = self.emit_operand_bits(e, src, scale_pos, &operand, o, k, k_total, out_total);
        let FieldEncoding::Float(format) = operand.encoding else {
            unreachable!("format tests: a Scale operand is always Float-encoded")
        };
        let scale = emit_float_decode(e, format, bits);
        let scale = self
            .float
            .map_or(scale, |decode| decode.fold_scale(e, scale));
        Factors {
            scale: Some(scale),
            min: None,
        }
    }

    /// One logical value at `(o, k)`, combined with `factors` through the value formula (`Scale *
    /// value`; planar formats never have a `Min` term). `ValueMap::Integer` reads `Zero` (when the
    /// format has one) and subtracts it from the code as an integer, before the one f32 cast
    /// (`poot_quant::decode::Formula::evaluate`'s `(code.integer() - zero) as f32`, never
    /// `code as f32 - zero as f32`): the codes bit reads to `emit_factors`.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn emit_value(
        &self,
        e: &mut Emit,
        src: &[Source],
        o: Local,
        k: Local,
        k_total: Local,
        out_total: Local,
        factors: &Factors,
    ) -> Local {
        let codes_operand = self.operands[self.codes];
        let code_bits =
            self.emit_operand_bits(e, src, self.codes, &codes_operand, o, k, k_total, out_total);
        let value = match self.value {
            ValueMap::Float => self
                .float
                .expect("a Float value map has a float decode")
                .emit(e, code_bits),
            ValueMap::Integer => {
                let code_int = emit_encoded_integer(e, codes_operand.encoding, code_bits);
                let zero_int = match self.zero {
                    Some(zero_pos) => {
                        let zero_operand = self.operands[zero_pos];
                        let zero_bits = self.emit_operand_bits(
                            e,
                            src,
                            zero_pos,
                            &zero_operand,
                            o,
                            k,
                            k_total,
                            out_total,
                        );
                        emit_encoded_integer(e, zero_operand.encoding, zero_bits)
                    }
                    None => e.let_(
                        Ty::I32,
                        Rvalue::Use(poot_kernel_ir::Operand::Const(
                            poot_kernel_ir::Constant::I32(0),
                        )),
                    ),
                };
                let diff = bin(e, Ty::I32, BinOp::Sub, copy(code_int), copy(zero_int));
                e.let_(
                    Ty::F32,
                    Rvalue::Cast {
                        to: Ty::F32,
                        operand: copy(diff),
                    },
                )
            }
            ValueMap::Lattice(_) => {
                unreachable!("format tests: a planar format's value map is Integer or Float")
            }
        };
        match factors.scale {
            Some(scale) => e.let_(
                Ty::F32,
                Rvalue::BinaryOp(BinOp::Mul, copy(scale), copy(value)),
            ),
            None => value,
        }
    }

    /// The `count` logical values `(o, k0 + j)` for `j in 0..count` of one weight row, `k0` a multiple of `count`
    /// and every index below `k_total`: [`Self::emit_factors`] and [`Self::emit_value`] per element, except that
    /// the run shares what its elements share. A scale its elements cannot straddle (a `K` extent that `count`
    /// divides) is decoded once. A `Float` format whose codes are one unpacked stored element per `K` index
    /// reads the whole run as one `count * bits` field (at most 32 bits) and slices it, so the address chain and
    /// the word loads are emitted once per run instead of once per element. It is [`Self::emit_run_load`] then
    /// [`Self::emit_run_decode`].
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn emit_run(
        &self,
        e: &mut Emit,
        src: &[Source],
        o: Local,
        k0: Local,
        count: usize,
        k_total: Local,
        out_total: Local,
    ) -> Vec<Local> {
        let loaded = self.emit_run_load(e, src, o, k0, count, k_total, out_total);
        self.emit_run_decode(e, src, o, k0, count, k_total, out_total, &loaded)
    }

    /// The width in bits of the one field a run of `count` reads whole, or `None` when the run reads each element's
    /// codes through its own address chain: a `Float` format whose codes are one unpacked stored element per `K`
    /// index, at most 32 bits for the run.
    fn run_field_bits(&self, count: usize) -> Option<u32> {
        let codes = self.operands[self.codes];
        let field_bits = codes.bits * count as u32;
        let sliced = codes.packing.is_none()
            && codes.major == Major::OutMajor
            && codes.grid[1] == Extent::Values(1)
            && field_bits <= 32;
        (self.float.is_some() && count > 1 && sliced).then_some(field_bits)
    }

    /// What a run reads from memory ahead of its decode: the scale the run shares (decoded, when its elements cannot
    /// straddle a scale group) then the one field of codes ([`Self::run_field_bits`]). A run that has neither loads
    /// nothing here and reads per element in [`Self::emit_run_decode`], so a software pipeline over it is a no-op.
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn emit_run_load(
        &self,
        e: &mut Emit,
        src: &[Source],
        o: Local,
        k0: Local,
        count: usize,
        k_total: Local,
        out_total: Local,
    ) -> Vec<Local> {
        let mut loaded = Vec::new();
        if self.run_shares_factors(count) {
            let factors = self.emit_factors(e, src, o, k0, k_total, out_total);
            loaded.extend(factors.scale);
        }
        if let Some(field_bits) = self.run_field_bits(count) {
            let codes = self.operands[self.codes];
            let bit = self.emit_operand_bit(e, src, &codes, o, k0, k_total, out_total);
            loaded.push(src[self.codes].read_bits(e, bit, field_bits));
        }
        loaded
    }

    /// The values of the run whose loads are `loaded` ([`Self::emit_run_load`]).
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn emit_run_decode(
        &self,
        e: &mut Emit,
        src: &[Source],
        o: Local,
        k0: Local,
        count: usize,
        k_total: Local,
        out_total: Local,
        loaded: &[Local],
    ) -> Vec<Local> {
        let shared = self.run_shares_factors(count).then(|| Factors {
            scale: self.scale.map(|_| loaded[0]),
            min: None,
        });
        let Some(field_bits) = self.run_field_bits(count) else {
            return self.emit_run_elements(e, src, o, k0, count, k_total, out_total, shared);
        };
        let codes = self.operands[self.codes];
        let decode = self.float.expect("a field run has a float decode");
        let field = *loaded.last().expect("a field run loaded its field");
        debug_assert!(field_bits <= 32);
        let mask = (1u32 << codes.bits) - 1;
        (0..count)
            .map(|j| {
                let shifted = match j {
                    0 => field,
                    _ => bin(
                        e,
                        Ty::U32,
                        BinOp::Shr,
                        copy(field),
                        c32(j as u32 * codes.bits),
                    ),
                };
                let code = bin(e, Ty::U32, BinOp::BitAnd, copy(shifted), c32(mask));
                let value = decode.emit(e, code);
                let own;
                let factors = match &shared {
                    Some(factors) => factors,
                    None => {
                        let k_index = bin(e, Ty::Usize, BinOp::Add, copy(k0), cu(j));
                        own = self.emit_factors(e, src, o, k_index, k_total, out_total);
                        &own
                    }
                };
                match factors.scale {
                    Some(scale) => e.let_(
                        Ty::F32,
                        Rvalue::BinaryOp(BinOp::Mul, copy(scale), copy(value)),
                    ),
                    None => value,
                }
            })
            .collect()
    }

    /// [`Self::emit_run`]'s element-by-element form: each element addresses its own codes, and decodes its own
    /// factors unless `shared` holds the run's.
    #[expect(clippy::too_many_arguments)]
    fn emit_run_elements(
        &self,
        e: &mut Emit,
        src: &[Source],
        o: Local,
        k0: Local,
        count: usize,
        k_total: Local,
        out_total: Local,
        shared: Option<Factors>,
    ) -> Vec<Local> {
        (0..count)
            .map(|j| {
                let k_index = bin(e, Ty::Usize, BinOp::Add, copy(k0), cu(j));
                let own;
                let factors = match &shared {
                    Some(factors) => factors,
                    None => {
                        own = self.emit_factors(e, src, o, k_index, k_total, out_total);
                        &own
                    }
                };
                self.emit_value(e, src, o, k_index, k_total, out_total, factors)
            })
            .collect()
    }

    /// What a Gemv lane run shares of this format's decode ([`PackedRunSharing`]). Codes stored one to an
    /// element read contiguously, so a run shares its scale group like a block format does. Codes packed into words
    /// (GPTQ, AWQ) are addressed one element at a time whatever the scale does: a run decodes its scale once
    /// ([`Self::emit_run`]) and still pays an address chain per element, which is why a longer run only spreads the
    /// loads (gfx1151, 17408 x 5120, GPTQ group 128: 111 GB/s at run 8, 60 at 16, 51 at 32).
    pub(crate) fn run_sharing(&self) -> PackedRunSharing {
        if self.operands[self.codes].packing.is_some() {
            return PackedRunSharing::Elementwise;
        }
        match self.scale_run() {
            ScaleRun::Whole => PackedRunSharing::WholeRow,
            ScaleRun::Values(factor_run) => PackedRunSharing::FactorRun { factor_run },
            ScaleRun::PerElement => PackedRunSharing::Elementwise,
        }
    }

    /// The `K` length over which the scale is one value.
    fn scale_run(&self) -> ScaleRun {
        match self.scale.map(|position| self.operands[position].grid[1]) {
            None | Some(Extent::Whole) => ScaleRun::Whole,
            Some(Extent::Values(values)) => ScaleRun::Values(values),
            Some(Extent::Group) => self
                .group_size()
                .map_or(ScaleRun::PerElement, ScaleRun::Values),
        }
    }

    /// Whether every element of a run of `count` (starting at a multiple of `count`) has the same factors: the
    /// scale's `K` extent is a multiple of `count`, or whole, or there is no scale.
    fn run_shares_factors(&self, count: usize) -> bool {
        match self.scale_run() {
            ScaleRun::Whole => true,
            ScaleRun::Values(values) => values.is_multiple_of(count),
            ScaleRun::PerElement => false,
        }
    }

    /// `operand`'s raw stored bits at logical `(o, k)`: `poot_quant::decode::read_planar`'s
    /// `element`/`lane`/`stored_shape`/bit-offset chain, emitted.
    #[expect(clippy::too_many_arguments)]
    fn emit_operand_bits(
        &self,
        e: &mut Emit,
        src: &[Source],
        source_index: usize,
        operand: &PlanarOperand,
        o: Local,
        k: Local,
        k_total: Local,
        out_total: Local,
    ) -> Local {
        let bit = self.emit_operand_bit(e, src, operand, o, k, k_total, out_total);
        src[source_index].read_bits(e, bit, operand.bits)
    }

    /// The bit offset in `operand`'s source of the stored bits at logical `(o, k)`.
    #[expect(clippy::too_many_arguments)]
    fn emit_operand_bit(
        &self,
        e: &mut Emit,
        src: &[Source],
        operand: &PlanarOperand,
        o: Local,
        k: Local,
        k_total: Local,
        out_total: Local,
    ) -> Local {
        let needs_group = operand.grid.contains(&Extent::Group);
        let group_of_k = needs_group.then(|| self.emit_group_of(e, src, k));
        let groups_total = needs_group.then(|| self.emit_groups_total(e, k_total));
        let out_element = emit_extent_element(e, operand.grid[0], o, group_of_k);
        let k_element = emit_extent_element(e, operand.grid[1], k, group_of_k);
        let mut element = [out_element, k_element];
        let mut lane = e.let_(Ty::Usize, Rvalue::Use(cu(0)));
        if let Some(packing) = operand.packing {
            let axis = axis_index(packing.axis);
            let lane_value = bin(
                e,
                Ty::Usize,
                BinOp::Rem,
                copy(element[axis]),
                cu(packing.values_per_word),
            );
            element[axis] = bin(
                e,
                Ty::Usize,
                BinOp::Div,
                copy(element[axis]),
                cu(packing.values_per_word),
            );
            lane = emit_lane_order(e, packing.lanes, lane_value);
        }
        let out_grid = emit_extent_grid(e, operand.grid[0], out_total, groups_total);
        let k_grid = emit_extent_grid(e, operand.grid[1], k_total, groups_total);
        let mut grid = [out_grid, k_grid];
        if let Some(packing) = operand.packing {
            let axis = axis_index(packing.axis);
            grid[axis] = emit_div_ceil(e, grid[axis], packing.values_per_word);
        }
        let (row, columns) = match operand.major {
            Major::OutMajor => (element[0], grid[1]),
            Major::KMajor => (element[1], grid[0]),
        };
        let column = match operand.major {
            Major::OutMajor => element[1],
            Major::KMajor => element[0],
        };
        let row_cols = bin(e, Ty::Usize, BinOp::Mul, copy(row), copy(columns));
        let elem_index = bin(e, Ty::Usize, BinOp::Add, copy(row_cols), copy(column));
        let elem_bits = operand.element_bytes() * 8;
        let base_bit = bin(e, Ty::Usize, BinOp::Mul, copy(elem_index), cu(elem_bits));
        let lane_bit = bin(
            e,
            Ty::Usize,
            BinOp::Mul,
            copy(lane),
            cu(operand.bits as usize),
        );
        bin(e, Ty::Usize, BinOp::Add, copy(base_bit), copy(lane_bit))
    }

    /// The static per-group K-value count (`GroupMap::Contiguous`'s size), when this format's
    /// groups are contiguous.
    fn group_size(&self) -> Option<usize> {
        match self.format {
            WeightFormat::Gptq {
                groups: GroupMap::Contiguous { size },
            }
            | WeightFormat::Awq { group_size: size } => Some(size.get()),
            _ => None,
        }
    }

    /// The static total group count (`GroupMap::Indexed`'s count), when this format's groups are
    /// read from a `GroupIndex` operand rather than computed from `k`.
    fn group_count(&self) -> Option<usize> {
        match self.format {
            WeightFormat::Gptq {
                groups: GroupMap::Indexed { groups },
            } => Some(groups.get()),
            _ => None,
        }
    }

    /// The quantization group total (`poot_quant::decode::FormatDescriptor::groups`, emitted):
    /// `ceil(k_total / group_size)` for contiguous groups, the static count for act-order groups.
    fn emit_groups_total(&self, e: &mut Emit, k_total: Local) -> Local {
        match self.group_size() {
            Some(size) => emit_div_ceil(e, k_total, size),
            None => {
                let count = self
                    .group_count()
                    .expect("a Group-extent operand's format has contiguous or indexed groups");
                e.let_(Ty::Usize, Rvalue::Use(cu(count)))
            }
        }
    }

    /// The quantization group of K index `k` (`poot_quant::decode::FormatDescriptor::group_of`,
    /// emitted): `k / group_size` for contiguous groups, `g_idx[k]` (the `GroupIndex` operand, one
    /// `i32` word per K index) for act-order groups.
    fn emit_group_of(&self, e: &mut Emit, src: &[Source], k: Local) -> Local {
        match self.group_size() {
            Some(size) => bin(e, Ty::Usize, BinOp::Div, copy(k), cu(size)),
            None => {
                let group_index = self
                    .group_index
                    .expect("act-order GPTQ has a GroupIndex source");
                let bit = bin(e, Ty::Usize, BinOp::Mul, copy(k), cu(32));
                let bits = src[group_index].read_bits(e, bit, 32);
                // g_idx is stored as a signed i32, but every admitted value is a nonnegative,
                // in-range group index: `poot_quant::PackedPayload::try_new`'s content validation
                // (`FormatDescriptor::first_invalid_group_index`, card 542c) refuses a
                // negative or out-of-range g_idx for every element at construction, before this
                // kernel (or any other decoder) ever reads one - not merely the CPU oracle's own
                // per-element `group_of` check, which alone would leave a device kernel with no
                // check of its own to fail closed with. Reading the same bits as unsigned and
                // widening to usize gives the identical value for that domain, one Cast instead of
                // a Bitcast plus a sign check the kernel never needs (kernels decode the admitted
                // domain only, dquant.md 4).
                e.let_(
                    Ty::Usize,
                    Rvalue::Cast {
                        to: Ty::Usize,
                        operand: copy(bits),
                    },
                )
            }
        }
    }
}

/// `poot_quant::format::Axis` as a `[out, k]` array index.
const fn axis_index(axis: Axis) -> usize {
    match axis {
        Axis::Out => 0,
        Axis::K => 1,
    }
}

/// `ceil(value / divisor)`, emitted (`divisor` a compile-time constant, at least 1).
fn emit_div_ceil(e: &mut Emit, value: Local, divisor: usize) -> Local {
    let sum = bin(e, Ty::Usize, BinOp::Add, copy(value), cu(divisor - 1));
    bin(e, Ty::Usize, BinOp::Div, copy(sum), cu(divisor))
}

/// One axis of `poot_quant::decode::FormatDescriptor::stored_shape`'s grid, before packing: the
/// element count `axis_total` (`k_total` or `out_total`) reduces to under one `Extent::Values`,
/// `Extent::Whole` is always one element, `Extent::Group` is the precomputed group total.
fn emit_extent_grid(
    e: &mut Emit,
    extent: Extent,
    axis_total: Local,
    groups_total: Option<Local>,
) -> Local {
    match extent {
        Extent::Values(values) => emit_div_ceil(e, axis_total, values),
        Extent::Whole => e.let_(Ty::Usize, Rvalue::Use(cu(1))),
        Extent::Group => {
            groups_total.expect("an Extent::Group grid axis has a precomputed group total")
        }
    }
}

/// One axis of `poot_quant::decode::FormatDescriptor::read_planar`'s `element` map, before packing:
/// which stored element logical coordinate `coordinate` (`o` or `k`) falls into.
fn emit_extent_element(
    e: &mut Emit,
    extent: Extent,
    coordinate: Local,
    group_of_k: Option<Local>,
) -> Local {
    match extent {
        Extent::Values(values) => bin(e, Ty::Usize, BinOp::Div, copy(coordinate), cu(values)),
        Extent::Whole => e.let_(Ty::Usize, Rvalue::Use(cu(0))),
        Extent::Group => {
            group_of_k.expect("an Extent::Group element axis has a precomputed group index")
        }
    }
}

/// `poot_quant::format::LaneOrder::lane`, emitted, for a runtime `value` (0 below the packing's
/// `values_per_word`). `Awq`'s permutation `[0, 4, 1, 5, 2, 6, 3, 7]` is a right rotation by one bit
/// of `value`'s low three bits (`AWQ_OPERANDS` is always `int4x8`, eight lanes): `value / 2 + (value
/// % 2) * 4`, checked against every entry of the table (0->0, 1->4, ..., 7->7). Plain arithmetic
/// rather than an 8-way `select_u32` tree, since no bit ops on `Ty::Usize` exist in this crate's KIR
/// emission today (`emit.rs`'s bit tricks are all `Ty::U32`).
fn emit_lane_order(e: &mut Emit, lanes: LaneOrder, value: Local) -> Local {
    match lanes {
        LaneOrder::Natural => value,
        LaneOrder::Awq => {
            let half = bin(e, Ty::Usize, BinOp::Div, copy(value), cu(2));
            let rem = bin(e, Ty::Usize, BinOp::Rem, copy(value), cu(2));
            let quad = bin(e, Ty::Usize, BinOp::Mul, copy(rem), cu(4));
            bin(e, Ty::Usize, BinOp::Add, copy(half), copy(quad))
        }
    }
}

/// `bits` through `encoding` as an `i32`, before any f32 cast: the intermediate
/// `poot_quant::decode::Raw::integer` computes for `PlanarPlan::emit_value`'s `Zero`-subtracting
/// `ValueMap::Integer` branch. Every `Codes`/`Zero` operand of GPTQ and AWQ is `Unsigned` (`format
/// tests`, `planar.rs`'s `GPTQ_CODES`/`GPTQ_ZERO`/`AWQ_OPERANDS`); the one `Signed` planar operand
/// (GPTQ's act-order `GroupIndex`) is a group index, not a `Codes`/`Zero` value, and is read by
/// `PlanarPlan::emit_group_of` instead, never through here.
fn emit_encoded_integer(e: &mut Emit, encoding: FieldEncoding, bits: Local) -> Local {
    let FieldEncoding::Unsigned { offset } = encoding else {
        unreachable!(
            "format tests: a planar Codes/Zero operand is always Unsigned (Signed is GroupIndex-only, Float is the ValueMap::Float branch)"
        )
    };
    let signed = e.let_(
        Ty::I32,
        Rvalue::Bitcast {
            to: Ty::I32,
            operand: copy(bits),
        },
    );
    bin(
        e,
        Ty::I32,
        BinOp::Add,
        copy(signed),
        poot_kernel_ir::Operand::Const(poot_kernel_ir::Constant::I32(offset)),
    )
}
