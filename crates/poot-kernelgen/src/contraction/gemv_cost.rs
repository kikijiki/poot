//! What one trip of a [`super::Schedule::Gemv`]'s `K` loop costs a lane, read off the body the generator emits
//! (POOT-1550 D4, POOT-1562): the planner scores a launch geometry from these facts of the kernel it would run, so a
//! format's decode never reaches a table. The counts are structural (statements and loads of the emitted loop), not a
//! measurement; the device's rates that turn them into time are the planner's calibration.

use poot_kernel_ir::{Body, Operand, Rvalue, Statement, Terminator, Ty};

use super::GLOBAL_REQUEST_BYTES;

/// What a load of the trip reads, which sets how the warp's lanes spread over cache lines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Stream {
    /// The activation row: every column group of a warp that shares a block reads the same elements, one lane per
    /// run.
    Activation,
    /// A weight source: a lane's run lies one run stride from its neighbour's, a column group per weight row.
    Weight,
    /// An address every lane shares (the metadata, a constant table word): one line.
    Uniform,
}

/// One load of a trip: what it reads and the bytes one lane takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TripLoad {
    stream: Stream,
    bytes: u32,
}

/// The lanes of one warp over the columns of a workgroup: how many lanes read one weight row, and how many lanes
/// issue together.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneMap {
    /// Lanes that issue one instruction together.
    pub warp_lanes: u32,
    /// Lanes that share one output column (`width / cols` of the launch): adjacent runs of one weight row.
    pub lanes_per_col: u32,
}

/// The cost of one trip of a Gemv's `K` loop on one lane, before any device rate weighs it.
#[derive(Clone, Debug, PartialEq)]
pub struct GemvBodyCost {
    /// The instructions a warp issues per trip: every statement of the loop that executes (a register move, a
    /// load, an arithmetic op, a branch), and nothing for a statement that only names storage or picks a vector lane.
    pub warp_instructions: u32,
    loads: Vec<TripLoad>,
    /// The bytes between the starts of two adjacent lanes' weight runs.
    weight_stride_bytes: f64,
    /// The bytes between the starts of two adjacent lanes' activation runs.
    activation_stride_bytes: f64,
}

impl GemvBodyCost {
    /// The cost of the `K` loop of `body`, a Gemv (parameters `(unit, activation, sources.., metadata, out)`): its first
    /// back edge closes the loop, and the blocks from its head to that edge are one trip (the guard, the prefetch of a
    /// deeper pipeline, the decode and the sums). The element-by-element tail loop that follows is not a trip.
    /// `weight_stride_bytes` and `activation_stride_bytes` are what adjacent lanes' runs are apart in memory.
    pub(crate) fn of_trip(
        body: &Body,
        weight_stride_bytes: f64,
        activation_stride_bytes: f64,
    ) -> Option<Self> {
        let trip =
            body.blocks
                .iter()
                .enumerate()
                .find_map(|(index, block)| match block.terminator {
                    Terminator::Goto { target } if target.index as usize <= index => {
                        Some(target.index as usize..=index)
                    }
                    _ => None,
                })?;
        let metadata = body.param_count - 1;
        let stream_of = |local: u32| match local {
            1 => Stream::Activation,
            local if local == metadata => Stream::Uniform,
            _ => Stream::Weight,
        };
        let mut warp_instructions = 0;
        let mut loads = Vec::new();
        for index in trip {
            let block = &body.blocks[index];
            for statement in &block.statements {
                let Statement::Assign(dest, rvalue) = statement else {
                    continue;
                };
                let (instructions, load) = instructions_of(body, dest, rvalue, stream_of);
                warp_instructions += instructions;
                loads.extend(load);
            }
            // A guard branches; the back edge jumps. A forward `Goto` falls through.
            warp_instructions += match block.terminator {
                Terminator::Goto { target } if (target.index as usize) > index => 0,
                _ => 1,
            };
        }
        Some(Self {
            warp_instructions,
            loads,
            weight_stride_bytes,
            activation_stride_bytes,
        })
    }

    /// The cache lines the warp's loads of one trip touch under `lanes`.
    ///
    /// A load of `bytes` per lane at a lane stride of `stride` bytes spans `(lanes - 1) * stride + bytes` bytes over the
    /// lanes of a column group, and no more lines than one per lane and per line of its own bytes. A weight load repeats
    /// that for each column group of the warp (each its own row); an activation load is one group's, the groups
    /// sharing the row; a uniform load is one line.
    pub fn lines_per_trip(&self, lanes: LaneMap) -> f64 {
        let line = GLOBAL_REQUEST_BYTES as f64;
        let group = f64::from(lanes.lanes_per_col.clamp(1, lanes.warp_lanes.max(1)));
        let groups = (f64::from(lanes.warp_lanes.max(1)) / group).ceil();
        let group_lines = |stride: f64, bytes: f64| {
            let span = (group - 1.0) * stride.max(0.0) + bytes;
            (span / line).ceil().min(group * (bytes / line).ceil())
        };
        self.loads
            .iter()
            .map(|load| {
                let bytes = f64::from(load.bytes);
                match load.stream {
                    Stream::Activation => group_lines(self.activation_stride_bytes, bytes),
                    Stream::Weight => groups * group_lines(self.weight_stride_bytes, bytes),
                    Stream::Uniform => 1.0,
                }
            })
            .sum()
    }
}

/// The instructions `rvalue` issues and the load it makes, if any. A move between registers issues (the pipeline's
/// rotation is one per carried word); a constant, a lane pick of a vector the load already filled and a bit
/// reinterpretation of a register do not.
fn instructions_of(
    body: &Body,
    dest: &poot_kernel_ir::Place,
    rvalue: &Rvalue,
    stream_of: impl Fn(u32) -> Stream,
) -> (u32, Option<TripLoad>) {
    let load = |place: &poot_kernel_ir::Place, bytes: u32| TripLoad {
        stream: stream_of(place.local.index),
        bytes,
    };
    match rvalue {
        Rvalue::Use(Operand::Const(_)) | Rvalue::VectorExtract { .. } | Rvalue::Bitcast { .. } => {
            (0, None)
        }
        Rvalue::Use(Operand::Copy(place) | Operand::Move(place)) if place.projection.is_empty() => {
            (1, None)
        }
        Rvalue::Use(Operand::Copy(place) | Operand::Move(place)) => {
            let bytes = element_bytes(&body.locals[place.local.index as usize].ty);
            (1, Some(load(place, bytes)))
        }
        // The loaded vector is the destination's own type.
        Rvalue::VectorLoad { place, .. } => {
            let bytes = element_bytes(&body.locals[dest.local.index as usize].ty);
            (1, Some(load(place, bytes)))
        }
        Rvalue::ConstantTableRead { .. } => (
            1,
            Some(TripLoad {
                stream: Stream::Uniform,
                bytes: 4,
            }),
        ),
        _ => (1, None),
    }
}

/// The bytes of one element of the slice parameter type `ty`, or of the whole vector `ty` names.
fn element_bytes(ty: &Ty) -> u32 {
    match ty {
        Ty::Ref { pointee, .. } | Ty::Slice(pointee) => element_bytes(pointee),
        Ty::Vec { elem, lanes } => element_bytes(elem) * lanes,
        Ty::Bool => 1,
        Ty::F16 | Ty::BF16 => 2,
        Ty::F64 | Ty::Usize => 8,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packed_gemv_body_cost;
    use crate::{GemvActivation, GemvRows, LoadDepth, Schedule};
    use poot_quant::format::WeightFormat;
    use poot_target::{AmdArch, Backend};

    fn schedule(run: u32, rows: u32, depth: u32) -> Schedule {
        Schedule::Gemv {
            width: 256,
            cols: 4,
            unroll: run,
            activation: GemvActivation::Vec4,
            rows: GemvRows::new(rows).unwrap(),
            depth: LoadDepth::new(depth).unwrap(),
        }
    }

    fn cost(format: WeightFormat, run: u32, rows: u32, depth: u32) -> GemvBodyCost {
        let caps =
            poot_test_util::device_caps::default_caps_for(Backend::AmdGcn(AmdArch::gfx1151()));
        packed_gemv_body_cost(format, schedule(run, rows, depth), &caps).unwrap()
    }

    fn count(cost: &GemvBodyCost, stream: Stream) -> usize {
        cost.loads
            .iter()
            .filter(|load| load.stream == stream)
            .count()
    }

    /// A second row per lane repeats the weight loads and decode of the first over the same activation read, so the trip
    /// issues more instructions and weight loads and exactly the activation loads it did; a vector activation read of a
    /// 16-value run is four vector loads. Mutation: take the last back edge of the body, the element-by-element tail,
    /// as the trip; the activation loads become one scalar.
    #[test]
    fn a_second_row_repeats_the_weight_loads_but_not_the_activation_read() {
        let one = cost(WeightFormat::Q4_K, 16, 1, 1);
        let two = cost(WeightFormat::Q4_K, 16, 2, 1);
        assert_eq!(count(&one, Stream::Activation), 4);
        assert_eq!(count(&two, Stream::Activation), 4);
        assert_eq!(count(&two, Stream::Weight), 2 * count(&one, Stream::Weight));
        assert!(
            two.warp_instructions > one.warp_instructions + one.warp_instructions / 2,
            "{} vs {}",
            two.warp_instructions,
            one.warp_instructions
        );
    }

    /// A deeper pipeline issues the same loads per trip (the prefetch loads the run `depth - 1` trips ahead in place
    /// of this trip's) and adds the rotation of its carried state to the trip's instructions.
    #[test]
    fn a_deeper_pipeline_rotates_its_carried_runs_each_trip() {
        let one = cost(WeightFormat::Q4_K, 16, 1, 1);
        let two = cost(WeightFormat::Q4_K, 16, 1, 2);
        assert_eq!(count(&one, Stream::Weight), count(&two, Stream::Weight));
        assert!(two.warp_instructions > one.warp_instructions);
    }

    /// The warp's lines follow its lane map. A lane per run of 64 bytes spreads the activation read of 32 lanes over 16
    /// lines per load; two column groups of 16 lanes read the same activation elements, so the warp touches half of
    /// them and adds only the second row's few weight lines; a lane per row shares nothing and touches a line per
    /// weight load of every lane.
    #[test]
    fn a_warps_lines_follow_how_its_lanes_share_rows() {
        let cost = cost(WeightFormat::Q4_K, 16, 1, 1);
        let lines = |lanes_per_col| {
            cost.lines_per_trip(LaneMap {
                warp_lanes: 32,
                lanes_per_col,
            })
        };
        let (one_row, two_rows, a_row_per_lane) = (lines(32), lines(16), lines(1));
        assert!(two_rows < one_row, "{two_rows} vs {one_row}");
        assert!(a_row_per_lane > one_row, "{a_row_per_lane} vs {one_row}");
    }
}
