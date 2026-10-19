//! Where a driver's model runs: the typed [`Placement`] the caller states, and the layer-pipeline
//! balance a caller can derive a cut from.
//!
//! Partition policy is the caller's, never the planner's (POOT-944): `compile_staged` takes the
//! [`DevicePlacement`] a [`Placed`] builds and plans the cut it names. A placement names *how* the
//! executor's devices are used, never which: the devices are the executor's target set, in rank order.
//! The balance is a host-only dynamic program over per-layer weight bytes; nothing here opens a device.
//!
//! Rebuilt from the retired exact-partition core's objective (read from history, not restored): the
//! global dense owners count in the balance. Device 0's block pays the embedding and the last
//! device's block pays the final norm and head, so the cut minimizes the largest per-device cost
//! rather than the largest layer block.
//!
//! The model's per-layer byte sizes are not on `ModelConfig`; [`LayerBytes::of`] reads them from the
//! model's weight map and store.

use std::sync::Arc;

use poot_graph_plan::multi_device::{DeviceSpec, Topology, TopologyError, validate_topology};
use poot_graph_plan::{
    ShardRule, multi_device::topology::DeviceId, staged::DevicePlacement, staged::ExpertPlacement,
    staged::Partition, staged::TargetSet,
};
use poot_quant::weights::{WeightId, WeightMap, WeightRole, WeightView};

use crate::driver::ModelHandle;
use crate::driver::error::{DriverError, InvalidOptions};

/// How a driver uses its executor's devices.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Placement {
    /// The whole model on the executor's one device.
    Single,
    /// A layer pipeline: one stage per device, in rank order, cut after each of `after_layers`.
    Pipeline {
        /// The layers each stage ends after: `n` cuts make `n + 1` stages.
        after_layers: Vec<usize>,
    },
    /// Tensor parallelism: every device is one rank and runs every layer over its share of the weights,
    /// divided by `rule`.
    Tensor {
        /// How each weight is divided across ranks.
        rule: ShardRule,
    },
}

/// A [`Placement`] resolved against the devices an executor drives.
#[derive(Clone, Debug)]
pub(crate) struct Placed {
    targets: TargetSet,
    placement: Placement,
}

impl Placed {
    /// `placement` over `targets`, refused when the two disagree: a single placement needs one device, a
    /// pipeline one device per stage, a tensor placement at least one.
    pub(crate) fn resolve(targets: TargetSet, placement: Placement) -> Result<Self, DriverError> {
        let devices = targets.devices().len();
        match &placement {
            Placement::Single if devices != 1 => {
                return Err(InvalidOptions::Devices { devices }.into());
            }
            Placement::Pipeline { after_layers }
                if devices < 2 || after_layers.len() + 1 != devices =>
            {
                return Err(InvalidOptions::PipelineStages {
                    stages: after_layers.len() + 1,
                    devices,
                }
                .into());
            }
            Placement::Tensor { .. } if devices == 0 => {
                return Err(InvalidOptions::TensorWithoutDevices.into());
            }
            _ => {}
        }
        Ok(Self { targets, placement })
    }

    pub(crate) fn targets(&self) -> &TargetSet {
        &self.targets
    }

    pub(crate) fn placement(&self) -> &Placement {
        &self.placement
    }

    /// The devices, in rank order.
    pub(crate) fn devices(&self) -> Vec<DeviceId> {
        self.targets.devices().iter().map(|&(id, _)| id).collect()
    }

    /// Whether the placement divides the weights over ranks.
    pub(crate) fn is_tensor(&self) -> bool {
        matches!(self.placement, Placement::Tensor { .. })
    }

    /// The partition `compile_staged` plans: the placement over these devices, with `weights` naming the
    /// roles a tensor placement divides by.
    pub(crate) fn partition(
        &self,
        experts: ExpertPlacement,
        weights: &Arc<WeightMap>,
    ) -> Partition {
        let devices = self.devices();
        Partition {
            experts,
            devices: match &self.placement {
                Placement::Single => DevicePlacement::Single(devices[0]),
                Placement::Pipeline { after_layers } => DevicePlacement::Pipeline {
                    after_layers: after_layers.clone(),
                    devices,
                },
                Placement::Tensor { rule } => DevicePlacement::Tensor {
                    rule: *rule,
                    devices,
                    weights: Arc::clone(weights),
                },
            },
        }
    }
}

/// The bytes the balance sees: one entry per layer, plus the owners fixed to the first and last device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerBytes {
    /// Weight bytes of each layer, in layer order.
    pub layers: Vec<u64>,
    /// Bytes fixed to the first device (the embedding).
    pub first: u64,
    /// Bytes fixed to the last device (the final norm and the head).
    pub last: u64,
}

impl LayerBytes {
    /// The stored bytes of every weight `handle`'s model maps, by the layer that owns it: a layer weight
    /// is its layer's, the head and the final norm are the last device's, every other model-level weight
    /// the first's. A weight several views share counts once per view, so the figure is an upper bound.
    pub fn of(handle: &ModelHandle) -> Self {
        let (weights, store) = (handle.model().weights(), handle.shared_store());
        let mut layers: Vec<u64> = Vec::new();
        let (mut first, mut last) = (0u64, 0u64);
        for (id, view, _) in weights.iter() {
            let keys = match view {
                WeightView::Stored(key)
                | WeightView::IndexNarrow(key)
                | WeightView::RowRange { key, .. }
                | WeightView::BlockColumns { key, .. } => std::slice::from_ref(key),
                WeightView::RowStack(keys) | WeightView::ExpertStack(keys) => keys.as_slice(),
            };
            let bytes: u64 = keys
                .iter()
                .filter_map(|key| store.get(key.as_str()))
                .map(|entry| entry.stored_bytes() as u64)
                .sum();
            match owner_of(id) {
                Owner::Layer(layer) => {
                    if layers.len() <= layer {
                        layers.resize(layer + 1, 0);
                    }
                    layers[layer] = layers[layer].saturating_add(bytes);
                }
                Owner::First => first = first.saturating_add(bytes),
                Owner::Last => last = last.saturating_add(bytes),
            }
        }
        Self {
            layers,
            first,
            last,
        }
    }
}

enum Owner {
    Layer(usize),
    First,
    Last,
}

fn owner_of(id: WeightId) -> Owner {
    match (id.layer, id.role) {
        (Some(layer), _) => Owner::Layer(layer),
        (None, WeightRole::Head | WeightRole::FinalNorm | WeightRole::FinalNormBias) => Owner::Last,
        (None, _) => Owner::First,
    }
}

/// Why no balanced pipeline exists.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum BalanceError {
    /// No device to pipeline over.
    #[error("a layer pipeline needs at least one device")]
    NoDevices,
    /// More devices than layers: some device would run no layer.
    #[error("{devices} devices cannot each run a layer of a {layers}-layer model")]
    MoreDevicesThanLayers {
        /// Devices offered.
        devices: usize,
        /// Layers the model has.
        layers: usize,
    },
    /// The device topology is invalid.
    #[error(transparent)]
    Devices(#[from] TopologyError),
    /// The layer byte arithmetic overflowed `u64`.
    #[error("layer byte arithmetic overflowed u64")]
    Overflow,
    /// A device's contiguous block of layers needs more bytes than the device has usable.
    #[error(
        "device {device:?} would hold {required} bytes of weights, over its {usable} usable bytes"
    )]
    DeviceTooSmall {
        /// The device that is over.
        device: DeviceId,
        /// Bytes its weights need.
        required: u64,
        /// Bytes it can use.
        usable: u64,
    },
}

/// Split `bytes.layers` into one contiguous, non-empty block per device, in `devices` order, so the
/// largest per-device weight cost is minimal. Ties keep the earliest cuts, so the result is
/// deterministic. Each device's weight cost must fit its `usable_bytes`.
pub fn balance_layers(
    bytes: &LayerBytes,
    devices: &[DeviceSpec],
) -> Result<DevicePlacement, BalanceError> {
    let count = devices.len();
    let n = bytes.layers.len();
    if count == 0 {
        return Err(BalanceError::NoDevices);
    }
    if count > n {
        return Err(BalanceError::MoreDevicesThanLayers {
            devices: count,
            layers: n,
        });
    }
    validate_topology(&Topology {
        devices: devices.to_vec(),
        links: Vec::new(),
    })?;
    let mut prefix = vec![0u64; n + 1];
    for (index, &layer) in bytes.layers.iter().enumerate() {
        prefix[index + 1] = prefix[index]
            .checked_add(layer)
            .ok_or(BalanceError::Overflow)?;
    }
    let cost = |device: usize, from: usize, to: usize| -> Result<u64, BalanceError> {
        let mut total = prefix[to] - prefix[from];
        if device == 0 {
            total = total
                .checked_add(bytes.first)
                .ok_or(BalanceError::Overflow)?;
        }
        if device + 1 == count {
            total = total
                .checked_add(bytes.last)
                .ok_or(BalanceError::Overflow)?;
        }
        Ok(total)
    };

    // best[d][i]: the least largest-device cost of the first `i` layers on devices `0..=d`, each
    // block non-empty; cut[d][i] is where device `d`'s block starts.
    const NONE: u64 = u64::MAX;
    let mut best = vec![vec![NONE; n + 1]; count];
    let mut cut = vec![vec![0usize; n + 1]; count];
    for (i, slot) in best[0].iter_mut().enumerate().skip(1) {
        *slot = cost(0, 0, i)?;
    }
    for d in 1..count {
        for i in (d + 1)..=n {
            for j in d..i {
                if best[d - 1][j] == NONE {
                    continue;
                }
                let load = best[d - 1][j].max(cost(d, j, i)?);
                if load < best[d][i] {
                    best[d][i] = load;
                    cut[d][i] = j;
                }
            }
        }
    }

    let mut after_layers = Vec::with_capacity(count - 1);
    let mut end = n;
    for d in (1..count).rev() {
        let start = cut[d][end];
        after_layers.push(start - 1);
        let load = cost(d, start, end)?;
        if load > devices[d].usable_bytes {
            return Err(BalanceError::DeviceTooSmall {
                device: devices[d].id,
                required: load,
                usable: devices[d].usable_bytes,
            });
        }
        end = start;
    }
    let first = cost(0, 0, end)?;
    if first > devices[0].usable_bytes {
        return Err(BalanceError::DeviceTooSmall {
            device: devices[0].id,
            required: first,
            usable: devices[0].usable_bytes,
        });
    }
    after_layers.reverse();
    Ok(DevicePlacement::Pipeline {
        after_layers,
        devices: devices.iter().map(|device| device.id).collect(),
    })
}

#[cfg(test)]
mod tests {
    use poot_target::Backend;

    use super::*;

    fn devices(usable: &[u64]) -> Vec<DeviceSpec> {
        usable
            .iter()
            .enumerate()
            .map(|(index, &usable_bytes)| DeviceSpec {
                id: DeviceId(index as u32),
                target: Backend::Nvptx.into(),
                usable_bytes,
            })
            .collect()
    }

    fn bytes(layers: &[u64], first: u64, last: u64) -> LayerBytes {
        LayerBytes {
            layers: layers.to_vec(),
            first,
            last,
        }
    }

    fn cuts(placement: DevicePlacement) -> Vec<usize> {
        match placement {
            DevicePlacement::Pipeline { after_layers, .. } => after_layers,
            other => panic!("expected a pipeline, got {other:?}"),
        }
    }

    /// The fixed owners count in the balance: with a heavy embedding and head, the end devices take
    /// fewer layers than an even split would give them.
    #[test]
    fn the_global_owners_count_in_the_balance() {
        let layers = [10u64; 8];
        let even = balance_layers(&bytes(&layers, 0, 0), &devices(&[1000; 2])).unwrap();
        assert_eq!(cuts(even), [3], "equal layers split evenly");
        let heavy = balance_layers(&bytes(&layers, 40, 0), &devices(&[1000; 2])).unwrap();
        // Device 0 pays 40 for the embedding: 2 layers (60) against 6 layers (60).
        assert_eq!(cuts(heavy), [1]);
    }

    /// The result is the optimum: every contiguous split of a small irregular model is no better.
    #[test]
    fn the_cut_minimizes_the_largest_device_cost() {
        let layers = [7u64, 1, 9, 4, 4, 12, 2];
        let spec = devices(&[u64::MAX; 3]);
        let placement = balance_layers(&bytes(&layers, 5, 3), &spec).unwrap();
        let chosen = cuts(placement);
        let cost = |cuts: &[usize]| -> u64 {
            let mut edges = vec![0];
            edges.extend(cuts.iter().map(|c| c + 1));
            edges.push(layers.len());
            edges
                .windows(2)
                .enumerate()
                .map(|(d, w)| {
                    let mut c: u64 = layers[w[0]..w[1]].iter().sum();
                    if d == 0 {
                        c += 5;
                    }
                    if d == 2 {
                        c += 3;
                    }
                    c
                })
                .max()
                .unwrap()
        };
        let optimum = (0..layers.len() - 2)
            .flat_map(|a| (a + 1..layers.len() - 1).map(move |b| [a, b]))
            .map(|c| cost(&c))
            .min()
            .unwrap();
        assert_eq!(cost(&chosen), optimum);
    }

    #[test]
    fn a_device_over_its_budget_or_an_impossible_pipeline_is_refused() {
        let layers = bytes(&[10, 10, 10, 10], 0, 0);
        assert_eq!(
            balance_layers(&layers, &devices(&[1000, 15])),
            Err(BalanceError::DeviceTooSmall {
                device: DeviceId(1),
                required: 20,
                usable: 15
            })
        );
        assert_eq!(balance_layers(&layers, &[]), Err(BalanceError::NoDevices));
        assert_eq!(
            balance_layers(&layers, &devices(&[1000; 5])),
            Err(BalanceError::MoreDevicesThanLayers {
                devices: 5,
                layers: 4
            })
        );
    }
}
