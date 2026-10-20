//! The LoRA registry (POOT-735, POOT-ADR-0112, POOT-579): the immutable adapter-set generations the
//! driver serves, the stacked adapter weights each generation's executable loads beside the base
//! weights, and the per-row adapter index each request binds.
//!
//! A *set* is what one prepared entry is compiled for: logical adapter names and factor shapes, plus the
//! immutable identity of the source that produced the weights. Two sets with the same names and shapes
//! but different sources are different generations, so a reloaded adapter never reuses an entry (or a
//! cached prefix block) built for the old weights. Inside one generation a request picks its adapter by
//! [`LoraIdx`], a bound input that never specializes an entry.
//!
//! The graph transform that inserts the adapters is `poot_graph_plan`'s LoRA pass, applied by `compile`
//! when an entry is prepared for a generation ([`AdapterStack::plan`]). The generation's weights are a
//! small adapter store built beside the base store ([`AdapterStack::weights`]): the base entries are
//! shared with their generations intact, so loading a generation uploads only the adapter tensors.

use std::collections::{BTreeSet, HashMap};
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};
use std::sync::Arc;

use poot_graph_plan::LoraPlan;
use poot_load::lora::{LoraAdapter, LoraWeight};
use poot_quant::StoredBytes;
use poot_quant::weights::{
    DenseWeight, DenseWeightError, DuplicateWeightKey, WeightEntry, WeightKey, WeightStore,
};
use poot_tensor::DType;
use poot_weights::{
    LoraRole, LoraTarget, WeightId, WeightMap, WeightMapError, WeightRole, WeightView,
};

use super::prefix_cache::PrefixIdentity;

/// The immutable identity of the files (or the loader incarnation) an adapter set was read from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AdapterSource(pub NonZeroU64);

/// One adapter of a set: its logical name and the factor shape its entry is compiled for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdapterSpec {
    /// The adapter's logical name.
    pub name: String,
    /// The adapter's factor shape.
    pub shape: Vec<usize>,
}

/// An adapter set as captured: what a generation is keyed on.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AdapterSet {
    /// The adapters of the set, in index order: `LoraIdx` 1 names the first.
    pub adapters: Vec<AdapterSpec>,
    /// The identity of what the weights were read from.
    pub source: AdapterSource,
}

/// One admitted adapter-set generation. Driver-issued and never reused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GenerationId(NonZeroU32);

#[cfg(test)]
pub(crate) mod test_only {
    use super::*;

    /// A factor as an adapter file stores it: `values` row-major in `shape`, F32.
    pub(crate) fn f32_factor(shape: Vec<usize>, values: &[f32]) -> DenseWeight {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        DenseWeight::try_new(DType::F32, shape, StoredBytes::from(bytes)).unwrap()
    }

    impl GenerationId {
        /// A generation id no registry issued, for a test of a refusal that comes before the registry is
        /// asked.
        pub(crate) fn unissued(id: u32) -> Self {
            Self(NonZeroU32::new(id).expect("an id is at least one"))
        }
    }
}

/// A request's adapter within a generation, 1-based: 0 is the base weights (`NO_ADAPTER`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LoraIdx(NonZeroU32);

impl LoraIdx {
    /// The reference for `index`, or `None` for 0, which is the base weights.
    pub fn new(index: u32) -> Option<Self> {
        NonZeroU32::new(index).map(Self)
    }

    /// The index as an integer, at least 1.
    pub fn get(self) -> u32 {
        self.0.get()
    }
}

/// Which weights a sequence runs under.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum AdapterRef {
    /// The base model's weights, no adapter.
    #[default]
    Base,
    /// One adapter of a loaded generation.
    Adapter {
        /// The adapter-set generation the adapter belongs to.
        generation: GenerationId,
        /// The adapter's index within the generation.
        index: LoraIdx,
    },
}

impl AdapterRef {
    /// The generation this reference holds, if any.
    pub fn generation(self) -> Option<GenerationId> {
        match self {
            AdapterRef::Base => None,
            AdapterRef::Adapter { generation, .. } => Some(generation),
        }
    }

    /// The effective-weights identity the prefix cache keys this selection's blocks on: the base
    /// model's, or its adapter incarnation's.
    pub fn prefix_identity(self) -> PrefixIdentity {
        match self {
            AdapterRef::Base => PrefixIdentity::BASE,
            AdapterRef::Adapter { generation, index } => prefix_identity(generation, index.get()),
        }
    }

    /// The row value bound to `Slot::LoraIdx`: 0 for the base weights.
    pub(crate) fn row_value(self) -> i32 {
        match self {
            AdapterRef::Base => 0,
            AdapterRef::Adapter { index, .. } => {
                i32::try_from(index.get()).expect("an adapter index is below i32::MAX")
            }
        }
    }
}

/// Why the registry refuses an adapter operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LoraError {
    /// No loaded generation has this id.
    #[error("adapter generation {0:?} is not loaded")]
    UnknownGeneration(GenerationId),
    /// The adapter index is beyond the generation's adapters.
    #[error("generation {generation:?} holds {adapters} adapters; index {index} is out of range")]
    IndexOutOfRange {
        /// The generation named.
        generation: GenerationId,
        /// The index asked for.
        index: u32,
        /// Adapters the generation holds.
        adapters: usize,
    },
    /// A live request holds the generation: unloading it would free weights a sequence still runs.
    #[error("generation {generation:?} is held by {live} live sequences")]
    InUse {
        /// The generation named.
        generation: GenerationId,
        /// Live sequences holding it.
        live: usize,
    },
    /// The generation ids are all used up.
    #[error("the registry has issued every generation id")]
    Exhausted,
}

#[derive(Debug)]
struct Generation {
    set: AdapterSet,
    /// The generation's stacked adapter tensors, shared with the executable that loads them.
    stack: Arc<AdapterStack>,
    /// Open sequences holding this generation.
    live: usize,
}

/// The loaded adapter-set generations.
#[derive(Debug, Default)]
pub(crate) struct LoraRegistry {
    generations: HashMap<GenerationId, Generation>,
    by_set: HashMap<AdapterSet, GenerationId>,
    issued: u32,
}

impl LoraRegistry {
    /// Admit `set`: the generation already holding an identical set (same names, shapes and source), or
    /// a new one. A set that differs in source only is a new generation.
    pub(crate) fn capture(
        &mut self,
        set: AdapterSet,
        stack: AdapterStack,
    ) -> Result<GenerationId, LoraError> {
        if let Some(&id) = self.by_set.get(&set) {
            return Ok(id);
        }
        let issued = self.issued.checked_add(1).ok_or(LoraError::Exhausted)?;
        self.issued = issued;
        let id = GenerationId(NonZeroU32::new(issued).expect("the counter starts at one"));
        self.by_set.insert(set.clone(), id);
        self.generations.insert(
            id,
            Generation {
                set,
                stack: Arc::new(stack),
                live: 0,
            },
        );
        Ok(id)
    }

    /// The stacked adapter tensors of `generation`.
    pub(crate) fn stack(&self, generation: GenerationId) -> Option<Arc<AdapterStack>> {
        self.generations
            .get(&generation)
            .map(|held| Arc::clone(&held.stack))
    }

    pub(crate) fn contains(&self, generation: GenerationId) -> bool {
        self.generations.contains_key(&generation)
    }

    /// A sequence takes `adapter`: validates it and counts the hold. Returns the effective-weights
    /// identity the prefix cache keys that sequence's blocks on.
    pub(crate) fn retain(&mut self, adapter: AdapterRef) -> Result<PrefixIdentity, LoraError> {
        let AdapterRef::Adapter { generation, index } = adapter else {
            return Ok(PrefixIdentity::BASE);
        };
        let held = self
            .generations
            .get_mut(&generation)
            .ok_or(LoraError::UnknownGeneration(generation))?;
        if index.get() as usize > held.set.adapters.len() {
            return Err(LoraError::IndexOutOfRange {
                generation,
                index: index.get(),
                adapters: held.set.adapters.len(),
            });
        }
        held.live += 1;
        Ok(prefix_identity(generation, index.get()))
    }

    /// A sequence gives `adapter` back.
    pub(crate) fn release(&mut self, adapter: AdapterRef) {
        if let Some(generation) = adapter.generation() {
            let held = self
                .generations
                .get_mut(&generation)
                .expect("a retained generation stays loaded while held");
            held.live -= 1;
        }
    }

    /// `generation` can be dropped: it is loaded and no open sequence holds it.
    pub(crate) fn check_unloadable(&self, generation: GenerationId) -> Result<(), LoraError> {
        let held = self
            .generations
            .get(&generation)
            .ok_or(LoraError::UnknownGeneration(generation))?;
        if held.live > 0 {
            return Err(LoraError::InUse {
                generation,
                live: held.live,
            });
        }
        Ok(())
    }

    /// The prefix identities of every adapter of `generation`.
    pub(crate) fn identities_of(&self, generation: GenerationId) -> Vec<PrefixIdentity> {
        let adapters = self
            .generations
            .get(&generation)
            .map_or(0, |held| held.set.adapters.len());
        (1..=adapters as u32)
            .map(|index| prefix_identity(generation, index))
            .collect()
    }

    /// Drop `generation`. Refused while any open sequence holds it.
    pub(crate) fn unload(&mut self, generation: GenerationId) -> Result<(), LoraError> {
        self.check_unloadable(generation)?;
        let held = self.generations.remove(&generation).expect("checked above");
        self.by_set.remove(&held.set);
        Ok(())
    }
}

/// The effective-weights identity the prefix cache keys `index` of `generation` on.
fn prefix_identity(generation: GenerationId, index: u32) -> PrefixIdentity {
    PrefixIdentity::adapter(u64::from(generation.0.get()) << 32 | u64::from(index))
}

/// Why a set of adapters cannot be stacked for the model they are captured against.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum AdapterError {
    /// The set holds no adapter.
    #[error("an adapter set holds no adapter")]
    Empty,
    /// A module path the adapter targets is not a projection this build can adapt.
    #[error("adapter module `{module}` is not a targetable projection")]
    UnknownModule {
        /// The module path as the adapter named it.
        module: String,
    },
    /// The model has no weight for a projection the adapter targets.
    #[error("adapter `{adapter}` targets layer {layer} {target:?}, which the model does not have")]
    NoWeight {
        /// Index of the adapter in the set.
        adapter: usize,
        /// The layer the module belongs to.
        layer: usize,
        /// The projection the adapter targets.
        target: LoraTarget,
    },
    /// An adapter's factors do not fit the base weight they correct.
    #[error(
        "adapter {adapter} layer {layer} {target:?}: factors {a:?} and {b:?} do not fit a base weight of [{out}, {inner}]"
    )]
    Shape {
        /// Index of the adapter in the set.
        adapter: usize,
        /// The layer the module belongs to.
        layer: usize,
        /// The projection the adapter targets.
        target: LoraTarget,
        /// Shape of the A factor.
        a: Vec<usize>,
        /// Shape of the B factor.
        b: Vec<usize>,
        /// The base weight's output dimension.
        out: usize,
        /// The base weight's inner (input) dimension.
        inner: usize,
    },
    /// The adapter weight map was malformed.
    #[error(transparent)]
    Weights(#[from] WeightMapError),
    /// An adapter holds the same projection twice, under two module paths.
    #[error("adapter {adapter} names layer {layer} {target:?} twice")]
    DuplicateModule {
        /// Index of the adapter in the set.
        adapter: usize,
        /// The layer the module belongs to.
        layer: usize,
        /// The projection named twice.
        target: LoraTarget,
    },
    /// Two factors of the set are stored in different dtypes: one stack holds one dtype.
    #[error("adapter {adapter}: a factor is {found:?} but the set's factors are {expected:?}")]
    MixedDtype {
        /// Index of the adapter in the set.
        adapter: usize,
        /// The dtype of the set's first factor.
        expected: DType,
        /// The dtype of the factor that differs.
        found: DType,
    },
    /// A stacked adapter tensor could not be built.
    #[error("a stacked adapter tensor: {0}")]
    Tensor(#[from] DenseWeightError),
    /// The generation's weight store holds a duplicate key.
    #[error("the generation's store: {0}")]
    Store(#[from] DuplicateWeightKey),
}

/// The projection a PEFT module path names: `model.layers.{layer}.self_attn.q_proj` and its siblings.
fn parse_module(module: &str) -> Option<(usize, LoraTarget)> {
    let rest = module.strip_prefix("model.layers.")?;
    let (layer, projection) = rest.split_once('.')?;
    let target = match projection {
        "self_attn.q_proj" => LoraTarget::AttnQ,
        "self_attn.k_proj" => LoraTarget::AttnK,
        "self_attn.v_proj" => LoraTarget::AttnV,
        "self_attn.o_proj" => LoraTarget::AttnO,
        "mlp.gate_proj" => LoraTarget::FfnGate,
        "mlp.up_proj" => LoraTarget::FfnUp,
        "mlp.down_proj" => LoraTarget::FfnDown,
        _ => return None,
    };
    Some((layer.parse().ok()?, target))
}

fn dense(dtype: DType, shape: Vec<usize>, bytes: Vec<u8>) -> Result<WeightEntry, DenseWeightError> {
    DenseWeight::try_new(dtype, shape, StoredBytes::from(bytes)).map(WeightEntry::Dense)
}

/// The dtype every factor of `adapters` is stored in, or [`AdapterError::MixedDtype`]. One stack
/// tensor holds one dtype, and a factor is never widened, so a set of adapters stored differently
/// cannot be stacked.
fn factor_dtype(adapters: &[LoraAdapter]) -> Result<DType, AdapterError> {
    let mut common: Option<DType> = None;
    for (slot, adapter) in adapters.iter().enumerate() {
        for weight in adapter.weights.values() {
            for found in [weight.a.dtype(), weight.b.dtype()] {
                let expected = *common.get_or_insert(found);
                if found != expected {
                    return Err(AdapterError::MixedDtype {
                        adapter: slot,
                        expected,
                        found,
                    });
                }
            }
        }
    }
    common.ok_or(AdapterError::Empty)
}

/// Copy the row-major `[rows, cols]` matrix `src` of `width`-byte elements into `dst` transposed:
/// element `(r, c)` lands at element index `c * dst_stride + r` of `dst`. This is the `x @ W` reading of a
/// PEFT `[out, in]` factor, taken on the stored bytes without decoding them.
fn copy_transposed(dst: &mut [u8], dst_stride: usize, src: &[u8], cols: usize, width: usize) {
    for (index, element) in src.chunks_exact(width).enumerate() {
        let (r, c) = (index / cols, index % cols);
        let at = (c * dst_stride + r) * width;
        dst[at..at + width].copy_from_slice(element);
    }
}

/// The adapters of one generation stacked as the LoRA pass reads them: per adapted projection of every
/// layer, `A [adapters + 1, in, r]` and `B [adapters + 1, r, out]` zero-padded to the largest rank, in the
/// dtype the adapters were stored in, and the model-level F32 `scale [adapters + 1]`; entry 0 is the
/// no-adapter row.
#[derive(Debug)]
pub(crate) struct AdapterStack {
    adapters: NonZeroUsize,
    rank: NonZeroUsize,
    /// The stored dtype of every stacked factor.
    dtype: DType,
    projections: Vec<LoraTarget>,
    entries: Vec<(WeightId, WeightEntry)>,
}

impl AdapterStack {
    /// Stack `adapters` against the base weights of `base`. Every projection any adapter targets is
    /// stacked for every layer that has it, so a layer an adapter leaves alone corrects by exactly zero.
    pub(crate) fn build(adapters: &[LoraAdapter], base: &WeightMap) -> Result<Self, AdapterError> {
        let count = NonZeroUsize::new(adapters.len()).ok_or(AdapterError::Empty)?;
        // Each adapter's factors by (layer, projection).
        let mut sites: Vec<HashMap<(usize, LoraTarget), &LoraWeight>> = Vec::new();
        for (slot, adapter) in adapters.iter().enumerate() {
            let mut mine = HashMap::new();
            for (module, weight) in &adapter.weights {
                let site = parse_module(module).ok_or_else(|| AdapterError::UnknownModule {
                    module: module.clone(),
                })?;
                if mine.insert(site, weight).is_some() {
                    return Err(AdapterError::DuplicateModule {
                        adapter: slot,
                        layer: site.0,
                        target: site.1,
                    });
                }
            }
            sites.push(mine);
        }
        let targets: BTreeSet<LoraTarget> = sites
            .iter()
            .flat_map(|mine| mine.keys().map(|&(_, target)| target))
            .collect();
        let rank = adapters
            .iter()
            .flat_map(|adapter| adapter.weights.values().map(|w| w.a.shape()[0]))
            .max()
            .and_then(NonZeroUsize::new)
            .ok_or(AdapterError::Empty)?;
        let dtype = factor_dtype(adapters)?;
        let width = dtype.byte_size();
        let r = rank.get();
        let stack = adapters.len() + 1;

        let mut scale = vec![0.0f32];
        scale.extend(adapters.iter().map(|adapter| adapter.config.scaling()));
        let scale_bytes: Vec<u8> = scale.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut entries = vec![(
            WeightId::model(WeightRole::Lora(LoraRole::Scale)),
            dense(DType::F32, vec![stack], scale_bytes)?,
        )];
        let mut covered: BTreeSet<(usize, LoraTarget)> = BTreeSet::new();
        for (id, _, handle) in base.iter() {
            let (Some(layer), Some(target)) = (id.layer, LoraTarget::of(id.role)) else {
                continue;
            };
            if !targets.contains(&target) {
                continue;
            }
            let &[out_dim, in_dim] = handle.shape.as_slice() else {
                continue;
            };
            let mut a = vec![0u8; stack * in_dim * r * width];
            let mut b = vec![0u8; stack * r * out_dim * width];
            for (slot, mine) in sites.iter().enumerate() {
                let Some(weight) = mine.get(&(layer, target)) else {
                    continue;
                };
                covered.insert((layer, target));
                let (a_shape, b_shape) = (weight.a.shape().to_vec(), weight.b.shape().to_vec());
                let rank_here = a_shape.first().copied().unwrap_or(0);
                if a_shape != [rank_here, in_dim] || b_shape != [out_dim, rank_here] {
                    return Err(AdapterError::Shape {
                        adapter: slot,
                        layer,
                        target,
                        a: a_shape,
                        b: b_shape,
                        out: out_dim,
                        inner: in_dim,
                    });
                }
                // `A [r, in]` lands as `[in, r]` at this adapter's entry, `B [out, r]` as `[r, out]`.
                let a_base = (slot + 1) * in_dim * r;
                copy_transposed(
                    &mut a[a_base * width..],
                    r,
                    weight.a.bytes().as_slice(),
                    in_dim,
                    width,
                );
                let b_base = (slot + 1) * r * out_dim;
                copy_transposed(
                    &mut b[b_base * width..],
                    out_dim,
                    weight.b.bytes().as_slice(),
                    rank_here,
                    width,
                );
            }
            entries.push((
                WeightId::layer(layer, WeightRole::Lora(LoraRole::A(target))),
                dense(dtype, vec![stack, in_dim, r], a)?,
            ));
            entries.push((
                WeightId::layer(layer, WeightRole::Lora(LoraRole::B(target))),
                dense(dtype, vec![stack, r, out_dim], b)?,
            ));
        }
        for (slot, mine) in sites.iter().enumerate() {
            if let Some(&(layer, target)) = mine.keys().find(|site| !covered.contains(site)) {
                return Err(AdapterError::NoWeight {
                    adapter: slot,
                    layer,
                    target,
                });
            }
        }
        Ok(Self {
            adapters: count,
            rank,
            dtype,
            projections: targets.into_iter().collect(),
            entries,
        })
    }

    /// The pass plan for an entry of `rows` rows.
    pub(crate) fn plan(&self, rows: NonZeroUsize) -> LoraPlan {
        LoraPlan {
            rows,
            adapters: self.adapters,
            rank: self.rank,
            factor_dtype: self.dtype,
            projections: self.projections.clone(),
        }
    }

    /// The store and weight map of an executable that serves this generation: `base` and its map, plus
    /// the stacked tensors. Base entries keep their generations, so an executor holding both uploads each
    /// base weight once.
    pub(crate) fn weights(
        &self,
        base: &WeightStore,
        base_map: &WeightMap,
    ) -> Result<(WeightStore, WeightMap), AdapterError> {
        let mut store = WeightStore::builder();
        store.extend_from(base)?;
        for (id, entry) in &self.entries {
            store.insert(id.const_name(), entry.clone())?;
        }
        let store = store.build();
        let mut map = base_map.extending(&store);
        for (id, _) in &self.entries {
            map.map(*id, WeightView::Stored(WeightKey::from(id.const_name())))?;
        }
        let map = map.build();
        Ok((store, map))
    }
}

#[cfg(test)]
mod tests {
    use poot_load::lora::LoraAdapterConfig;

    use super::test_only::f32_factor;
    use super::*;

    fn entry(shape: Vec<usize>) -> WeightEntry {
        let elements: usize = shape.iter().product();
        let bytes = vec![0.5f32; elements]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        dense(DType::F32, shape, bytes).unwrap()
    }

    /// A base model of two layers: `q` `[6, 4]` (out 6, in 4) in both, `v` `[3, 4]` in layer 0.
    fn base() -> (WeightStore, WeightMap) {
        let mut store = WeightStore::builder();
        store.insert("q0", entry(vec![6, 4])).unwrap();
        store.insert("q1", entry(vec![6, 4])).unwrap();
        store.insert("v0", entry(vec![3, 4])).unwrap();
        let store = store.build();
        let mut map = WeightMap::builder(&store);
        for (id, key) in [
            (WeightId::layer(0, LoraTarget::AttnQ.weight_role()), "q0"),
            (WeightId::layer(1, LoraTarget::AttnQ.weight_role()), "q1"),
            (WeightId::layer(0, LoraTarget::AttnV.weight_role()), "v0"),
        ] {
            map.map(id, WeightView::Stored(WeightKey::from(key)))
                .unwrap();
        }
        let map = map.build();
        (store, map)
    }

    /// `values` stored as `dtype` (F32 or BF16; the tests use values both represent exactly).
    fn factor_as(dtype: DType, shape: Vec<usize>, values: &[f32]) -> DenseWeight {
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| match dtype {
                DType::BF16 => (v.to_bits() >> 16).to_le_bytes()[..2].to_vec(),
                _ => v.to_le_bytes().to_vec(),
            })
            .collect();
        DenseWeight::try_new(dtype, shape, StoredBytes::from(bytes)).unwrap()
    }

    /// An adapter of `rank` over `modules` `(path, in, out)`, its factors in PEFT layout (`A [rank, in]`
    /// all `fill`, `B [out, rank]` all `fill + 1`) stored as `dtype`.
    fn adapter_as(
        dtype: DType,
        rank: usize,
        alpha: f32,
        modules: &[(&str, usize, usize)],
        fill: f32,
    ) -> LoraAdapter {
        LoraAdapter {
            config: LoraAdapterConfig {
                r: rank,
                lora_alpha: alpha,
                target_modules: Vec::new(),
                use_rslora: false,
            },
            weights: modules
                .iter()
                .map(|&(module, in_dim, out_dim)| {
                    (
                        module.to_string(),
                        LoraWeight {
                            a: factor_as(dtype, vec![rank, in_dim], &vec![fill; in_dim * rank]),
                            b: factor_as(
                                dtype,
                                vec![out_dim, rank],
                                &vec![fill + 1.0; rank * out_dim],
                            ),
                        },
                    )
                })
                .collect(),
        }
    }

    fn adapter(
        rank: usize,
        alpha: f32,
        modules: &[(&str, usize, usize)],
        fill: f32,
    ) -> LoraAdapter {
        adapter_as(DType::F32, rank, alpha, modules, fill)
    }

    fn entry_of(stack: &AdapterStack, id: WeightId) -> &DenseWeight {
        let (_, WeightEntry::Dense(dense)) = stack
            .entries
            .iter()
            .find(|(held, _)| *held == id)
            .unwrap_or_else(|| panic!("no entry for {id}"))
        else {
            panic!("{id} is not dense");
        };
        dense
    }

    /// The stacked entry `id` widened to f32 (F32 and BF16 entries).
    fn values(stack: &AdapterStack, id: WeightId) -> Vec<f32> {
        let dense = entry_of(stack, id);
        let bytes = dense.bytes().as_slice();
        match dense.dtype() {
            DType::F32 => bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&word| f32::from_le_bytes(word))
                .collect(),
            DType::BF16 => bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&pair| f32::from_bits(u32::from(u16::from_le_bytes(pair)) << 16))
                .collect(),
            other => panic!("{id} is {other:?}"),
        }
    }

    /// POOT-1258 SC-001: a BF16 adapter is stacked, and so uploaded, as BF16: the `A`/`B` entries the
    /// executor binds carry the BF16 dtype and half the bytes of the F32 twin, the plan declares BF16
    /// factors, and only the scale vector is F32. Mutation: restore the f32 decode (build the stack as F32
    /// whatever the adapter stored); the entries are F32 and the dtype assertions fail.
    #[test]
    fn a_bf16_adapter_is_stacked_and_planned_as_bf16() {
        let (store, map) = base();
        let modules = [("model.layers.0.self_attn.q_proj", 4, 6)];
        let a_id = WeightId::layer(0, WeightRole::Lora(LoraRole::A(LoraTarget::AttnQ)));
        let b_id = WeightId::layer(0, WeightRole::Lora(LoraRole::B(LoraTarget::AttnQ)));
        let scale_id = WeightId::model(WeightRole::Lora(LoraRole::Scale));

        let narrow =
            AdapterStack::build(&[adapter_as(DType::BF16, 2, 4.0, &modules, 1.0)], &map).unwrap();
        let wide =
            AdapterStack::build(&[adapter_as(DType::F32, 2, 4.0, &modules, 1.0)], &map).unwrap();
        for id in [a_id, b_id] {
            assert_eq!(entry_of(&narrow, id).dtype(), DType::BF16, "{id}");
            assert_eq!(
                entry_of(&narrow, id).bytes().len() * 2,
                entry_of(&wide, id).bytes().len(),
                "{id}: half the bytes of the F32 stack"
            );
        }
        assert_eq!(entry_of(&narrow, scale_id).dtype(), DType::F32);
        assert_eq!(
            narrow.plan(NonZeroUsize::MIN).factor_dtype,
            DType::BF16,
            "the program declares the factors as stored"
        );
        // The uploaded generation store holds the BF16 entries as built.
        let (extended, _) = narrow.weights(&store, &map).unwrap();
        let Some(WeightEntry::Dense(uploaded)) = extended.get(&a_id.const_name()) else {
            panic!("the generation's store holds the A stack");
        };
        assert_eq!(uploaded.dtype(), DType::BF16);
        assert_eq!(
            values(&narrow, a_id),
            values(&wide, a_id),
            "BF16 holds these values exactly"
        );
    }

    /// The stack reads PEFT's `A [r, in]` and `B [out, r]` as the `x @ W` factors `[in, r]` and `[r, out]`,
    /// element by element on the stored bytes: distinct values at every position, a rank-2 adapter
    /// padded into a rank-3 stack. Mutation: copy `A` without transposing it; position `(i, k)` holds
    /// `A[i, k]` instead of `A[k, i]`.
    #[test]
    fn peft_factors_are_read_transposed_into_the_stack() {
        let (_, map) = base();
        let (rank, in_dim, out_dim) = (2usize, 4usize, 6usize);
        // A[k, i] = 10k + i, B[o, k] = 100 + 10o + k.
        let a_values: Vec<f32> = (0..rank)
            .flat_map(|k| (0..in_dim).map(move |i| (10 * k + i) as f32))
            .collect();
        let b_values: Vec<f32> = (0..out_dim)
            .flat_map(|o| (0..rank).map(move |k| (100 + 10 * o + k) as f32))
            .collect();
        let narrow = LoraAdapter {
            config: LoraAdapterConfig {
                r: rank,
                lora_alpha: 2.0,
                target_modules: Vec::new(),
                use_rslora: false,
            },
            weights: [(
                "model.layers.0.self_attn.q_proj".to_string(),
                LoraWeight {
                    a: f32_factor(vec![rank, in_dim], &a_values),
                    b: f32_factor(vec![out_dim, rank], &b_values),
                },
            )]
            .into(),
        };
        // A wider second adapter makes the stack rank 3, so the narrow one is padded.
        let wide = adapter(3, 3.0, &[("model.layers.0.self_attn.q_proj", 4, 6)], 1.0);
        let stack = AdapterStack::build(&[narrow, wide], &map).unwrap();
        let r = 3;
        let a = values(
            &stack,
            WeightId::layer(0, WeightRole::Lora(LoraRole::A(LoraTarget::AttnQ))),
        );
        let b = values(
            &stack,
            WeightId::layer(0, WeightRole::Lora(LoraRole::B(LoraTarget::AttnQ))),
        );
        for i in 0..in_dim {
            for k in 0..r {
                let expected = if k < rank { (10 * k + i) as f32 } else { 0.0 };
                assert_eq!(
                    a[in_dim * r + i * r + k],
                    expected,
                    "A entry 1 at ({i}, {k})"
                );
            }
        }
        for k in 0..r {
            for o in 0..out_dim {
                let expected = if k < rank {
                    (100 + 10 * o + k) as f32
                } else {
                    0.0
                };
                assert_eq!(
                    b[r * out_dim + k * out_dim + o],
                    expected,
                    "B entry 1 at ({k}, {o})"
                );
            }
        }
    }

    /// One stack holds one dtype: a BF16 adapter beside an F32 one is the typed refusal, never a widened
    /// copy.
    #[test]
    fn adapters_stored_in_different_dtypes_cannot_share_a_stack() {
        let (_, map) = base();
        let modules = [("model.layers.0.self_attn.q_proj", 4, 6)];
        let error = AdapterStack::build(
            &[
                adapter_as(DType::F32, 2, 4.0, &modules, 1.0),
                adapter_as(DType::BF16, 2, 4.0, &modules, 1.0),
            ],
            &map,
        )
        .unwrap_err();
        assert_eq!(
            error,
            AdapterError::MixedDtype {
                adapter: 1,
                expected: DType::F32,
                found: DType::BF16
            }
        );
    }

    /// Two adapters of different ranks stack to the larger rank, zero-padded; entry 0 is the no-adapter
    /// row; a layer an adapter does not target is exactly zero; the scale vector is each adapter's
    /// `alpha / r` behind a zero.
    #[test]
    fn adapters_stack_padded_with_a_zero_no_adapter_entry() {
        let (_, map) = base();
        let narrow = adapter(2, 4.0, &[("model.layers.0.self_attn.q_proj", 4, 6)], 1.0);
        let wide = adapter(
            3,
            6.0,
            &[
                ("model.layers.0.self_attn.q_proj", 4, 6),
                ("model.layers.1.self_attn.q_proj", 4, 6),
            ],
            2.0,
        );
        let stack = AdapterStack::build(&[narrow, wide], &map).unwrap();
        assert_eq!(stack.rank.get(), 3);
        assert_eq!(stack.projections, vec![LoraTarget::AttnQ]);

        let a0 = values(
            &stack,
            WeightId::layer(0, WeightRole::Lora(LoraRole::A(LoraTarget::AttnQ))),
        );
        // [3 entries, in 4, rank 3]: entry 0 zero, the narrow adapter's third rank column zero, the wide one full.
        assert_eq!(a0.len(), 3 * 4 * 3);
        assert!(a0[..12].iter().all(|&v| v == 0.0), "entry 0 is zero");
        let narrow_a = &a0[12..24];
        assert!(
            narrow_a.chunks(3).all(|row| row == [1.0, 1.0, 0.0]),
            "{narrow_a:?}"
        );
        assert!(a0[24..].iter().all(|&v| v == 2.0));

        let b1 = values(
            &stack,
            WeightId::layer(1, WeightRole::Lora(LoraRole::B(LoraTarget::AttnQ))),
        );
        // Layer 1: only the wide adapter targets it.
        assert!(b1[..3 * 6 * 2].iter().all(|&v| v == 0.0));
        assert!(b1[3 * 6 * 2..].iter().all(|&v| v == 3.0));

        let scale = values(&stack, WeightId::model(WeightRole::Lora(LoraRole::Scale)));
        assert_eq!(
            scale,
            vec![0.0, 2.0, 2.0],
            "alpha / r behind the no-adapter zero"
        );
    }

    #[test]
    fn a_module_the_model_lacks_or_a_factor_that_does_not_fit_is_a_typed_refusal() {
        let (_, map) = base();
        let beyond = adapter(2, 4.0, &[("model.layers.5.self_attn.q_proj", 4, 6)], 1.0);
        assert_eq!(
            AdapterStack::build(&[beyond], &map).unwrap_err(),
            AdapterError::NoWeight {
                adapter: 0,
                layer: 5,
                target: LoraTarget::AttnQ
            }
        );
        let unknown = adapter(2, 4.0, &[("model.layers.0.self_attn.rotary", 4, 6)], 1.0);
        assert_eq!(
            AdapterStack::build(&[unknown], &map).unwrap_err(),
            AdapterError::UnknownModule {
                module: "model.layers.0.self_attn.rotary".to_string()
            }
        );
        let narrow = adapter(2, 4.0, &[("model.layers.0.self_attn.q_proj", 5, 6)], 1.0);
        assert!(matches!(
            AdapterStack::build(&[narrow], &map).unwrap_err(),
            AdapterError::Shape { layer: 0, .. }
        ));
        assert_eq!(
            AdapterStack::build(&[], &map).unwrap_err(),
            AdapterError::Empty
        );
    }

    /// The generation's store shares every base entry's generation and adds the stacked tensors under
    /// their weight-id names, and its map names both.
    #[test]
    fn a_generations_weights_share_the_base_entries_and_add_the_stack() {
        let (store, map) = base();
        let only = adapter(2, 4.0, &[("model.layers.0.self_attn.q_proj", 4, 6)], 1.0);
        let stack = AdapterStack::build(&[only], &map).unwrap();
        let (extended, extended_map) = stack.weights(&store, &map).unwrap();
        for key in ["q0", "q1", "v0"] {
            assert_eq!(
                extended.generation(key),
                store.generation(key),
                "{key} keeps its generation, so an executor uploads it once"
            );
        }
        let a = WeightId::layer(0, WeightRole::Lora(LoraRole::A(LoraTarget::AttnQ)));
        assert!(extended.contains(&a.const_name()));
        assert!(extended_map.handle(a).is_some());
        assert!(
            extended_map
                .handle(WeightId::layer(1, LoraTarget::AttnQ.weight_role()))
                .is_some()
        );
        assert!(
            !store.contains(&a.const_name()),
            "the base store is untouched"
        );
    }
}
