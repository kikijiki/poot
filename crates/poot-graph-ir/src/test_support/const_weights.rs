//! Weights for a toy graph whose consts are named by the test, not by a model's weight table.
//!
//! The executor binds a const through a [`WeightMap`]: its name must be a
//! [`WeightId::const_name`] (or a [`PackedSourceName`] over one). A fixture graph names its consts
//! `x`, `w`, `table`; [`ConstWeights`] gives each const the store holds a [`WeightId`], renames the
//! graph's const to that id's spelling, and builds the map that views the store entry the const was
//! named after. The store is untouched, so an oracle that reads the original graph and store sees
//! the same bytes the executor binds. A const the store does not hold keeps its name and stays
//! unbound, so a row about a missing weight still sees the original name.
//!
//! A packed expert stack is named by its stack id and held in the store as the experts `{id}.0`,
//! `{id}.1`, .. (the key spelling a GGUF load gives a rank-3 expert tensor): its id views them as one
//! [`WeightView::ExpertStack`].

use std::collections::BTreeMap;

use crate::{Graph, PackedSourceName, Storage, WeightRef};
use poot_quant::weights::{WeightKey, WeightStore};
use poot_weights::{AttnRole, FfnRole, WeightId, WeightMap, WeightRole, WeightView};

/// The ids handed to the consts of one or more graphs that share one executable and store.
pub struct ConstWeights<'s> {
    store: &'s WeightStore,
    ids: BTreeMap<String, WeightId>,
}

impl<'s> ConstWeights<'s> {
    /// An empty id table over `store`; ids are handed out in first-use order as graphs name store entries.
    pub fn new(store: &'s WeightStore) -> Self {
        Self {
            store,
            ids: BTreeMap::new(),
        }
    }

    /// The id of store entry `key`: one per entry, so two graphs naming one const share it.
    fn id(&mut self, key: &str) -> WeightId {
        let next = self.ids.len();
        *self.ids.entry(key.to_string()).or_insert_with(|| {
            let role = if next.is_multiple_of(2) {
                WeightRole::Attn(AttnRole::Q)
            } else {
                WeightRole::Ffn(FfnRole::Gate)
            };
            WeightId::layer(next / 2, role)
        })
    }

    /// `g` with each const the store holds named by its [`WeightId`]; every other value is untouched.
    pub fn graph(&mut self, g: &Graph) -> Graph {
        let mut named = g.edit();
        for &input in g.inputs() {
            let meta = g.meta(input);
            if meta.storage != Storage::Const {
                continue;
            }
            let name = meta.name.as_deref().expect("a const has a name");
            let (renamed, weight) = match PackedSourceName::parse(name) {
                Some(source)
                    if self.store.contains(source.linear_id())
                        || self.store.contains(&format!("{}.0", source.linear_id())) =>
                {
                    let id = self.id(source.linear_id());
                    let weight = WeightRef::whole(id).with_source(source.role());
                    (weight.name(), weight)
                }
                Some(_) => continue,
                None if self.store.contains(name) => {
                    let weight = WeightRef::whole(self.id(name));
                    (weight.name(), weight)
                }
                None => match name.rsplit_once(".chunk") {
                    // legalize's `<base>.chunkN` split of a stored weight follows its parent's id.
                    Some((base, index))
                        if index.parse::<usize>().is_ok() && self.store.contains(base) =>
                    {
                        let index = index.parse().expect("checked above");
                        let weight = WeightRef::whole(self.id(base)).with_chunk(index);
                        (weight.name(), weight)
                    }
                    _ => continue,
                },
            };
            named.values_mut()[input].name = Some(renamed);
            // The reference follows the name, so a pass that derives a part's name from it (legalize's
            // chunks) stays consistent with the store entries the map views.
            named.values_mut()[input].weight = Some(weight);
        }
        named
            .commit()
            .expect("renaming a const keeps the graph well formed")
    }

    /// `g` alone: its named graph and the map over `store`, for an executable that loads one graph.
    pub fn bind(g: &Graph, store: &'s WeightStore) -> (Graph, WeightMap) {
        let mut weights = Self::new(store);
        let named = weights.graph(g);
        (named, weights.map())
    }

    /// The map over the store: each id views the entry its const was named after. Consumes the
    /// binding, so every graph is named first: a map taken earlier would miss their consts.
    pub fn map(self) -> WeightMap {
        let mut map = WeightMap::builder(self.store);
        for (key, &id) in &self.ids {
            let view = if self.store.contains(key) {
                WeightView::Stored(WeightKey::from(key.as_str()))
            } else {
                let experts = (0..)
                    .map(|expert| format!("{key}.{expert}"))
                    .take_while(|expert| self.store.contains(expert))
                    .map(|expert| WeightKey::from(expert.as_str()))
                    .collect();
                WeightView::ExpertStack(experts)
            };
            map.map(id, view)
                .expect("the store holds the entry a const was named after");
        }
        map.build()
    }
}
