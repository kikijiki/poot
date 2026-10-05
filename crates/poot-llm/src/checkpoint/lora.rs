//! The Runner's named LoRA adapter pool for batched serving, and the request leases that hold an adapter
//! alive. The pool's tracer is the dense qwen2 block's, and the dense families run on the driver (whose
//! own adapter registry is `driver::lora`), so no family left on the Runner admits an adapter:
//! registration refuses every Runner until `poot-serve` serves through the driver (POOT-753).

use std::path::Path;

use poot_load::lora::{LoraAdapter, LoraAdapterPool, LoraStackedModule};
use poot_models::qwen2::LoraBatchedSpec;
use poot_tensor::HostTensor;

use crate::core::runner::{LoraHotState, Runner};
use crate::driver::prefix_cache::PrefixIdentity;
use crate::error::{OptionExt, Result};
/// A request-owned claim on one LoRA pool identity. The adapter index is resolved and its counter
/// incremented together under [`Runner`]'s pool lock; the counter is decremented when the last clone of this
/// handle is dropped. Cloning shares one claim, so an HTTP request can retain the same adapter across its
/// response task and queued job.
///
/// The no-adapter handle owns no counter and always reports [`LoraAdapterPool::NO_ADAPTER`]. Named handles
/// can only be created by [`Runner::acquire_lora_adapter`], so queued work cannot obtain a reusable pool
/// index without acquiring its lifetime.
#[derive(Clone)]
pub struct LoraAdapterLease {
    inner: Option<std::sync::Arc<LoraAdapterLeaseInner>>,
}

struct LoraAdapterLeaseInner {
    index: usize,
    /// The adapter's effective-weights identity at acquisition, captured under the pool lock alongside
    /// the index. Paged prefix reuse keys cached K/V by this value, so a lease keeps the same identity
    /// through queueing, admission, preemption, and completion even if the pool index is later reloaded
    /// with a different adapter (spec 248, ADR-0030).
    identity: PrefixIdentity,
    in_flight: std::sync::Arc<std::sync::atomic::AtomicUsize>,
}

impl Drop for LoraAdapterLeaseInner {
    fn drop(&mut self) {
        let previous = self
            .in_flight
            .fetch_sub(1, std::sync::atomic::Ordering::AcqRel);
        assert!(previous > 0, "LoRA adapter lease counter underflow");
    }
}

impl std::fmt::Debug for LoraAdapterLease {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoraAdapterLease")
            .field("index", &self.index())
            .finish_non_exhaustive()
    }
}

impl Default for LoraAdapterLease {
    fn default() -> Self {
        Self::none()
    }
}

impl LoraAdapterLease {
    /// Construct the inert selection used when a request omits `lora_adapter`.
    pub fn none() -> Self {
        Self { inner: None }
    }

    /// The immutable pool index selected for this request's entire lifetime.
    pub fn index(&self) -> usize {
        self.inner
            .as_ref()
            .map_or(LoraAdapterPool::NO_ADAPTER, |inner| inner.index)
    }

    /// The effective-weights identity of this request's adapter, for paged prefix-cache keying (spec
    /// 248, ADR-0030). [`PrefixIdentity::BASE`] for the inert no-adapter handle. Stable for the lease's
    /// lifetime, so preemption/resume re-register and re-match under the same identity.
    pub fn prefix_identity(&self) -> PrefixIdentity {
        self.inner
            .as_ref()
            .map_or(PrefixIdentity::BASE, |inner| inner.identity)
    }

    pub fn is_none(&self) -> bool {
        self.inner.is_none()
    }
}

/// The projections the batched-serving LoRA decode wires through `lora_linear_batched`: the four attention and three MLP
/// projections.
const SUPPORTED_LORA_POOL_TARGETS: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

/// The seven full module paths (`self_attn.`/`mlp.`-prefixed, matching the base weight naming)
/// [`Runner::rebind_lora_pool`] iterates per layer: the module-path form of
/// [`SUPPORTED_LORA_POOL_TARGETS`].
const LORA_POOL_PROJ_PATHS: &[&str] = &[
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

impl Runner {
    /// The architecture prerequisite [`Self::register_lora_adapter`] enforces before binding anything,
    /// plus the `target_modules subset-of supported` gate. The pool's tracer is the dense qwen2 block's,
    /// which runs on the driver, so every family the Runner loads is refused here.
    fn validate_lora_adapter(&self, adapter: &LoraAdapter, supported: &[&str]) -> Result<()> {
        bail!(
            "lora adapter: no family the Runner loads ({}) has a LoRA-wired tracer; adapters run on the \
             driver (driver::lora); target_modules {:?} of {supported:?} were not bound",
            self.arch,
            adapter.config.target_modules
        );
    }

    /// Register (or, if `name` is already registered, replace; see
    /// [`poot_load::lora::LoraAdapterPool::register`]) a named adapter in this Runner's multi-adapter pool for
    /// batched multi-adapter serving. Enforces [`Self::validate_lora_adapter`] first, so a pool adapter cannot silently apply a correction
    /// the batched LoRA decode does not wire. Creates the pool on first call.
    /// Returns the adapter's stable pool index (1-based; `NO_ADAPTER` = 0 stays reserved).
    ///
    /// Binds nothing into `Self::weights`: call [`Self::rebind_lora_pool`] after registering every adapter
    /// that should be live.
    pub fn register_lora_adapter(
        &mut self,
        name: impl Into<String>,
        adapter: LoraAdapter,
    ) -> Result<usize> {
        self.validate_lora_adapter(&adapter, SUPPORTED_LORA_POOL_TARGETS)?;
        let pool = self
            .lora_hot
            .get_mut()
            .unwrap()
            .pool
            .get_or_insert_with(LoraAdapterPool::new);
        Ok(pool.register(name, adapter))
    }

    /// [`Self::register_lora_adapter`] from a PEFT adapter directory (the file-IO entry point).
    pub fn register_lora_adapter_dir(
        &mut self,
        name: impl Into<String>,
        dir: impl AsRef<Path>,
    ) -> Result<usize> {
        let adapter = LoraAdapter::load(dir.as_ref())?;
        self.register_lora_adapter(name, adapter)
    }

    /// The pool index a registered adapter's name maps to ([`poot_load::lora::LoraAdapterPool::index_of`]),
    /// or `None` if no pool exists yet or `name` was never registered. Callers typically fall back to
    /// `NO_ADAPTER` for an unrecognized name.
    pub fn lora_pool_index_of(&self, name: &str) -> Option<usize> {
        self.lora_hot
            .read()
            .unwrap()
            .pool
            .as_ref()
            .and_then(|p| p.index_of(name))
    }

    /// Resolve a request's optional adapter name and acquire its request lifetime atomically under the pool
    /// lock. A named handle prevents hot-unload and slot reuse until its last clone is dropped, including time
    /// queued before admission. An absent name returns an inert handle without touching the pool. Unknown names
    /// are errors.
    pub fn acquire_lora_adapter(&self, name: Option<&str>) -> Result<LoraAdapterLease> {
        let Some(name) = name else {
            return Ok(LoraAdapterLease::none());
        };
        let mut hot = self.lora_hot.write().unwrap();
        let (idx, identity) = {
            let pool = hot.pool.as_ref().ok_or_else(|| {
                err!(
                    "unknown lora_adapter {name:?} - not registered at server startup (see the \
                     --lora-adapter CLI flag)"
                )
            })?;
            let idx = pool.index_of(name).ok_or_else(|| {
                err!(
                    "unknown lora_adapter {name:?} - not registered at server startup (see the \
                     --lora-adapter CLI flag)"
                )
            })?;
            let identity = pool
                .identity_of(idx)
                .map(PrefixIdentity::adapter)
                .ok_or_else(|| {
                err!(
                    "lora_adapter {name:?} (pool index {idx}) has no effective-weights identity - \
                     the pool slot is inconsistent with its name mapping"
                )
            })?;
            (idx, identity)
        };
        let in_flight = hot
            .in_flight
            .entry(idx)
            .or_insert_with(|| std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0)))
            .clone();
        in_flight.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        Ok(LoraAdapterLease {
            inner: Some(std::sync::Arc::new(LoraAdapterLeaseInner {
                index: idx,
                identity,
                in_flight,
            })),
        })
    }

    /// Bind every registered pool adapter's stacked constants for all seven projections
    /// (`q_proj`/`k_proj`/`v_proj`/`o_proj`/`gate_proj`/`up_proj`/`down_proj`) into `Runner::weights`, under the
    /// names the batched LoRA decode graph declares
    /// (`model.layers.{li}.self_attn.{q,k,v,o}_proj.lora_{a,b}_stacked` /
    /// `model.layers.{li}.mlp.{gate,up,down}_proj.lora_{a,b}_stacked` / `...lora_scaling_vec`), and return the
    /// [`LoraBatchedSpec`] (`n_adapters`, shared `rank`) the tracer needs to declare matching const shapes.
    /// Call again whenever the pool's adapter set changes; every targeted module is rebound in one pass rather
    /// than patched in place. Replacing this binding while a batched step is in flight is unexplored.
    ///
    /// Reuses [`poot_load::lora::LoraAdapterPool::stack_for_module`]'s zero-padding/sentinel convention per
    /// `(layer, proj)`. The tracer uses one shared `rank` across every targeted module
    /// (see [`poot_models::qwen2::LoraBatchedSpec`]), so a module whose own `stack_for_module` rank is smaller
    /// than the shared max is padded with further zero columns/rows; a zero-padded extra rank contributes
    /// nothing to B@A. A `(layer, proj)` that no registered adapter targets still gets an all-zero stacked slot
    /// (the tracer declares these consts unconditionally once a LoRA spec is present).
    ///
    /// Errors (no partial rebind) if the pool is empty or missing, or if no registered adapter targets any of
    /// the seven projections on any layer.
    pub fn rebind_lora_pool(&mut self) -> Result<LoraBatchedSpec> {
        let hot = self.lora_hot.get_mut().unwrap();
        let pool = hot
            .pool
            .as_ref()
            .context("rebind_lora_pool: no pool - call register_lora_adapter first")?;
        if pool.is_empty() {
            bail!("rebind_lora_pool: the pool is empty (no adapters registered)");
        }
        let n_adapters = pool.len();
        let slots = n_adapters + 1;

        let mut per_module: Vec<(String, LoraStackedModule)> = Vec::new();
        for li in 0..self.cfg.layers {
            for proj in LORA_POOL_PROJ_PATHS {
                let module = format!("model.layers.{li}.{proj}");
                if let Some(sm) = pool.stack_for_module(&module)? {
                    per_module.push((module, sm));
                }
            }
        }
        if per_module.is_empty() {
            bail!(
                "rebind_lora_pool: no registered adapter targets any of {LORA_POOL_PROJ_PATHS:?} on any \
                 layer - nothing to bind"
            );
        }
        let rank = per_module.iter().map(|(_, sm)| sm.r).max().unwrap();

        for (module, sm) in &per_module {
            let in_dim = sm.a.shape()[1];
            let out_dim = sm.b.shape()[2];
            let r = sm.r;
            let a = if r == rank {
                sm.a.as_f32().unwrap().to_vec()
            } else {
                pad_rank_cols(sm.a.as_f32().unwrap(), slots, in_dim, r, rank)
            };
            let b = if r == rank {
                sm.b.as_f32().unwrap().to_vec()
            } else {
                pad_rank_rows(sm.b.as_f32().unwrap(), slots, r, out_dim, rank)
            };
            self.weights.insert(
                format!("{module}.lora_a_stacked"),
                poot_eval::Value::from(HostTensor::f32(vec![slots, in_dim, rank], a)),
            );
            self.weights.insert(
                format!("{module}.lora_b_stacked"),
                poot_eval::Value::from(HostTensor::f32(vec![slots, rank, out_dim], b)),
            );
            self.weights.insert(
                format!("{module}.lora_scaling_vec"),
                poot_eval::Value::from(HostTensor::f32(
                    vec![slots],
                    sm.scaling.as_f32().unwrap().to_vec(),
                )),
            );
        }
        // A (layer, proj) that no registered adapter targets never appeared in `per_module` above
        // (`stack_for_module` returns `Ok(None)`), but the batched tracer declares its consts unconditionally once
        // `lora.is_some()`, so bind an all-zero, inert stack there too. `in_dim`/`out_dim`: `o_proj` projects from `q_dim`; `gate_proj`/`up_proj` project `hidden -> inter`;
        // `down_proj` projects `inter -> hidden`.
        let q_dim = self.cfg.n_heads * self.cfg.head_dim;
        let kv_dim = self.cfg.n_kv_heads * self.cfg.head_dim;
        for li in 0..self.cfg.layers {
            for (proj, in_dim, out_dim) in [
                ("self_attn.q_proj", self.cfg.hidden, q_dim),
                ("self_attn.k_proj", self.cfg.hidden, kv_dim),
                ("self_attn.v_proj", self.cfg.hidden, kv_dim),
                ("self_attn.o_proj", q_dim, self.cfg.hidden),
                ("mlp.gate_proj", self.cfg.hidden, self.cfg.inter),
                ("mlp.up_proj", self.cfg.hidden, self.cfg.inter),
                ("mlp.down_proj", self.cfg.inter, self.cfg.hidden),
            ] {
                let module = format!("model.layers.{li}.{proj}");
                let key_a = format!("{module}.lora_a_stacked");
                if self.weights.contains_key(&key_a) {
                    continue;
                }
                self.weights.insert(
                    key_a,
                    poot_eval::Value::from(HostTensor::f32(
                        vec![slots, in_dim, rank],
                        vec![0.0f32; slots * in_dim * rank],
                    )),
                );
                self.weights.insert(
                    format!("{module}.lora_b_stacked"),
                    poot_eval::Value::from(HostTensor::f32(
                        vec![slots, rank, out_dim],
                        vec![0.0f32; slots * rank * out_dim],
                    )),
                );
                self.weights.insert(
                    format!("{module}.lora_scaling_vec"),
                    poot_eval::Value::from(HostTensor::f32(vec![slots], vec![0.0f32; slots])),
                );
            }
        }
        // hot-load/unload: record the graph's traced rank dimension. A later hot-loaded adapter whose own `r`
        // exceeds it is refused (see `Self::hot_load_lora_adapter`), since only a re-trace can change the graph's
        // stacked-const shape.
        self.lora_hot.get_mut().unwrap().rank_ceiling = rank;
        Ok(LoraBatchedSpec { n_adapters, rank })
    }

    /// Load a new named adapter from a PEFT directory into the live pool of an already-serving `Runner`
    /// (`&self`, so callable through the `Arc<Runner>` every HTTP/engine thread shares), without a restart or
    /// disturbing in-flight requests for a different adapter or the base model.
    ///
    /// Checks before anything is registered:
    /// 1. The architecture + `target_modules subset-of SUPPORTED_LORA_POOL_TARGETS` gate of
    ///    [`Self::validate_lora_adapter`].
    /// 2. `adapter.config.r` must not exceed this pool's traced rank ceiling ([`Self::rebind_lora_pool`]'s last
    ///    `LoraBatchedSpec::rank`): the resident graph's stacked-const rank is fixed at trace time; a higher-rank
    ///    adapter needs a bigger `--lora-max-rank` and a restart.
    /// 3. A free (reserved-but-unused, or [`Self::hot_unload_lora_adapter`]ed) pool slot must exist; see
    ///    [`poot_load::lora::LoraAdapterPool::hot_load`] (the slot count is also fixed at trace time).
    ///
    /// On success, overrides only the stacked consts for modules this adapter targets and bumps the
    /// hot-load generation counter. Returns the new adapter's stable pool index.
    pub fn hot_load_lora_adapter_dir(
        &self,
        name: impl Into<String>,
        dir: impl AsRef<Path>,
    ) -> Result<usize> {
        let adapter = LoraAdapter::load(dir.as_ref())?;
        self.hot_load_lora_adapter(name, adapter)
    }

    /// [`Self::hot_load_lora_adapter_dir`] from an already-parsed [`LoraAdapter`] (the non-file-IO entry point).
    pub fn hot_load_lora_adapter(
        &self,
        name: impl Into<String>,
        adapter: LoraAdapter,
    ) -> Result<usize> {
        self.validate_lora_adapter(&adapter, SUPPORTED_LORA_POOL_TARGETS)?;
        let name = name.into();
        let mut hot = self.lora_hot.write().unwrap();
        if hot.rank_ceiling == 0 {
            bail!(
                "hot_load_lora_adapter: no LoRA pool has been traced yet - call register_lora_adapter + \
                 rebind_lora_pool at startup first (at least one adapter is required before any \
                 hot-load; --lora-pool-capacity reserves headroom for FUTURE hot-loads)"
            );
        }
        if adapter.config.r > hot.rank_ceiling {
            bail!(
                "hot_load_lora_adapter: adapter rank {} exceeds this pool's traced rank ceiling of {} - \
                 restart with a larger --lora-max-rank (or a startup adapter of at least this rank) to \
                 reserve headroom",
                adapter.config.r,
                hot.rank_ceiling
            );
        }
        let rank_ceiling = hot.rank_ceiling;
        let targets = adapter.config.target_modules.clone();
        let pool = hot.pool.get_or_insert_with(LoraAdapterPool::new);
        let idx = pool.hot_load(name, adapter)?;
        Self::refresh_lora_overrides(&mut hot, self.cfg.layers, &targets, rank_ceiling)?;
        hot.generation += 1;
        Ok(idx)
    }

    /// The inverse of [`Self::hot_load_lora_adapter_dir`]: remove `name` from the live pool. Refuses (a typed
    /// error the HTTP admin layer maps to 409 Conflict) while any decode slot is in flight for this adapter
    /// (a nonzero lease count), since unloading mid-generation would switch that request's
    /// correction mid-stream. Otherwise reverts the adapter's slot to the inert placeholder (other adapters'
    /// indices are unaffected), recomputes the now-zero stacked consts for every module it targeted, and bumps
    /// the hot-load generation counter as a hot-load does. Returns the freed pool index.
    pub fn hot_unload_lora_adapter(&self, name: &str) -> Result<usize> {
        let mut hot = self.lora_hot.write().unwrap();
        let idx = hot
            .pool
            .as_ref()
            .and_then(|p| p.index_of(name))
            .with_context(|| format!("hot_unload_lora_adapter: {name:?} is not registered"))?;
        let in_flight = hot
            .in_flight
            .get(&idx)
            .map(|count| count.load(std::sync::atomic::Ordering::Acquire))
            .unwrap_or(0);
        if in_flight > 0 {
            bail!(
                "hot_unload_lora_adapter: {name:?} (pool index {idx}) has {in_flight} request(s) still \
                 in flight - wait for them to finish (or let them fail/complete) before unloading"
            );
        }
        let rank_ceiling = hot.rank_ceiling;
        let (freed_idx, old_targets) = hot
            .pool
            .as_mut()
            .unwrap()
            .hot_unload(name)
            .expect("index_of just confirmed this name is registered");
        debug_assert_eq!(freed_idx, idx);
        Self::refresh_lora_overrides(&mut hot, self.cfg.layers, &old_targets, rank_ceiling)?;
        hot.in_flight.remove(&idx);
        hot.generation += 1;
        Ok(freed_idx)
    }

    /// Recompute and override the stacked consts for every `(layer, target)` module in `targets` (a
    /// hot-loaded/unloaded adapter's own `target_modules`; other modules are left untouched), padding each up to
    /// `rank_ceiling` (the graph's traced shape; see [`Runner::rebind_lora_pool`]'s second pad). Writes into
    /// `hot.overrides`, which [`Runner::bind_decode_batched`]'s `Storage::Const` arm checks before
    /// `Runner::weights`.
    fn refresh_lora_overrides(
        hot: &mut LoraHotState,
        layers: usize,
        targets: &[String],
        rank_ceiling: usize,
    ) -> Result<()> {
        let pool = hot
            .pool
            .as_ref()
            .context("refresh_lora_overrides: no pool")?;
        let n_adapters = pool.len();
        let slots = n_adapters + 1;
        let mut fresh: Vec<(String, LoraStackedModule)> = Vec::new();
        for target in targets {
            let prefix = match target.as_str() {
                "gate_proj" | "up_proj" | "down_proj" => "mlp",
                _ => "self_attn",
            };
            for li in 0..layers {
                let module = format!("model.layers.{li}.{prefix}.{target}");
                if let Some(sm) = pool.stack_for_module(&module)? {
                    fresh.push((module, sm));
                }
            }
        }
        for (module, sm) in &fresh {
            let in_dim = sm.a.shape()[1];
            let out_dim = sm.b.shape()[2];
            let r = sm.r;
            if r > rank_ceiling {
                bail!(
                    "refresh_lora_overrides: {module} stacked rank {r} exceeds the graph's rank ceiling \
                     {rank_ceiling} - this should have been rejected before registering"
                );
            }
            let a = if r == rank_ceiling {
                sm.a.as_f32().unwrap().to_vec()
            } else {
                pad_rank_cols(sm.a.as_f32().unwrap(), slots, in_dim, r, rank_ceiling)
            };
            let b = if r == rank_ceiling {
                sm.b.as_f32().unwrap().to_vec()
            } else {
                pad_rank_rows(sm.b.as_f32().unwrap(), slots, r, out_dim, rank_ceiling)
            };
            hot.overrides.insert(
                format!("{module}.lora_a_stacked"),
                poot_eval::Value::from(HostTensor::f32(vec![slots, in_dim, rank_ceiling], a)),
            );
            hot.overrides.insert(
                format!("{module}.lora_b_stacked"),
                poot_eval::Value::from(HostTensor::f32(vec![slots, rank_ceiling, out_dim], b)),
            );
            hot.overrides.insert(
                format!("{module}.lora_scaling_vec"),
                poot_eval::Value::from(HostTensor::f32(
                    vec![slots],
                    sm.scaling.as_f32().unwrap().to_vec(),
                )),
            );
        }
        Ok(())
    }

    /// Every registered adapter name with its pool index and live in-flight count (the hot-load/unload admin
    /// surface, `poot-serve`'s `GET /v1/lora_adapters`). Empty if no pool exists yet. Order is unspecified.
    pub fn lora_pool_names(&self) -> Vec<(String, usize, usize)> {
        let hot = self.lora_hot.read().unwrap();
        let Some(pool) = hot.pool.as_ref() else {
            return Vec::new();
        };
        pool.names()
            .into_iter()
            .map(|(name, idx)| {
                let in_flight = hot
                    .in_flight
                    .get(&idx)
                    .map(|count| count.load(std::sync::atomic::Ordering::Acquire))
                    .unwrap_or(0);
                (name, idx, in_flight)
            })
            .collect()
    }
}

/// Zero-pad a stacked `A` `[slots, in_dim, r]` (row-major) up to `[slots, in_dim, rank_to]`
/// (`rank_to >= r`): copies each slot's `r` columns per row and leaves the new columns zero. See
/// [`Runner::rebind_lora_pool`] for why a second pad beyond `LoraAdapterPool::stack_for_module`'s is needed.
fn pad_rank_cols(data: &[f32], slots: usize, in_dim: usize, r: usize, rank_to: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; slots * in_dim * rank_to];
    for slot in 0..slots {
        for row in 0..in_dim {
            let src = (slot * in_dim + row) * r;
            let dst = (slot * in_dim + row) * rank_to;
            out[dst..dst + r].copy_from_slice(&data[src..src + r]);
        }
    }
    out
}

/// Zero-pad a stacked `B` `[slots, r, out_dim]` (row-major) up to `[slots, rank_to, out_dim]`: copies each
/// slot's `r` rows (a contiguous `r*out_dim` block) and leaves the new rows zero.
fn pad_rank_rows(data: &[f32], slots: usize, r: usize, out_dim: usize, rank_to: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; slots * rank_to * out_dim];
    for slot in 0..slots {
        let src_base = slot * r * out_dim;
        let dst_base = slot * rank_to * out_dim;
        out[dst_base..dst_base + r * out_dim]
            .copy_from_slice(&data[src_base..src_base + r * out_dim]);
    }
    out
}

/// No family the Runner loads has a LoRA-wired tracer, so registering an adapter refuses before anything
/// is bound. Mutation: let `validate_lora_adapter` accept; the adapter registers and the row fails.
#[cfg(test)]
mod pool_tests {
    use poot_load::lora::{LoraAdapter, LoraAdapterConfig};

    use crate::core::decode_arch::DecodeArch;
    use crate::core::decode_arch::fixtures::runner_for;

    #[test]
    fn registering_an_adapter_on_any_runner_family_is_refused() {
        for &arch in DecodeArch::ALL {
            let mut runner = runner_for(arch);
            let adapter = LoraAdapter {
                config: LoraAdapterConfig {
                    r: 2,
                    lora_alpha: 4.0,
                    target_modules: vec!["q_proj".to_string()],
                    use_rslora: false,
                },
                weights: Default::default(),
            };
            let error = runner
                .register_lora_adapter("a", adapter)
                .expect_err("no Runner family admits an adapter");
            assert!(
                error.to_string().contains("no family the Runner loads"),
                "{error}"
            );
            assert!(
                runner.lora_pool_names().is_empty(),
                "{arch:?}: nothing registered"
            );
        }
    }
}
