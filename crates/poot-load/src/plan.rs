//! The one load pipeline (dmap.md section 7, ADR-0103 decision 4): a checkpoint reader turns its
//! container's index into a [`LoadPlan`] (one [`TensorJob`] per owner, built on one thread), and
//! [`build_store`] runs the jobs on a bounded pool of workers, each taking a view of its tensor's bytes
//! and constructing the validated entry (`DenseWeight::try_new` / `PackedPayload::try_new`, whose
//! content check reads the pages the worker just faulted in). The store is published only after every
//! job passed and every file is still the one that was mapped: nothing reaches a device from a
//! checkpoint that was refused.
//!
//! A worker holds a view of its tensor and the entry it builds, never a copy of the tensor's bytes; the
//! anonymous memory that grows with the checkpoint is the per-tensor bookkeeping. (The load receipt records
//! the process's anonymous peak for a 7B load; no test bounds it.)
//!
//! Determinism: the store, its insertion order, its generations and the reported error do not depend
//! on the worker count or on scheduling. Entries are inserted in plan order, and a failure reports the
//! LOWEST failing job, whatever finished first.

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
use poot_quant::{Access, PackedPayload, PackedWeight, SourceRole, StoredBytes};
use poot_tensor::DType;

use crate::LoadError;

/// The most workers the driver starts, however many cores the host has: the pages are faulted in by the
/// workers, and past this the page cache and the disk are the limit, not the check.
const MAX_WORKERS: usize = 16;

/// The worker count a load uses when the caller has no reason to pick one: the host's parallelism,
/// capped at 16. It is not a user option, and nothing in the pipeline skips validation.
pub fn default_workers() -> NonZeroUsize {
    std::thread::available_parallelism()
        .unwrap_or(NonZeroUsize::MIN)
        .min(NonZeroUsize::new(MAX_WORKERS).expect("16 is non-zero"))
}

/// Which container a plan reads: a dense tensor that fails its checks is reported in that container's
/// own words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Container {
    Gguf,
    Safetensors,
}

/// What one job builds from its bytes.
#[derive(Debug)]
pub(crate) enum JobKind {
    Dense { dtype: DType, shape: Vec<usize> },
    Packed { weight: PackedWeight },
}

/// One owner of the store: the entry stored under `key`, built from `range` of `files[file]`.
#[derive(Debug)]
pub(crate) struct TensorJob {
    pub(crate) key: WeightKey,
    pub(crate) file: usize,
    pub(crate) range: Range<usize>,
    pub(crate) kind: JobKind,
}

/// The per-tensor work of one checkpoint, planned single-threaded from its index. Jobs are in the
/// order the entries are inserted into the store.
#[derive(Debug)]
pub struct LoadPlan {
    container: Container,
    files: Vec<StoredBytes>,
    jobs: Vec<TensorJob>,
}

impl LoadPlan {
    pub(crate) fn new(container: Container, files: Vec<StoredBytes>) -> Self {
        Self {
            container,
            files,
            jobs: Vec::new(),
        }
    }

    pub(crate) fn push(&mut self, job: TensorJob) {
        debug_assert!(job.file < self.files.len());
        self.jobs.push(job);
    }

    /// The number of owners the plan builds.
    pub fn len(&self) -> usize {
        self.jobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs.is_empty()
    }
}

impl TensorJob {
    /// The validated entry for this job, built from a view of its bytes.
    fn run(&self, container: Container, files: &[StoredBytes]) -> Result<WeightEntry, LoadError> {
        let bytes = files[self.file].slice(self.range.clone()).map_err(|_| {
            LoadError::TensorOutsideFile {
                tensor: self.key.to_string(),
                need: self.range.end,
                have: files[self.file].len(),
            }
        })?;
        match &self.kind {
            JobKind::Dense { dtype, shape } => DenseWeight::try_new(*dtype, shape.clone(), bytes)
                .map(WeightEntry::Dense)
                .map_err(|error| match container {
                    Container::Gguf => LoadError::from(error),
                    Container::Safetensors => {
                        LoadError::SafeTensors(format!("{}: {error}", self.key))
                    }
                }),
            JobKind::Packed { weight } => {
                PackedPayload::try_new(*weight, [(SourceRole::Blocks, bytes)])
                    .map(|payload| WeightEntry::Packed(std::sync::Arc::new(payload)))
                    .map_err(LoadError::from)
            }
        }
    }
}

/// Run `plan` on `workers` threads and publish the store.
///
/// Workers take jobs from a shared cursor in ascending file offset, so each file is read front to
/// back. After the last job, each file is checked once (`StoredBytes::check_unchanged`, ADR-0103
/// amendment boundary 1) and the entries are inserted in plan order. On failure the error of the
/// lowest failing job is returned, whatever finished first; workers stop taking jobs past the lowest
/// failure seen.
pub fn build_store(plan: LoadPlan, workers: NonZeroUsize) -> Result<WeightStore, LoadError> {
    let LoadPlan {
        container,
        files,
        jobs,
    } = plan;
    for file in &files {
        file.advise(Access::Sequential);
    }
    let mut order: Vec<usize> = (0..jobs.len()).collect();
    order.sort_by_key(|&job| (jobs[job].file, jobs[job].range.start, job));

    let cursor = AtomicUsize::new(0);
    // The lowest job index that has failed so far (`usize::MAX`: none). A job above it can never be
    // the reported error, so it is not started.
    let lowest_failure = AtomicUsize::new(usize::MAX);
    let results: Vec<Mutex<Option<Result<WeightEntry, LoadError>>>> =
        jobs.iter().map(|_| Mutex::new(None)).collect();
    let workers = workers.get().min(jobs.len().max(1));
    std::thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| {
                loop {
                    let next = cursor.fetch_add(1, Ordering::Relaxed);
                    let Some(&job) = order.get(next) else { break };
                    if job > lowest_failure.load(Ordering::Relaxed) {
                        continue;
                    }
                    let result = jobs[job].run(container, &files);
                    if result.is_err() {
                        lowest_failure.fetch_min(job, Ordering::Relaxed);
                    }
                    *results[job].lock().expect("a worker does not panic") = Some(result);
                }
            });
        }
    });

    let mut builder = WeightStore::builder();
    for (job, result) in jobs.into_iter().zip(results) {
        let entry = result
            .into_inner()
            .expect("a worker does not panic")
            .expect("every job below the lowest failure ran")?;
        builder.insert(job.key, entry)?;
    }
    for file in &files {
        file.check_unchanged()?;
    }
    Ok(builder.build())
}
