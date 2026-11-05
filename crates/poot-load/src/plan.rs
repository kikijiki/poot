//! The one load pipeline (dmap.md section 7, ADR-0103 decision 4): a checkpoint reader turns its
//! container's index into a [`LoadPlan`] (one [`TensorJob`] per owner), and [`build_store`] constructs the
//! store's entries from views of each tensor's bytes. Building an entry checks structure, never a byte of
//! content: lengths, shapes, the format's block layout. The checkpoint's bytes are read once, by the upload
//! that stages them for the device (`StoredBytes::read_at`, a positional file read), and that read checks
//! a packed payload's stored values as it goes (`PackedPayload::try_new_deferred`). The store is published
//! after every job passed and every file is still the one that was mapped; no executable binds a byte of a
//! checkpoint whose content check refused it (ADR-0103 amendment of 2026-10-27).
//!
//! The load touches no page of the mapping, so it counts nothing in the process's `RssFile`; the memory
//! that grows with the checkpoint is the per-tensor bookkeeping.
//!
//! Determinism: the store, its insertion order, its generations and the reported error follow the plan's
//! order: a failure reports the first failing job.

use std::ops::Range;

use poot_quant::weights::{DenseWeight, WeightEntry, WeightKey, WeightStore};
use poot_quant::{PackedPayload, PackedWeight, SourceRole, StoredBytes};
use poot_tensor::DType;

use crate::LoadError;

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

    /// Whether the plan builds no owners.
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
                PackedPayload::try_new_deferred(*weight, [(SourceRole::Blocks, bytes)])
                    .map(|payload| WeightEntry::Packed(std::sync::Arc::new(payload)))
                    .map_err(LoadError::from)
            }
        }
    }
}

/// Build the store `plan` describes and publish it.
///
/// Entries are inserted in plan order. After the last job, each file is checked once
/// (`StoredBytes::check_unchanged`, ADR-0103 amendment boundary 1). On failure the error of the first
/// failing job is returned.
pub fn build_store(plan: LoadPlan) -> Result<WeightStore, LoadError> {
    let LoadPlan {
        container,
        files,
        jobs,
    } = plan;
    let mut builder = WeightStore::builder();
    for job in jobs {
        let entry = job.run(container, &files)?;
        builder.insert(job.key, entry)?;
    }
    for file in &files {
        file.check_unchanged()?;
    }
    Ok(builder.build())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-job plan over a 16-byte file: `range` of it read as `kind`.
    fn plan_of(container: Container, range: Range<usize>, kind: JobKind) -> LoadPlan {
        let mut plan = LoadPlan::new(container, vec![StoredBytes::from(vec![0u8; 16])]);
        plan.push(TensorJob {
            key: WeightKey::from("w"),
            file: 0,
            range,
            kind,
        });
        plan
    }

    fn dense(dtype: DType, shape: Vec<usize>) -> JobKind {
        JobKind::Dense { dtype, shape }
    }

    /// POOT-1289: every reader checks its ranges against the container before it plans, so a job whose range
    /// leaves its file is a plan some other builder made; the pipeline's own guard refuses it by tensor name
    /// with both numbers, never a slice panic in a worker. The control is the same job one byte inside.
    ///
    /// Mutation: have `TensorJob::run` clamp the range to the file (`range.end.min(len)`); the job reads a
    /// short view and this row goes red (`expected TensorOutsideFile`).
    #[test]
    fn a_job_whose_range_leaves_its_file_is_a_tensor_outside_file_refusal() {
        let fits = plan_of(Container::Safetensors, 0..16, dense(DType::U8, vec![16]));
        build_store(fits).expect("control: the job ends at the file's end");
        let past = plan_of(Container::Safetensors, 8..17, dense(DType::U8, vec![9]));
        match build_store(past) {
            Err(LoadError::TensorOutsideFile { tensor, need, have }) => {
                assert_eq!((tensor.as_str(), need, have), ("w", 17, 16));
            }
            other => panic!("expected TensorOutsideFile, got {:?}", other.map(|_| ())),
        }
    }

    /// POOT-1289: a GGUF dense job whose bytes are not its shape's byte count is refused as a dense weight
    /// in the GGUF container's own variant (a safetensors one reads as a message naming the tensor). The
    /// GGUF reader sizes every dense range as element count times width, so this plan is a forged one.
    ///
    /// Mutation: make `DenseWeight::try_new` skip its byte-length comparison; the 8 bytes bind as `[4]` F32
    /// and this row goes red (`expected GgufDenseWeight`).
    #[test]
    fn a_gguf_dense_job_of_the_wrong_byte_count_is_a_dense_weight_refusal() {
        let plan = plan_of(Container::Gguf, 0..8, dense(DType::F32, vec![4]));
        match build_store(plan) {
            Err(LoadError::GgufDenseWeight(_)) => {}
            other => panic!("expected GgufDenseWeight, got {:?}", other.map(|_| ())),
        }
        let plan = plan_of(Container::Safetensors, 0..8, dense(DType::F32, vec![4]));
        match build_store(plan) {
            Err(LoadError::SafeTensors(message)) => {
                assert!(message.starts_with("w: "), "{message}")
            }
            other => panic!(
                "expected a safetensors message, got {:?}",
                other.map(|_| ())
            ),
        }
    }
}
