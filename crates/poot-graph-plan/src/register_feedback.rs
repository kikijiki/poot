//! Compile feedback for the register feasibility of a tiling (POOT-1505).
//!
//! The planner's register model (`RegisterTiling::candidates`) estimates what a tiling holds across its `K` loop;
//! whether the compiled kernel spills is the compiler's to say. A runtime that can ask its compiler implements
//! [`CompiledRegisters`] and hands it to the compile ([`crate::CompileOptions::register_feedback`]): the selection
//! then compiles each candidate's [`RegisterProbe`] in cost order and takes the cheapest that does not spill, the
//! register model only pruning what cannot fit the lane at all.
//!
//! The contract:
//! - No feedback (the default): the plan is the register model's, unchanged. The plan summary, the launch census
//!   and `ptx-graph-check` compile without feedback, stay GPU-free and describe this plan.
//! - Feedback: the plan may differ from the no-feedback plan, and records the compiler that decided it
//!   ([`RegisterFeedback::compiler`], a reported fact). The same graph, caps and compiler give the same plan: a
//!   verdict is a function of the probe body and the compiler.
//! - A probe is the generator's body for the contraction's shape and tiling, its operands, format and rows,
//!   without an epilogue: the precision pass, the expert grouping and the final plan ask the same question of the
//!   same body, so they cannot disagree.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use poot_target::KernelCompiler;

/// A kernel compiled only to learn its registers: the body the generator emits for one candidate tiling of a
/// contraction's shape (POOT-1505).
#[derive(Clone, Debug, PartialEq)]
pub struct RegisterProbe {
    /// The kernel body; its `name` is unique per shape and tiling.
    pub body: poot_kernel_ir::Body,
}

impl RegisterProbe {
    /// A stable digest of the body (its serialized form), the key of a cached verdict together with the compiler.
    pub fn digest(&self) -> blake3::Hash {
        blake3::hash(&serde_json::to_vec(&self.body).expect("a kernel body serializes"))
    }
}

/// What the compiler reports for a compiled probe.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RegisterVerdict {
    /// Registers per lane the compiled kernel allocates (PTX `CU_FUNC_ATTRIBUTE_NUM_REGS`, AMD `.vgpr_count`).
    pub registers: u32,
    /// Bytes of per-lane local or private memory the kernel uses (PTX `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES`, AMD the
    /// private segment): where a spilled register goes.
    pub local_bytes: u32,
}

impl RegisterVerdict {
    /// Whether the kernel keeps values in per-lane memory. Any local memory counts: a register spill and a local
    /// array alike put memory traffic in the loop, and the verdict does not tell them apart (POOT-1509: the driver
    /// JIT's local-memory verdict agrees with ptxas' spill verdict on all 385 sm_120 kernels of the sweep).
    pub const fn spills(self) -> bool {
        self.local_bytes > 0
    }
}

/// Why a probe has no verdict.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FeedbackError {
    /// The compiler refused the body.
    #[error("the compiler refused probe {name}: {reason}")]
    Compile {
        /// The probe body's name.
        name: String,
        /// What the compiler said.
        reason: String,
    },
    /// The device or driver failed outside compiling (a context, a query).
    #[error("the device failed for probe {name}: {reason}")]
    Device {
        /// The probe body's name.
        name: String,
        /// What failed.
        reason: String,
    },
}

/// A device's compiler, asked how many registers a probe allocates. Implemented by a runtime (PTX: the driver JIT
/// and `cuFuncGetAttribute`; ROCm: amd-clang and the code object's notes). The planner treats a probe with no
/// verdict (an error) as a tiling the device does not run.
pub trait CompiledRegisters: Send + Sync + std::fmt::Debug {
    /// The compiler that compiles this device's kernels: part of what names a plan made with this feedback.
    fn compiler(&self) -> KernelCompiler;

    /// Compile `probe` and report its registers.
    fn compile(&self, probe: &RegisterProbe) -> Result<RegisterVerdict, FeedbackError>;
}

/// The compile feedback a compile plans with. Two handles are equal when they name the same compiler: the same
/// compiler gives the same verdicts, so the plan is the same.
#[derive(Clone, Debug)]
pub struct RegisterFeedback(Arc<dyn CompiledRegisters>);

impl RegisterFeedback {
    /// A handle on `compiler`.
    pub fn new(compiler: impl CompiledRegisters + 'static) -> Self {
        Self(Arc::new(compiler))
    }

    /// The compiler this feedback asks.
    pub fn compiler(&self) -> KernelCompiler {
        self.0.compiler()
    }

    /// Whether `probe` compiles without spilling: `false` when it spills or has no verdict.
    pub fn holds(&self, probe: &RegisterProbe) -> bool {
        self.0.compile(probe).is_ok_and(|verdict| !verdict.spills())
    }

    /// The oracle behind the handle, for a test that reads a verdict, not only whether it holds.
    #[cfg(any(test, feature = "test-support"))]
    pub fn oracle(&self) -> &dyn CompiledRegisters {
        &*self.0
    }
}

impl poot_kernelgen::SpillCheck for RegisterFeedback {
    fn holds(&self, probe: &poot_kernel_ir::Body) -> bool {
        RegisterFeedback::holds(
            self,
            &RegisterProbe {
                body: probe.clone(),
            },
        )
    }
}

/// How a selection decides register feasibility under `feedback`: the compiler where the compile has one, else the
/// register model.
pub(crate) fn feasibility(feedback: Option<&RegisterFeedback>) -> poot_kernelgen::Feasibility<'_> {
    match feedback {
        Some(feedback) => poot_kernelgen::Feasibility::Compiled(feedback),
        None => poot_kernelgen::Feasibility::Model,
    }
}

impl PartialEq for RegisterFeedback {
    fn eq(&self, other: &Self) -> bool {
        self.compiler() == other.compiler()
    }
}

impl Eq for RegisterFeedback {}

/// An oracle whose verdicts are kept on disk, under `dir`, by compiler and probe digest: a probe is compiled once per
/// compiler, and a later plan of the same graph reads the verdict back (a planner's compile cache is per process;
/// the ROCm code-object cache is in memory only). A verdict file holds `registers local_bytes`. An error is not
/// cached.
#[derive(Debug)]
pub struct VerdictCache<O> {
    inner: O,
    dir: PathBuf,
    memory: Mutex<HashMap<blake3::Hash, RegisterVerdict>>,
}

impl<O: CompiledRegisters> VerdictCache<O> {
    /// `inner`'s verdicts, kept under `dir`.
    pub fn new(inner: O, dir: impl Into<PathBuf>) -> Self {
        Self {
            inner,
            dir: dir.into(),
            memory: Mutex::new(HashMap::new()),
        }
    }

    /// The file of `probe`'s verdict under this cache's compiler.
    fn path(&self, digest: blake3::Hash) -> PathBuf {
        let compiler = blake3::hash(format!("{:?}", self.inner.compiler()).as_bytes());
        self.dir
            .join(&compiler.to_hex()[..16])
            .join(digest.to_hex().as_str())
    }
}

impl<O: CompiledRegisters> CompiledRegisters for VerdictCache<O> {
    fn compiler(&self) -> KernelCompiler {
        self.inner.compiler()
    }

    fn compile(&self, probe: &RegisterProbe) -> Result<RegisterVerdict, FeedbackError> {
        let digest = probe.digest();
        if let Some(&verdict) = self.memory.lock().expect("the verdict map").get(&digest) {
            return Ok(verdict);
        }
        let path = self.path(digest);
        let read = std::fs::read_to_string(&path).ok().and_then(|text| {
            let mut fields = text.split_whitespace().map(str::parse::<u32>);
            match (fields.next(), fields.next()) {
                (Some(Ok(registers)), Some(Ok(local_bytes))) => Some(RegisterVerdict {
                    registers,
                    local_bytes,
                }),
                _ => None,
            }
        });
        let verdict = match read {
            Some(verdict) => verdict,
            None => {
                let verdict = self.inner.compile(probe)?;
                // A cache that cannot be written is a slower planner, not a wrong one.
                if let Some(parent) = path.parent()
                    && std::fs::create_dir_all(parent).is_ok()
                {
                    let _ = std::fs::write(
                        &path,
                        format!("{} {}\n", verdict.registers, verdict.local_bytes),
                    );
                }
                verdict
            }
        };
        self.memory
            .lock()
            .expect("the verdict map")
            .insert(digest, verdict);
        Ok(verdict)
    }
}

/// How a [`FakeCompiledRegisters`] answers a probe.
#[cfg(any(test, feature = "test-support"))]
pub type FakeVerdict =
    Box<dyn Fn(&RegisterProbe) -> Result<RegisterVerdict, FeedbackError> + Send + Sync>;

/// A compiler for tests: it reports `verdict(probe)` for every probe and counts the probes it was asked.
#[cfg(any(test, feature = "test-support"))]
pub struct FakeCompiledRegisters {
    /// The identity it reports.
    pub compiler: KernelCompiler,
    /// The verdict of a probe.
    pub verdict: FakeVerdict,
    /// How many probes it compiled.
    pub compiles: std::sync::atomic::AtomicUsize,
}

#[cfg(any(test, feature = "test-support"))]
impl std::fmt::Debug for FakeCompiledRegisters {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FakeCompiledRegisters")
            .field("compiler", &self.compiler)
            .finish_non_exhaustive()
    }
}

#[cfg(any(test, feature = "test-support"))]
impl CompiledRegisters for FakeCompiledRegisters {
    fn compiler(&self) -> KernelCompiler {
        self.compiler
    }

    fn compile(&self, probe: &RegisterProbe) -> Result<RegisterVerdict, FeedbackError> {
        self.compiles
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        (self.verdict)(probe)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn probe(name: &str) -> RegisterProbe {
        RegisterProbe {
            body: poot_kernel_ir::Body::new(name, 0, Vec::new(), Vec::new()),
        }
    }

    const CUDA: KernelCompiler = KernelCompiler::CudaDriver {
        release: 13_020,
        compute_capability: (12, 0),
    };

    fn counting(local_bytes: u32) -> FakeCompiledRegisters {
        FakeCompiledRegisters {
            compiler: CUDA,
            verdict: Box::new(move |_| {
                Ok(RegisterVerdict {
                    registers: 200,
                    local_bytes,
                })
            }),
            compiles: Default::default(),
        }
    }

    /// A verdict is compiled once per compiler and probe, then read back: from memory in one process, and from disk
    /// in a later one (a fresh cache over the same directory). Mutation: skip the disk read in `VerdictCache::compile`;
    /// the second cache compiles again.
    #[test]
    fn a_verdict_is_compiled_once_and_read_back_from_disk() {
        let dir =
            std::env::temp_dir().join(format!("poot-verdict-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let first = VerdictCache::new(counting(0), &dir);
        assert_eq!(first.compile(&probe("a")).unwrap().local_bytes, 0);
        assert_eq!(first.compile(&probe("a")).unwrap().local_bytes, 0);
        assert_eq!(
            first
                .inner
                .compiles
                .load(std::sync::atomic::Ordering::Relaxed),
            1
        );
        let second = VerdictCache::new(counting(16), &dir);
        assert_eq!(
            second.compile(&probe("a")).unwrap().local_bytes,
            0,
            "the verdict on disk, not a new compile"
        );
        assert_eq!(
            second
                .inner
                .compiles
                .load(std::sync::atomic::Ordering::Relaxed),
            0
        );
        assert!(
            second.compile(&probe("b")).unwrap().spills(),
            "another probe compiles"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Local memory of any size is a spill, a probe with no verdict does not hold, and two handles on the same compiler
    /// are equal. Mutation: make `spills` compare `local_bytes` with 16; the 16-byte probe holds.
    #[test]
    fn local_memory_and_a_failed_compile_refuse_the_probe() {
        let spilled = RegisterFeedback::new(counting(16));
        assert!(!spilled.holds(&probe("a")));
        let clean = RegisterFeedback::new(counting(0));
        assert!(clean.holds(&probe("a")));
        let failing = RegisterFeedback::new(FakeCompiledRegisters {
            compiler: CUDA,
            verdict: Box::new(|probe| {
                Err(FeedbackError::Compile {
                    name: probe.body.name.clone(),
                    reason: "refused".into(),
                })
            }),
            compiles: Default::default(),
        });
        assert!(!failing.holds(&probe("a")));
        assert_eq!(spilled, clean, "the same compiler");
    }
}
