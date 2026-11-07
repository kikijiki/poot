//! What each device brings to a compile for it (POOT-1505): its NPU lowering and its compiler's register feedback.
//! These are facts of a device, never a caller's policy, so they reach a compile only from the executor that drives
//! the device ([`CompileOptions::with_device`]) and `compile_staged` gives each stage its own device's.

use crate::multi_device::DeviceId;
use crate::{CompileOptions, NpuHandle, RegisterFeedback};

/// One device's compile facts.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceFacts {
    /// The NPU's lowering ([`CompileOptions::npu`]); `None` for a GPU.
    pub npu: Option<NpuHandle>,
    /// The device compiler's register feedback ([`CompileOptions::register_feedback`]); `None` when the device's
    /// compiler cannot be asked, and the plan is the register model's.
    pub register_feedback: Option<RegisterFeedback>,
}

/// The compile facts of every device an executor drives, by device. A device without an entry has none.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DeviceOptions(Vec<(DeviceId, DeviceFacts)>);

impl DeviceOptions {
    /// No device: options no executor has filled ([`CompileOptions::new`]).
    pub const EMPTY: Self = Self(Vec::new());

    #[cfg(any(test, feature = "test-support"))]
    /// `devices` with no facts: a GPU-free tool (the plan summary, the census, a planning device) that has no
    /// device compiler to ask, stated on purpose. Its plans record no compiler.
    pub fn none(devices: impl IntoIterator<Item = DeviceId>) -> Self {
        Self(
            devices
                .into_iter()
                .map(|device| (device, DeviceFacts::default()))
                .collect(),
        )
    }

    #[cfg(any(test, feature = "test-support"))]
    /// [`Self::none`] for every device of `targets`.
    pub fn none_for(targets: &crate::staged::TargetSet) -> Self {
        Self::none(targets.devices().iter().map(|&(device, _)| device))
    }

    /// One device's facts.
    pub fn single(device: DeviceId, facts: DeviceFacts) -> Self {
        Self(vec![(device, facts)])
    }

    /// The facts of every device of `self` and of `other` (an executor over several devices).
    #[must_use]
    pub fn union(mut self, other: Self) -> Self {
        for (device, facts) in other.0 {
            match self.0.iter_mut().find(|(known, _)| *known == device) {
                Some(entry) => entry.1 = facts,
                None => self.0.push((device, facts)),
            }
        }
        self
    }

    /// `device`'s facts: none when it has no entry.
    pub fn of(&self, device: DeviceId) -> DeviceFacts {
        self.0
            .iter()
            .find(|(known, _)| *known == device)
            .map(|(_, facts)| facts.clone())
            .unwrap_or_default()
    }

    /// The devices with an entry, in order.
    pub fn devices(&self) -> impl Iterator<Item = DeviceId> + '_ {
        self.0.iter().map(|(device, _)| *device)
    }

    /// Whether no device has an entry: options no executor has filled.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl CompileOptions {
    /// These options for the devices `devices` describes: their facts replace whatever the options held, so a caller
    /// can neither inject a compiler nor leave the device's out. With one device its facts are the options' own (a
    /// single-target [`crate::compile`]); with several, `compile_staged` gives each stage its device's.
    #[must_use]
    pub fn with_device(mut self, devices: DeviceOptions) -> Self {
        let only = match devices.0.as_slice() {
            [(_, facts)] => facts.clone(),
            _ => DeviceFacts::default(),
        };
        self.npu = only.npu;
        self.register_feedback = only.register_feedback;
        self.devices = devices;
        self
    }

    /// These options for a stage on `device`: that device's facts when the options name devices, else as they are.
    pub(crate) fn for_device(&self, device: DeviceId) -> Self {
        if self.devices.is_empty() {
            return self.clone();
        }
        let facts = self.devices.of(device);
        Self {
            npu: facts.npu,
            register_feedback: facts.register_feedback,
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{CompileError, FakeCompiledRegisters, RegisterFeedback, RegisterVerdict, Target};
    use poot_target::{AmdArch, Backend, DeviceCaps, KernelCompiler};

    fn feedback(compiler: KernelCompiler) -> RegisterFeedback {
        RegisterFeedback::new(FakeCompiledRegisters {
            compiler,
            verdict: Box::new(|_| {
                Ok(RegisterVerdict {
                    registers: 32,
                    local_bytes: 0,
                })
            }),
            compiles: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    fn gpu(backend: Backend) -> Target {
        Target::Gpu {
            backend,
            caps: DeviceCaps::ptx_default(),
        }
    }

    /// POOT-1505: a feedback plans only for a target of its compiler's backend; the CUDA driver's verdicts on a
    /// SPIR-V or AMDGPU stage, or clang's on a PTX one, are refused by name, never used.
    /// Mutation: make `KernelCompiler::compiles_for` hold for every pair; the CUDA feedback on the SPIR-V target reads
    /// `Ok`.
    #[test]
    fn a_feedback_for_another_backend_is_refused() {
        const CUDA: KernelCompiler = KernelCompiler::CudaDriver {
            release: 13_020,
            compute_capability: (12, 0),
        };
        const CLANG: KernelCompiler = KernelCompiler::AmdClang {
            version: (23, 0, 0),
            commit: 0x4321_5c73_116c_4077,
            gfx: 0x1151,
        };
        let amd = Backend::AmdGcn(AmdArch::gfx1151());
        for (compiler, backend, holds) in [
            (CUDA, Backend::Nvptx, true),
            (CUDA, Backend::SpirvVulkan, false),
            (CUDA, amd, false),
            (CLANG, amd, true),
            (CLANG, Backend::Nvptx, false),
        ] {
            let result =
                crate::compile::refuse_foreign_feedback(&gpu(backend), Some(&feedback(compiler)));
            match (holds, result) {
                (true, Ok(())) => {}
                (
                    false,
                    Err(CompileError::FeedbackForAnotherBackend {
                        compiler: c,
                        backend: b,
                    }),
                ) => {
                    assert_eq!((c, b), (compiler, backend));
                }
                (_, other) => panic!("{compiler:?} on {backend:?}: {other:?}"),
            }
        }
    }
}
