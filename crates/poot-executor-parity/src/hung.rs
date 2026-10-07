//! A device that reports the routed-expert hang fact, for rows that need the guarded program on a
//! backend whose real target does not carry it (wgpu).
//!
//! `compile` skips the hang passes only where the target sets the fact and the graph has a routed-expert
//! site, and an engine loads only a program compiled for its own device's target. A row that compares a
//! fused program with the guarded one (one kernel per equation, no fusion) therefore compiles the guarded
//! program for the target [`WithRoutedHang`] reports and runs it on an engine over this wrapper; the fused
//! program runs on an engine over the bare device. Everything but [`Device::target`] is the inner device.

use poot_executor::{Arg, BufferRole, Device, DeviceTime, Dispatch, MemoryCounterSnapshot};
use poot_graph_plan::{Submission, Target};
use poot_runtime_common::CompiledKernel;
use poot_target::BufferStorage;

/// `target` as a device with the routed-expert hang would report it.
pub fn with_hang(target: Target) -> Target {
    Target {
        caps: poot_test_util::graph_fixtures::with_routed_expert_hang(target.caps),
        ..target
    }
}

/// `D`, reporting the routed-expert hang in its target.
pub struct WithRoutedHang<D>(pub D);

impl<D: Device> Device for WithRoutedHang<D> {
    type Buffer = D::Buffer;
    type Kernel = D::Kernel;
    type Recording = D::Recording;
    type Error = D::Error;

    fn target(&self) -> Target {
        with_hang(self.0.target())
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.0.memory()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<Self::Buffer, Self::Error> {
        self.0.allocate(role, storage, elems)
    }

    fn load_kernel(
        &mut self,
        key: &str,
        kernel: CompiledKernel,
    ) -> Result<Self::Kernel, Self::Error> {
        self.0.load_kernel(key, kernel)
    }

    fn begin(&mut self, submission: Submission) -> Result<(), Self::Error> {
        self.0.begin(submission)
    }

    fn dispatch(&mut self, dispatch: Dispatch<'_, Self>) -> Result<(), Self::Error> {
        let inputs: Vec<Arg<'_, D>> = dispatch
            .inputs
            .iter()
            .map(|a| Arg {
                buffer: a.buffer,
                elems: a.elems,
            })
            .collect();
        self.0.dispatch(Dispatch {
            kernel: dispatch.kernel,
            inputs: &inputs,
            output: Arg {
                buffer: dispatch.output.buffer,
                elems: dispatch.output.elems,
            },
            threads: dispatch.threads,
            workgroup: dispatch.workgroup,
            work: dispatch.work,
        })
    }

    fn copy(&mut self, src: &Self::Buffer, dst: &Self::Buffer) -> Result<(), Self::Error> {
        self.0.copy(src, dst)
    }

    fn finish(&mut self) -> Result<Option<Self::Recording>, Self::Error> {
        self.0.finish()
    }

    fn replay(&mut self, recording: &Self::Recording) -> Result<(), Self::Error> {
        self.0.replay(recording)
    }

    fn write(&mut self, dst: &Self::Buffer, bytes: &[u8]) -> Result<(), Self::Error> {
        self.0.write(dst, bytes)
    }

    fn read(&mut self, src: &Self::Buffer, out: &mut [u8]) -> Result<(), Self::Error> {
        self.0.read(src, out)
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        self.0.synchronize()
    }

    fn device_time(&self) -> DeviceTime {
        self.0.device_time()
    }

    fn abort(&mut self) -> Result<(), Self::Error> {
        self.0.abort()
    }

    fn classify_fault(&self, error: &Self::Error) -> Option<(String, u32)> {
        self.0.classify_fault(error)
    }
}
