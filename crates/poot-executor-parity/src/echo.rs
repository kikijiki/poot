//! A model-free [`Device`] that computes nothing and answers every readback with bytes the test set: the
//! "device" side of a parity row that runs without a GPU. A test sets the answer from inside the oracle
//! callback (which `run_parity` calls before each step), so the real `Engine`, compile and
//! `run_parity` wiring run while the device's result is exactly what the test chose.

use std::ops::Range;
use std::sync::{Arc, Mutex};

use poot_executor::{Device, DeviceTime, Dispatch, Engine, Executor};
use poot_graph_plan::Target;
use poot_runtime_common::{BufferRole, MemoryCounterSnapshot};
use poot_target::{Backend, BufferStorage};
use poot_tensor::HostTensor;
use poot_test_util::device_caps::default_caps_for;

/// The bytes the [`EchoDevice`] answers a readback with; the test side of the pair.
#[derive(Clone, Default)]
pub struct Answer(Arc<Mutex<Vec<u8>>>);

impl Answer {
    /// Answer the next readbacks with `tensor`'s bytes.
    pub fn set(&self, tensor: &HostTensor) {
        *self.0.lock().unwrap() = tensor.view().bytes().to_vec();
    }
}

/// A device that dispatches nothing and reads back the [`Answer`].
pub struct EchoDevice {
    answer: Answer,
    memory: poot_runtime_common::MemoryCounters,
    recording: bool,
}

/// The echo device never fails; this is the type its `Device::Error` names.
#[derive(Debug)]
pub struct EchoError;

impl std::fmt::Display for EchoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("echo device error (never raised)")
    }
}

impl std::error::Error for EchoError {}

/// An executor over an [`EchoDevice`], its compile target, and the answer handle.
pub fn echo_executor() -> (Box<dyn Executor>, Target, Answer) {
    let answer = Answer::default();
    let device = EchoDevice {
        answer: answer.clone(),
        memory: poot_runtime_common::MemoryCounters::default(),
        recording: false,
    };
    let target = device.target();
    (Box::new(Engine::new(device)), target, answer)
}

impl Device for EchoDevice {
    type Buffer = ();
    type Kernel = ();
    type Recording = ();
    type Error = EchoError;

    fn copy_mechanism(&mut self) -> poot_executor::CopyMechanism<'_, Self::Error> {
        poot_executor::CopyMechanism::Native
    }
    fn backend_name(&self) -> &'static str {
        "fake"
    }

    fn target(&self) -> Target {
        Target::Gpu {
            backend: Backend::SpirvVulkan,
            caps: default_caps_for(Backend::SpirvVulkan),
        }
    }

    fn memory(&self) -> Vec<(BufferRole, MemoryCounterSnapshot)> {
        self.memory.snapshot_all()
    }

    fn allocate(
        &mut self,
        role: BufferRole,
        storage: BufferStorage,
        elems: usize,
    ) -> Result<(), EchoError> {
        let bytes = (elems * storage.element().byte_width()) as u64;
        std::mem::forget(self.memory.record_alloc(role, bytes));
        Ok(())
    }

    fn buffer_elems(&self, _buffer: &()) -> usize {
        usize::MAX
    }

    fn load_kernel(
        &mut self,
        _key: &str,
        _kernel: poot_runtime_common::CompiledKernel,
    ) -> Result<(), EchoError> {
        Ok(())
    }

    fn begin(&mut self) -> Result<(), EchoError> {
        self.recording = true;
        Ok(())
    }

    fn dispatch(&mut self, _dispatch: Dispatch<'_, Self>) -> Result<(), EchoError> {
        Ok(())
    }

    fn copy(&mut self, _src: &(), _dst: &()) -> Result<(), EchoError> {
        Ok(())
    }

    fn finish(&mut self) -> Result<(), EchoError> {
        self.recording = false;
        Ok(())
    }

    fn replay(&mut self, _recording: &()) -> Result<(), EchoError> {
        Ok(())
    }

    fn write_with(
        &mut self,
        _dst: &(),
        _offset: usize,
        _len: usize,
        _fill: &mut dyn FnMut(Range<usize>, &mut [u8]),
    ) -> Result<(), EchoError> {
        Ok(())
    }

    fn read(&mut self, _src: &(), out: &mut [u8]) -> Result<(), EchoError> {
        let answer = self.answer.0.lock().unwrap();
        let n = out.len().min(answer.len());
        out[..n].copy_from_slice(&answer[..n]);
        out[n..].fill(0);
        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), EchoError> {
        Ok(())
    }

    fn device_time(&self) -> DeviceTime {
        DeviceTime::Unknown
    }

    fn abort(&mut self) -> Result<(), EchoError> {
        Ok(())
    }
}
