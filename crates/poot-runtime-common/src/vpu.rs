//! [`VpuGraphArtifact`]: the host side of a region the Intel NPU's vendor compiler turns into a blob.
//!
//! An Intel NPU has no kernel ISA. The artifact `poot-codegen` produces for it is OpenVINO IR (the model
//! text and the constants it points into) plus the region's argument ABI: the graph's inputs and then its
//! outputs, by name, storage and shape. The device compiles the IR with the driver's graph compiler, then
//! checks the compiled graph's own argument list against this ABI before it runs anything, so a vendor
//! compiler that reordered, retyped or reshaped an argument is a typed refusal and never a wrong answer.

use poot_target::{BufferStorage, VpuTargetId};

/// One argument of a region: the tensor the graph reads or writes there.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VpuBinding {
    name: Box<str>,
    storage: BufferStorage,
    shape: Box<[usize]>,
}

impl VpuBinding {
    /// A binding named `name` (the IR layer's name) holding a dense tensor of `storage` and `shape`.
    pub fn new(
        name: impl Into<Box<str>>,
        storage: BufferStorage,
        shape: impl Into<Box<[usize]>>,
    ) -> Self {
        Self {
            name: name.into(),
            storage,
            shape: shape.into(),
        }
    }

    /// The IR layer's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The storage of the host buffer bound here.
    pub fn storage(&self) -> BufferStorage {
        self.storage
    }

    /// The tensor's extents, outermost first.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// The tensor's element count.
    pub fn elements(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(1u64, |n, &d| n.checked_mul(u64::try_from(d).ok()?))
    }
}

/// One compute layer of a region's IR and the poot equation it lowers, so a layer the vendor compiler rejects
/// names the equation that produced it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VpuLayer {
    name: Box<str>,
    origin: u32,
}

impl VpuLayer {
    /// The layer named `name` in the IR, lowered from the equation whose output has index `origin`.
    pub fn new(name: impl Into<Box<str>>, origin: u32) -> Self {
        Self {
            name: name.into(),
            origin,
        }
    }

    /// The IR layer's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The index of the poot equation's output value.
    pub const fn origin(&self) -> u32 {
        self.origin
    }
}

/// Why a region is not a loadable artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VpuArtifactError {
    /// The region has no output: it computes nothing.
    NoOutputs,
    /// Binding `index` (inputs first, then outputs) has a storage the NPU does not carry.
    Storage {
        /// The binding position.
        index: usize,
        /// The storage it declared.
        storage: BufferStorage,
    },
    /// Binding `index` has a zero extent or an element count that overflows.
    Shape {
        /// The binding position.
        index: usize,
    },
}

impl std::fmt::Display for VpuArtifactError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VpuArtifactError::NoOutputs => f.write_str("a region with no output computes nothing"),
            VpuArtifactError::Storage { index, storage } => write!(
                f,
                "binding {index} has storage {storage:?}, which the NPU does not carry (F16, F32 and I32 dense)"
            ),
            VpuArtifactError::Shape { index } => {
                write!(f, "binding {index} has a zero extent or overflows")
            }
        }
    }
}

impl std::error::Error for VpuArtifactError {}

/// A region as the Intel NPU's device takes it: IR plus argument ABI, self-consistent by construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VpuGraphArtifact {
    xml: Box<[u8]>,
    weights: Box<[u8]>,
    target: VpuTargetId,
    inputs: Box<[VpuBinding]>,
    outputs: Box<[VpuBinding]>,
    layers: Box<[VpuLayer]>,
}

impl VpuGraphArtifact {
    /// An artifact for `target` from the IR text and constants and the region's `inputs` and `outputs`.
    ///
    /// # Errors
    ///
    /// [`VpuArtifactError`] when a binding has a storage or shape the device cannot bind, or the region has
    /// no output.
    pub fn new(
        target: VpuTargetId,
        xml: impl Into<Box<[u8]>>,
        weights: impl Into<Box<[u8]>>,
        inputs: Vec<VpuBinding>,
        outputs: Vec<VpuBinding>,
        layers: Vec<VpuLayer>,
    ) -> Result<Self, VpuArtifactError> {
        if outputs.is_empty() {
            return Err(VpuArtifactError::NoOutputs);
        }
        for (index, binding) in inputs.iter().chain(&outputs).enumerate() {
            let carried = [
                BufferStorage::f16(),
                BufferStorage::f32(),
                BufferStorage::i32(),
            ];
            if !carried.contains(&binding.storage) {
                return Err(VpuArtifactError::Storage {
                    index,
                    storage: binding.storage,
                });
            }
            match binding.elements() {
                Some(n) if n > 0 => {}
                _ => return Err(VpuArtifactError::Shape { index }),
            }
        }
        Ok(Self {
            xml: xml.into(),
            weights: weights.into(),
            target,
            inputs: inputs.into(),
            outputs: outputs.into(),
            layers: layers.into(),
        })
    }

    /// The OpenVINO IR model text.
    pub fn xml(&self) -> &[u8] {
        &self.xml
    }

    /// The constants the model text points into.
    pub fn weights(&self) -> &[u8] {
        &self.weights
    }

    /// The NPU part the IR was emitted for.
    pub fn target(&self) -> VpuTargetId {
        self.target
    }

    /// The region's inputs, in graph argument order.
    pub fn inputs(&self) -> &[VpuBinding] {
        &self.inputs
    }

    /// The region's outputs, in graph argument order after the inputs.
    pub fn outputs(&self) -> &[VpuBinding] {
        &self.outputs
    }

    /// The region's compute layers, in program order.
    pub fn layers(&self) -> &[VpuLayer] {
        &self.layers
    }

    /// A 64-bit FNV-1a digest of the IR, the constants and the ABI: what a dispatch's invocation carries so
    /// the device can tell it names this artifact.
    pub fn digest(&self) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        let mut feed = |bytes: &[u8]| {
            for &b in bytes {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
            // A separator no length can forge: each field is length-prefixed.
            for &b in &(bytes.len() as u64).to_le_bytes() {
                hash ^= u64::from(b);
                hash = hash.wrapping_mul(0x0100_0000_01b3);
            }
        };
        feed(&self.xml);
        feed(&self.weights);
        for binding in self.inputs.iter().chain(&self.outputs) {
            feed(binding.name.as_bytes());
        }
        hash
    }
}
