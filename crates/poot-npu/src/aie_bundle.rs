//! The blob a linked AIE artifact is stored as in the kernel cache: one self-describing, length-prefixed
//! bundle of the xclbin, the instruction stream and the shape ABI, re-validated by [`AieLinkedArtifact::new`]
//! when it is read back (POOT-664).

use poot_npu_artifacts::{AieArtifactError, AieBinding, AieLinkedArtifact, AieShapeAbi};
use poot_target::{AieTargetId, ArgAccess, BufferStorage, ElementKind};

const AIE_BUNDLE_MAGIC: &[u8; 8] = b"POOTAIE2";

/// Serialize `artifact` as one blob. Every field is length-prefixed, so [`decode_bundle`] is the exact inverse.
pub fn encode_bundle(artifact: &AieLinkedArtifact) -> Vec<u8> {
    let (xclbin, instructions) = (artifact.xclbin(), artifact.instructions());
    let target = artifact.target();
    let abi = artifact.abi();
    let mut out = Vec::with_capacity(xclbin.len() + instructions.len() + 128);
    out.extend_from_slice(AIE_BUNDLE_MAGIC);
    put_bytes(&mut out, xclbin);
    put_bytes(&mut out, instructions);
    put_bytes(&mut out, artifact.entry().as_bytes());
    put_bytes(&mut out, artifact.toolchain().as_bytes());
    put_u64(&mut out, u64::from(target.abi_revision()));
    put_bytes(&mut out, target.canonical().as_bytes());
    put_u64(&mut out, artifact.tile_local_bytes());
    put_u64(&mut out, abi.bindings().len() as u64);
    for binding in abi.bindings() {
        out.push(match binding.storage().element() {
            ElementKind::F32 => 0,
            _ => 1,
        });
        out.push(u8::from(binding.access() == ArgAccess::Write));
        put_u64(&mut out, binding.accessed_elements());
        put_u64(&mut out, binding.dims().len() as u64);
        for &dim in binding.dims() {
            put_u64(&mut out, dim);
        }
    }
    out
}

/// Parse and re-validate a blob [`encode_bundle`] wrote for `target`. A blob for a different target (ISA,
/// platform or ABI revision) is malformed here, not loaded.
///
/// # Errors
///
/// [`AieArtifactError`] when the blob is truncated, malformed, for another target, or not a valid artifact.
pub fn decode_bundle(
    bytes: &[u8],
    target: AieTargetId,
) -> Result<AieLinkedArtifact, AieArtifactError> {
    let mut r = Reader(bytes);
    if r.take(AIE_BUNDLE_MAGIC.len())? != AIE_BUNDLE_MAGIC {
        return Err(AieArtifactError::Malformed("bundle magic"));
    }
    let xclbin = r.bytes()?.to_vec();
    let instructions = r.bytes()?.to_vec();
    let entry = std::str::from_utf8(r.bytes()?)
        .map_err(|_| AieArtifactError::Malformed("entry is not UTF-8"))?
        .to_owned();
    let toolchain = std::str::from_utf8(r.bytes()?)
        .map_err(|_| AieArtifactError::Malformed("toolchain is not UTF-8"))?
        .to_owned();
    let revision = r.u64()?;
    let recorded = r.bytes()?;
    if revision != u64::from(target.abi_revision()) || recorded != target.canonical().as_bytes() {
        return Err(AieArtifactError::Malformed("built for a different target"));
    }
    let tile_local_bytes = r.u64()?;
    let count = r.u64()?;
    let mut bindings = Vec::new();
    for index in
        0..usize::try_from(count).map_err(|_| AieArtifactError::Malformed("binding count"))?
    {
        let &[element, write] = r.take(2)? else {
            return Err(AieArtifactError::Malformed("binding header"));
        };
        let storage = match element {
            0 => BufferStorage::f32(),
            1 => BufferStorage::i32(),
            _ => return Err(AieArtifactError::Malformed("binding element kind")),
        };
        let accessed = r.u64()?;
        let rank = r.u64()?;
        let dims = (0..rank).map(|_| r.u64()).collect::<Result<Vec<_>, _>>()?;
        let access = match write {
            0 => ArgAccess::Read,
            1 => ArgAccess::Write,
            _ => return Err(AieArtifactError::Malformed("binding access")),
        };
        bindings.push(AieBinding::new(index, storage, access, &dims, accessed)?);
    }
    if !r.0.is_empty() {
        return Err(AieArtifactError::Malformed("trailing bytes"));
    }
    AieLinkedArtifact::new(
        xclbin,
        instructions,
        &entry,
        AieShapeAbi::new(bindings)?,
        target,
        &toolchain,
        tile_local_bytes,
    )
}

fn put_u64(out: &mut Vec<u8>, v: u64) {
    out.extend_from_slice(&v.to_le_bytes());
}

fn put_bytes(out: &mut Vec<u8>, bytes: &[u8]) {
    put_u64(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

struct Reader<'a>(&'a [u8]);

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], AieArtifactError> {
        if self.0.len() < n {
            return Err(AieArtifactError::Malformed("truncated"));
        }
        let (head, rest) = self.0.split_at(n);
        self.0 = rest;
        Ok(head)
    }

    fn u64(&mut self) -> Result<u64, AieArtifactError> {
        let raw = self.take(8)?;
        Ok(u64::from_le_bytes(raw.try_into().expect("8 bytes")))
    }

    fn bytes(&mut self) -> Result<&'a [u8], AieArtifactError> {
        let len = usize::try_from(self.u64()?)
            .map_err(|_| AieArtifactError::Malformed("length does not fit"))?;
        self.take(len)
    }
}
