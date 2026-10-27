//! [`ArgAccess`]: whether a kernel argument is read or written, shared by the kernel argument schema
//! (`poot-codegen`) and the AIE shape ABI (`poot-npu-artifacts`).

/// Whether a kernel argument is read, or written back to the caller's buffer.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ArgAccess {
    /// The kernel only reads the buffer.
    Read,
    /// The kernel writes the buffer; the caller reads it back after the dispatch.
    Write,
}
