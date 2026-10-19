/// A Level Zero runtime failure. [`L0Error::LibraryNotFound`] and [`L0Error::NoDevice`] are the typed
/// "this machine has no Level Zero NPU" signals a device row turns into a skip.
#[derive(Debug, thiserror::Error)]
pub enum L0Error {
    /// `libze_loader.so.1` is not on the loader path. `0` carries the OS `dlopen` message.
    #[error("libze_loader.so.1 not found: {0}")]
    LibraryNotFound(String),
    /// The loader library lacks an entry point this runtime calls.
    #[error("the Level Zero loader has no symbol `{0}`")]
    MissingSymbol(&'static str),
    /// A Level Zero call returned a `ze_result_t` other than success.
    #[error("{call} returned ze_result_t 0x{result:x}")]
    Driver {
        /// The Level Zero function that failed.
        call: &'static str,
        /// The raw `ze_result_t`.
        result: u32,
    },
    /// The loader found no driver with an NPU device (no Intel NPU driver on the library path, or no NPU).
    #[error("no Level Zero NPU device is visible")]
    NoDevice,
    /// The device has no compute command queue group.
    #[error("the Level Zero device has no compute command queue group")]
    NoComputeQueue,
    /// A copy between buffers of different lengths.
    #[error("copy of a {host}-byte buffer into a {device}-byte buffer")]
    TransferLength {
        /// The host slice length in bytes.
        host: usize,
        /// The device buffer length in bytes.
        device: usize,
    },
    /// An execution bound a different number of buffers than the graph has arguments.
    #[error("graph takes {expected} arguments but the execution bound {given}")]
    ArgumentCount {
        /// The graph's argument count.
        expected: u32,
        /// The number of buffers the execution passed.
        given: usize,
    },
    /// The driver does not provide an extension this runtime needs.
    #[error("the Level Zero driver has no `{0}` extension")]
    MissingExtension(&'static str),
    /// The driver's graph compiler rejected a graph. `log` is the build log it left, which names the layer.
    #[error("graph creation returned ze_result_t 0x{result:x}: {log}")]
    GraphBuild {
        /// The raw `ze_result_t`.
        result: u32,
        /// The compiler's build log (empty when the driver kept none).
        log: String,
    },
    /// An operation on a recording that was already finished.
    #[error("the recording is already closed")]
    RecordingClosed,
    /// A graph's build flags or an argument name contained an interior NUL byte.
    #[error("a graph string contains a NUL byte")]
    GraphString,
    /// The queue did not drain within the allowed time.
    #[error("the Level Zero queue did not finish within {millis} ms")]
    Timeout {
        /// The wait that expired, in milliseconds.
        millis: u64,
    },
    /// A byte range outside a host buffer.
    #[error("range {start}..{end} is outside a {len}-byte host buffer")]
    HostRange {
        /// The start of the range.
        start: usize,
        /// The end of the range.
        end: usize,
        /// The buffer length.
        len: usize,
    },
    /// A buffer, graph or recording created by one context was given to another.
    #[error("{what} belongs to a different Level Zero context")]
    ForeignResource {
        /// Which resource kind was foreign.
        what: &'static str,
    },
}

/// Map a `ze_result_t` to `Ok(())` or a [`L0Error::Driver`] naming the failing call.
pub(crate) fn check(call: &'static str, result: u32) -> Result<(), L0Error> {
    if result == crate::ffi::ZE_RESULT_SUCCESS {
        Ok(())
    } else {
        Err(L0Error::Driver { call, result })
    }
}
