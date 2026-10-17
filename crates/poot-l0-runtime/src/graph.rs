//! Vendor-compiled graphs on a Level Zero NPU (`ZE_extension_graph`).
//!
//! An Intel NPU has no kernel ISA the runtime can target. Its only entry point is the driver's graph
//! compiler: OpenVINO IR goes in, a native blob comes out, and the blob is the thing the device runs. A
//! graph is created from IR (the driver compiles it, [`L0Context::compile_graph`]) or from a blob compiled
//! earlier ([`L0Context::load_native_graph`]); [`L0Graph::native_binary`] returns the blob, so a caller can
//! cache it. A recording ([`L0Recording`]) is a closed command list of graph executions and host copies: the
//! arguments are bound when an execution is appended, and the list replays any number of times.

use std::ffi::{CStr, CString, c_char};
use std::ptr::null_mut;
use std::rc::Rc;
use std::time::Duration;

use crate::buffer::L0HostBuffer;
use crate::context::L0Context;
use crate::error::{L0Error, check};
use crate::ffi::{
    CommandListDesc, DeviceGraphProperties2, GraphArgumentProperties, GraphDesc2, GraphProperties2,
    Handle, ZE_GRAPH_FLAG_DISABLE_CACHING, ZE_GRAPH_FORMAT_NATIVE, ZE_GRAPH_FORMAT_NGRAPH_LITE,
    ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE, ZE_GRAPH_STAGE_INITIALIZE, ZE_RESULT_NOT_READY,
};

/// OpenVINO IR: the model XML and the weights its constants point into.
pub struct GraphIr<'a> {
    /// The model XML.
    pub xml: &'a [u8],
    /// The weights file (`.bin`) the XML's constants index; empty when every constant is inline.
    pub weights: &'a [u8],
}

/// What the driver's graph compiler reports for the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphDeviceProperties {
    /// The graph extension version (`ZE_MAKE_VERSION`: major in the high half, minor in the low).
    pub extension_version: u32,
    /// The driver compiler's version, `(major, minor)`. Part of every blob's identity.
    pub compiler_version: (u16, u16),
    /// The highest OpenVINO opset the compiler supports.
    pub max_opset: u32,
}

/// The precision of a graph argument, as the compiled executable stores it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphPrecision {
    /// IEEE binary16.
    Fp16,
    /// IEEE binary32.
    Fp32,
    /// Signed 32-bit integer.
    Int32,
    /// Any other `ze_graph_argument_precision_t`, by its raw value.
    Other(u32),
}

impl GraphPrecision {
    fn from_raw(raw: u32) -> Self {
        match raw {
            0x02 => GraphPrecision::Fp16,
            0x01 => GraphPrecision::Fp32,
            0x05 => GraphPrecision::Int32,
            other => GraphPrecision::Other(other),
        }
    }
}

/// Whether a graph argument is read or written by the graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphDirection {
    /// The graph reads it.
    Input,
    /// The graph writes it.
    Output,
}

/// One argument of a compiled graph, as the driver reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphArgument {
    /// The name the IR gave the `Parameter` or `Result` layer.
    pub name: String,
    /// Whether the graph reads or writes it.
    pub direction: GraphDirection,
    /// The tensor extents, outermost first, without the driver's padding dimensions.
    pub dims: Vec<u32>,
    /// The precision of the compiled executable's argument.
    pub precision: GraphPrecision,
}

struct GraphInner {
    context: L0Context,
    handle: Handle,
    arguments: Vec<GraphArgument>,
}

/// A graph the driver created (`zeGraphCreate2`), destroyed when the last clone drops.
#[derive(Clone)]
pub struct L0Graph(Rc<GraphInner>);

impl L0Graph {
    /// The graph's arguments, inputs first and then outputs, in the order [`L0Recording`] binds buffers.
    pub fn arguments(&self) -> &[GraphArgument] {
        &self.0.arguments
    }

    /// The compiled blob, which [`L0Context::load_native_graph`] loads without compiling again.
    pub fn native_binary(&self) -> Result<Vec<u8>, L0Error> {
        let table = self.0.context.graph_table()?;
        let get = table
            .get_native_binary
            .ok_or(L0Error::MissingSymbol("pfnGetNativeBinary"))?;
        let mut size = 0usize;
        // SAFETY: the graph is live; a null buffer asks for the size only.
        check("zeGraphGetNativeBinary", unsafe {
            get(self.0.handle, &mut size, null_mut())
        })?;
        let mut blob = vec![0u8; size];
        // SAFETY: `blob` has room for `size` bytes, the size the driver just reported.
        check("zeGraphGetNativeBinary", unsafe {
            get(self.0.handle, &mut size, blob.as_mut_ptr())
        })?;
        blob.truncate(size);
        Ok(blob)
    }
}

impl Drop for GraphInner {
    fn drop(&mut self) {
        let Ok(table) = self.context.graph_table() else {
            return;
        };
        let Some(destroy) = table.destroy else {
            return;
        };
        // SAFETY: `handle` came from a successful `zeGraphCreate2` and is destroyed exactly once, here; every
        // recording that executes it holds a clone, so none is left in flight.
        let result = unsafe { destroy(self.handle) };
        if let Err(error) = check("zeGraphDestroy", result) {
            tracing::warn!(%error, "leaking a Level Zero graph");
        }
    }
}

impl L0Context {
    /// What the driver's graph compiler reports for the device.
    pub fn graph_device_properties(&self) -> Result<GraphDeviceProperties, L0Error> {
        let table = self.graph_table()?;
        let get = table
            .device_get_graph_properties_2
            .ok_or(L0Error::MissingSymbol("pfnDeviceGetGraphProperties2"))?;
        let mut properties = DeviceGraphProperties2::new();
        // SAFETY: the device is live and `properties` is a correctly sized, typed structure.
        check("zeDeviceGetGraphProperties2", unsafe {
            get(self.device(), &mut properties)
        })?;
        Ok(GraphDeviceProperties {
            extension_version: properties.graph_extension_version,
            compiler_version: (properties.compiler_major, properties.compiler_minor),
            max_opset: properties.max_ov_opset_version_supported,
        })
    }

    /// Compile `ir` with the driver's graph compiler under `build_flags` (the compiler's option string) and
    /// create the graph. The driver's own disk cache is bypassed, so a caller's cache is the only one.
    ///
    /// # Errors
    ///
    /// [`L0Error::GraphBuild`] with the compiler's build log when the compiler rejects the IR.
    pub fn compile_graph(&self, ir: &GraphIr<'_>, build_flags: &str) -> Result<L0Graph, L0Error> {
        let version = self.graph_device_properties()?.compiler_version;
        let packed = pack_ir(version, ir);
        self.create_graph(ZE_GRAPH_FORMAT_NGRAPH_LITE, &packed, build_flags)
    }

    /// Create a graph from a blob [`L0Graph::native_binary`] returned.
    pub fn load_native_graph(&self, blob: &[u8]) -> Result<L0Graph, L0Error> {
        self.create_graph(ZE_GRAPH_FORMAT_NATIVE, blob, "")
    }

    fn create_graph(
        &self,
        format: u32,
        input: &[u8],
        build_flags: &str,
    ) -> Result<L0Graph, L0Error> {
        let table = self.graph_table()?;
        let create = table.create_2.ok_or(L0Error::MissingSymbol("pfnCreate2"))?;
        let flags = CString::new(build_flags).map_err(|_| L0Error::GraphString)?;
        let desc = GraphDesc2::new(format, input, &flags, ZE_GRAPH_FLAG_DISABLE_CACHING);
        let mut handle = null_mut();
        // SAFETY: the context and device are live; `desc` and the buffers it points into outlive the call.
        let result = unsafe { create(self.handle(), self.device(), &desc, &mut handle) };
        if result != crate::ffi::ZE_RESULT_SUCCESS {
            return Err(L0Error::GraphBuild {
                result,
                log: self.last_build_log(),
            });
        }
        let mut inner = GraphInner {
            context: self.clone(),
            handle,
            arguments: Vec::new(),
        };
        inner.initialize()?;
        Ok(L0Graph(Rc::new(inner)))
    }

    /// The build log of the last failed graph creation on this thread (the driver keeps one).
    fn last_build_log(&self) -> String {
        let Ok(table) = self.graph_table() else {
            return String::new();
        };
        let Some(get) = table.build_log_get_string else {
            return String::new();
        };
        let mut size = 0u32;
        // SAFETY: a null graph handle asks for the thread's last failure log; a null buffer asks for its size.
        let sized = unsafe { get(null_mut(), &mut size, null_mut()) };
        if sized != crate::ffi::ZE_RESULT_SUCCESS || size == 0 {
            return String::new();
        }
        let mut log = vec![0u8; size as usize];
        // SAFETY: `log` has room for the `size` bytes the driver reported.
        let read = unsafe { get(null_mut(), &mut size, log.as_mut_ptr().cast::<c_char>()) };
        if read != crate::ffi::ZE_RESULT_SUCCESS {
            return String::new();
        }
        log.truncate(size as usize);
        let end = log.iter().position(|&b| b == 0).unwrap_or(log.len());
        String::from_utf8_lossy(&log[..end]).trim().to_owned()
    }

    /// Ask the driver's graph compiler which layers of `ir` it can compile, without compiling it
    /// (`vclQueryNetwork`). Returns the supported layers' names, as the compiler lists them.
    pub fn query_supported_layers(
        &self,
        ir: &GraphIr<'_>,
        build_flags: &str,
    ) -> Result<Vec<String>, L0Error> {
        let table = self.graph_table()?;
        let create = table
            .query_network_create_2
            .ok_or(L0Error::MissingSymbol("pfnQueryNetworkCreate2"))?;
        let destroy = table
            .query_network_destroy
            .ok_or(L0Error::MissingSymbol("pfnQueryNetworkDestroy"))?;
        let layers = table
            .query_network_get_supported_layers
            .ok_or(L0Error::MissingSymbol("pfnQueryNetworkGetSupportedLayers"))?;
        let version = self.graph_device_properties()?.compiler_version;
        let packed = pack_ir(version, ir);
        let flags = CString::new(build_flags).map_err(|_| L0Error::GraphString)?;
        let desc = GraphDesc2::new(
            ZE_GRAPH_FORMAT_NGRAPH_LITE,
            &packed,
            &flags,
            ZE_GRAPH_FLAG_DISABLE_CACHING,
        );
        let mut query = null_mut();
        // SAFETY: the context and device are live; `desc` and the buffers it points into outlive the call.
        let created = unsafe { create(self.handle(), self.device(), &desc, &mut query) };
        if created != crate::ffi::ZE_RESULT_SUCCESS {
            return Err(L0Error::GraphBuild {
                result: created,
                log: self.last_build_log(),
            });
        }
        let names = (|| {
            let mut size = 0usize;
            // SAFETY: the query is live; a null buffer asks for the size only.
            check("zeGraphQueryNetworkGetSupportedLayers", unsafe {
                layers(query, &mut size, null_mut())
            })?;
            let mut text = vec![0u8; size];
            // SAFETY: `text` has room for the `size` bytes the driver reported.
            check("zeGraphQueryNetworkGetSupportedLayers", unsafe {
                layers(query, &mut size, text.as_mut_ptr().cast::<c_char>())
            })?;
            let end = text.iter().position(|&b| b == 0).unwrap_or(text.len());
            // The compiler lists the layers as `<name><name>...`.
            Ok::<_, L0Error>(
                String::from_utf8_lossy(&text[..end])
                    .split('<')
                    .filter_map(|part| part.strip_suffix('>'))
                    .map(str::to_owned)
                    .collect::<Vec<_>>(),
            )
        })();
        // SAFETY: `query` came from a successful create and is destroyed exactly once, here.
        let destroyed = unsafe { destroy(query) };
        if let Err(error) = check("zeGraphQueryNetworkDestroy", destroyed) {
            tracing::warn!(%error, "leaking a Level Zero graph query");
        }
        names
    }

    /// Start a recording: a command list that graph executions and copies are appended to.
    pub fn begin_recording(&self) -> Result<L0RecordingBuilder, L0Error> {
        self.graph_table()?;
        let desc = CommandListDesc::new(self.ordinal());
        let mut list = null_mut();
        // SAFETY: the context and device are live and `desc` outlives the call.
        check("zeCommandListCreate", unsafe {
            (self.funcs().command_list_create)(self.handle(), self.device(), &desc, &mut list)
        })?;
        Ok(L0RecordingBuilder {
            list: Some(OwnedList {
                context: self.clone(),
                list,
            }),
            graphs: Vec::new(),
            buffers: Vec::new(),
        })
    }

    /// Execute `recording` once and wait up to `timeout` for the queue to drain.
    ///
    /// # Errors
    ///
    /// [`L0Error::Timeout`] when the wait expires. The recording's work may still be running, so the caller
    /// must not trust the device or free the recording's buffers.
    pub fn execute(&self, recording: &L0Recording, timeout: Duration) -> Result<(), L0Error> {
        if !recording.0.list.context.same(self) {
            return Err(L0Error::ForeignResource { what: "recording" });
        }
        let mut lists = [recording.0.list.list];
        // SAFETY: the queue is live, `lists` holds one closed list, and no fence is used.
        check("zeCommandQueueExecuteCommandLists", unsafe {
            (self.funcs().command_queue_execute_command_lists)(
                self.queue(),
                1,
                lists.as_mut_ptr(),
                null_mut(),
            )
        })?;
        let nanos = u64::try_from(timeout.as_nanos()).unwrap_or(u64::MAX);
        // SAFETY: the queue is live.
        let waited = unsafe { (self.funcs().command_queue_synchronize)(self.queue(), nanos) };
        if waited == ZE_RESULT_NOT_READY {
            return Err(L0Error::Timeout {
                millis: u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX),
            });
        }
        check("zeCommandQueueSynchronize", waited)
    }
}

/// The IR the driver's compiler takes: the compiler version it expects, the number of inputs (the XML and the
/// weights), then each as a length and its bytes. Little-endian.
fn pack_ir(version: (u16, u16), ir: &GraphIr<'_>) -> Vec<u8> {
    let mut packed = Vec::with_capacity(4 + 4 + 8 + ir.xml.len() + 8 + ir.weights.len());
    packed.extend_from_slice(&version.0.to_le_bytes());
    packed.extend_from_slice(&version.1.to_le_bytes());
    packed.extend_from_slice(&2u32.to_le_bytes());
    packed.extend_from_slice(&(ir.xml.len() as u64).to_le_bytes());
    packed.extend_from_slice(ir.xml);
    packed.extend_from_slice(&(ir.weights.len() as u64).to_le_bytes());
    packed.extend_from_slice(ir.weights);
    packed
}

impl GraphInner {
    /// Read the graph's arguments and run the initialization stage the driver asks for.
    fn initialize(&mut self) -> Result<(), L0Error> {
        let context = self.context.clone();
        let table = context.graph_table()?;
        let get_properties = table
            .get_properties_2
            .ok_or(L0Error::MissingSymbol("pfnGetProperties2"))?;
        let mut properties = GraphProperties2::new();
        // SAFETY: the graph is live and `properties` is a correctly sized, typed structure.
        check("zeGraphGetProperties2", unsafe {
            get_properties(self.handle, &mut properties)
        })?;
        let get_argument = table
            .get_argument_properties
            .ok_or(L0Error::MissingSymbol("pfnGetArgumentProperties"))?;
        for index in 0..properties.num_graph_args {
            let mut argument = GraphArgumentProperties::new();
            // SAFETY: the graph is live, `index` is below its argument count, and `argument` is a
            // correctly sized, typed structure.
            check("zeGraphGetArgumentProperties", unsafe {
                get_argument(self.handle, index, &mut argument)
            })?;
            self.arguments.push(argument_of(&argument));
        }
        match properties.init_stage_required {
            ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE => {
                let append = table
                    .append_graph_initialize
                    .ok_or(L0Error::MissingSymbol("pfnAppendGraphInitialize"))?;
                let mut recording = context.begin_recording()?;
                let list = recording.list()?;
                // SAFETY: `list` is open and the graph is live; no profiling query, events or waits.
                check("zeCommandListAppendGraphInitialize", unsafe {
                    append(list, self.handle, null_mut(), 0, null_mut())
                })?;
                let recording = recording.finish()?;
                context.execute(&recording, Duration::from_secs(30))?;
            }
            ZE_GRAPH_STAGE_INITIALIZE => {
                let initialize = table
                    .graph_initialize
                    .ok_or(L0Error::MissingSymbol("pfnGraphInitialize"))?;
                // SAFETY: the graph is live.
                check("zeGraphInitialize", unsafe { initialize(self.handle) })?;
            }
            _ => {}
        }
        Ok(())
    }
}

fn argument_of(raw: &GraphArgumentProperties) -> GraphArgument {
    // SAFETY: the driver writes a NUL-terminated name into the fixed array it was handed.
    let name = unsafe { CStr::from_ptr(raw.name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    // The driver pads `dims` to five entries with ones; the leading entries before the real rank are the
    // padding, which this crate does not interpret: callers compare the trailing product and the name.
    GraphArgument {
        name,
        direction: if raw.kind == 0 {
            GraphDirection::Input
        } else {
            GraphDirection::Output
        },
        dims: raw.dims.to_vec(),
        precision: GraphPrecision::from_raw(raw.device_precision),
    }
}

/// A command list that is destroyed on drop.
struct OwnedList {
    context: L0Context,
    list: Handle,
}

impl Drop for OwnedList {
    fn drop(&mut self) {
        // SAFETY: `list` came from a successful `zeCommandListCreate` and is destroyed exactly once, here.
        // A list is dropped only after the queue drained (or the caller gave up and keeps the recording).
        let result = unsafe { (self.context.funcs().command_list_destroy)(self.list) };
        if let Err(error) = check("zeCommandListDestroy", result) {
            tracing::warn!(%error, "leaking a Level Zero command list");
        }
    }
}

/// An open recording: graph executions and copies are appended, then [`L0RecordingBuilder::finish`] closes it.
pub struct L0RecordingBuilder {
    list: Option<OwnedList>,
    graphs: Vec<L0Graph>,
    buffers: Vec<L0HostBuffer>,
}

impl L0RecordingBuilder {
    fn list(&mut self) -> Result<Handle, L0Error> {
        self.list
            .as_ref()
            .map(|list| list.list)
            .ok_or(L0Error::RecordingClosed)
    }

    /// Append one execution of `graph` over `buffers`: one per graph argument, inputs first and then
    /// outputs. The binding is captured now, so the same graph can be appended again over other buffers.
    pub fn execute_graph(
        &mut self,
        graph: &L0Graph,
        buffers: &[&L0HostBuffer],
    ) -> Result<(), L0Error> {
        let context = graph.0.context.clone();
        let table = context.graph_table()?;
        if buffers.len() != graph.0.arguments.len() {
            return Err(L0Error::ArgumentCount {
                expected: u32::try_from(graph.0.arguments.len()).unwrap_or(u32::MAX),
                given: buffers.len(),
            });
        }
        let set = table
            .set_argument_value
            .ok_or(L0Error::MissingSymbol("pfnSetArgumentValue"))?;
        let execute = table
            .append_graph_execute
            .ok_or(L0Error::MissingSymbol("pfnAppendGraphExecute"))?;
        let list = self.list()?;
        if !list_context(self).same(&context) {
            return Err(L0Error::ForeignResource { what: "graph" });
        }
        for (index, buffer) in buffers.iter().enumerate() {
            if !buffer.belongs_to(&context) {
                return Err(L0Error::ForeignResource { what: "buffer" });
            }
            let index = u32::try_from(index).expect("a graph has fewer than 2^32 arguments");
            // SAFETY: the graph is live and the pointer is a host allocation of this context; the driver
            // reads the address, not the memory, here.
            check("zeGraphSetArgumentValue", unsafe {
                set(graph.0.handle, index, buffer.address())
            })?;
        }
        // SAFETY: `list` is open, the graph is live with its arguments bound; no profiling query, events
        // or waits.
        check("zeCommandListAppendGraphExecute", unsafe {
            execute(list, graph.0.handle, null_mut(), null_mut(), 0, null_mut())
        })?;
        self.graphs.push(graph.clone());
        self.buffers.extend(buffers.iter().map(|&b| b.clone()));
        Ok(())
    }

    /// Append a copy of all of `src` into `dst`, ordered with the executions around it.
    pub fn copy(&mut self, dst: &L0HostBuffer, src: &L0HostBuffer) -> Result<(), L0Error> {
        if dst.len() != src.len() {
            return Err(L0Error::TransferLength {
                host: src.len(),
                device: dst.len(),
            });
        }
        let list = self.list()?;
        let context = list_context(self).clone();
        if !dst.belongs_to(&context) || !src.belongs_to(&context) {
            return Err(L0Error::ForeignResource { what: "buffer" });
        }
        if dst.is_empty() {
            return Ok(());
        }
        // SAFETY: `list` is open; both addresses are live host allocations of this context of `len` bytes,
        // kept alive by the recording until it is dropped.
        check("zeCommandListAppendMemoryCopy", unsafe {
            (context.funcs().command_list_append_memory_copy)(
                list,
                dst.address().cast_mut(),
                src.address(),
                dst.len(),
                null_mut(),
                0,
                null_mut(),
            )
        })?;
        self.buffers.push(dst.clone());
        self.buffers.push(src.clone());
        Ok(())
    }

    /// Close the list. The recording keeps the graphs and buffers it names alive.
    pub fn finish(&mut self) -> Result<L0Recording, L0Error> {
        let owned = self.list.take().ok_or(L0Error::RecordingClosed)?;
        // SAFETY: `list` is a live, open command list.
        check("zeCommandListClose", unsafe {
            (owned.context.funcs().command_list_close)(owned.list)
        })?;
        Ok(L0Recording(Rc::new(RecordingInner {
            list: owned,
            _graphs: std::mem::take(&mut self.graphs),
            _buffers: std::mem::take(&mut self.buffers),
        })))
    }
}

fn list_context(builder: &L0RecordingBuilder) -> &L0Context {
    &builder
        .list
        .as_ref()
        .expect("checked open by the caller's `list()`")
        .context
}

struct RecordingInner {
    list: OwnedList,
    _graphs: Vec<L0Graph>,
    _buffers: Vec<L0HostBuffer>,
}

/// A closed command list of graph executions and copies. Cloning shares it; it replays any number of times.
#[derive(Clone)]
pub struct L0Recording(Rc<RecordingInner>);
