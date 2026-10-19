use std::ffi::{CString, c_void};
use std::ptr::{NonNull, null_mut};
use std::rc::Rc;

use crate::buffer::L0HostBuffer;
use crate::error::{L0Error, check};
use crate::ffi::{
    CommandQueueDesc, ContextDesc, DeviceProperties, Funcs, GraphTable, Handle, HostMemAllocDesc,
    Loader, QueueGroupProperties, ZE_INIT_FLAG_VPU_ONLY, ZE_QUEUE_GROUP_FLAG_COMPUTE,
};

/// Cache-line alignment for allocations.
pub(crate) const ALLOC_ALIGNMENT: usize = 64;

/// What a context owns. Dropping the last [`L0Context`] handle destroys the queue and context; the loader
/// library, the last field, closes after that.
struct Inner {
    device: Handle,
    context: Handle,
    queue: Handle,
    /// The command queue group (ordinal) the queue was created on; command lists must use the same one.
    ordinal: u32,
    /// The PCI vendor id the device reports (`0x8086` for Intel).
    vendor_id: u32,
    /// The driver's graph extension table. The driver owns it for as long as the loader stays loaded, which is
    /// as long as `loader` below.
    graph: NonNull<GraphTable>,
    loader: Loader,
}

/// A Level Zero NPU (VPU) device with a context and a compute queue. A handle is cheap to clone: every buffer,
/// graph and recording keeps one, so the queue and context outlive them.
#[derive(Clone)]
pub struct L0Context(Rc<Inner>);

impl L0Context {
    /// `dlopen` the loader, initialize Level Zero for the VPU drivers and open the first device of the first
    /// driver that has one, with the `ZE_extension_graph` extension.
    pub fn open() -> Result<Self, L0Error> {
        let loader = Loader::load()?;
        let funcs = &loader.funcs;
        // SAFETY: plain Level Zero initialization; the flag selects the VPU drivers.
        check("zeInit", unsafe { (funcs.init)(ZE_INIT_FLAG_VPU_ONLY) })?;

        let (driver, device) = first_device(funcs)?;
        let graph = graph_table(funcs, driver)?;
        let mut properties = DeviceProperties::new();
        // SAFETY: `device` is a live handle and `properties` is a correctly sized, typed structure.
        check("zeDeviceGetProperties", unsafe {
            (funcs.device_get_properties)(device, &mut properties)
        })?;
        tracing::info!(
            vendor_id = properties.vendor_id,
            device_id = properties.device_id,
            "Level Zero device"
        );
        let ordinal = compute_queue_group(funcs, device)?;

        let context_desc = ContextDesc::new();
        let mut context = null_mut();
        // SAFETY: `driver` is a live handle and `context_desc` outlives the call.
        check("zeContextCreate", unsafe {
            (funcs.context_create)(driver, &context_desc, &mut context)
        })?;
        let queue_desc = CommandQueueDesc::asynchronous(ordinal);
        let mut queue = null_mut();
        // SAFETY: `context` and `device` are live and `queue_desc` outlives the call.
        let created = check("zeCommandQueueCreate", unsafe {
            (funcs.command_queue_create)(context, device, &queue_desc, &mut queue)
        });
        if let Err(error) = created {
            // SAFETY: `context` was created above and nothing else refers to it.
            unsafe { (funcs.context_destroy)(context) };
            return Err(error);
        }
        Ok(L0Context(Rc::new(Inner {
            device,
            context,
            queue,
            ordinal,
            vendor_id: properties.vendor_id,
            graph,
            loader,
        })))
    }

    /// Whether `other` is a handle to this same context.
    pub(crate) fn same(&self, other: &L0Context) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }

    /// The graph extension's entry points.
    pub(crate) fn graph_table(&self) -> &GraphTable {
        // SAFETY: the table pointer came from the driver of this context's loader, which is loaded until the
        // last handle drops; the table is not written after the driver builds it.
        unsafe { self.0.graph.as_ref() }
    }

    /// Allocate `len` bytes of host memory the device can address (`zeMemAllocHost`), zero-initialized by the
    /// driver's page allocator.
    pub fn alloc_host(&self, len: usize) -> Result<L0HostBuffer, L0Error> {
        let desc = HostMemAllocDesc::new();
        let mut ptr: *mut c_void = null_mut();
        // SAFETY: the context is live and `desc` outlives the call.
        let result = unsafe {
            (self.funcs().mem_alloc_host)(self.0.context, &desc, len, ALLOC_ALIGNMENT, &mut ptr)
        };
        check("zeMemAllocHost", result)?;
        Ok(L0HostBuffer::new(self.clone(), ptr, len))
    }

    /// The PCI vendor id the device reports: `0x8086` for an Intel device.
    pub fn vendor_id(&self) -> u32 {
        self.0.vendor_id
    }

    pub(crate) fn funcs(&self) -> &Funcs {
        &self.0.loader.funcs
    }

    pub(crate) fn handle(&self) -> Handle {
        self.0.context
    }

    pub(crate) fn device(&self) -> Handle {
        self.0.device
    }

    pub(crate) fn queue(&self) -> Handle {
        self.0.queue
    }

    pub(crate) fn ordinal(&self) -> u32 {
        self.0.ordinal
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        // SAFETY: the queue and context were created in `new` and are destroyed exactly once, here, after every
        // buffer, graph and recording that holds the context is gone; the loader closes after this returns.
        let queue = unsafe { (self.loader.funcs.command_queue_destroy)(self.queue) };
        if let Err(error) = check("zeCommandQueueDestroy", queue) {
            tracing::warn!(%error, "leaking a Level Zero command queue");
        }
        // SAFETY: as above; the queue is gone, so nothing in the context is in flight.
        let context = unsafe { (self.loader.funcs.context_destroy)(self.context) };
        if let Err(error) = check("zeContextDestroy", context) {
            tracing::warn!(%error, "leaking a Level Zero context");
        }
    }
}

/// The first device of the first driver that has one.
fn first_device(funcs: &Funcs) -> Result<(Handle, Handle), L0Error> {
    let mut n_drivers = 0u32;
    // SAFETY: a null handle array asks for the count only.
    check("zeDriverGet", unsafe {
        (funcs.driver_get)(&mut n_drivers, null_mut())
    })?;
    let mut drivers = vec![null_mut(); n_drivers as usize];
    // SAFETY: `drivers` has room for `n_drivers` handles.
    check("zeDriverGet", unsafe {
        (funcs.driver_get)(&mut n_drivers, drivers.as_mut_ptr())
    })?;
    for &driver in &drivers[..n_drivers as usize] {
        let mut n_devices = 0u32;
        // SAFETY: `driver` is a live handle; a null array asks for the count only.
        check("zeDeviceGet", unsafe {
            (funcs.device_get)(driver, &mut n_devices, null_mut())
        })?;
        if n_devices == 0 {
            continue;
        }
        let mut devices = vec![null_mut(); n_devices as usize];
        // SAFETY: `devices` has room for `n_devices` handles.
        check("zeDeviceGet", unsafe {
            (funcs.device_get)(driver, &mut n_devices, devices.as_mut_ptr())
        })?;
        return Ok((driver, devices[0]));
    }
    Err(L0Error::NoDevice)
}

/// The ordinal of the first command queue group that has compute queues.
fn compute_queue_group(funcs: &Funcs, device: Handle) -> Result<u32, L0Error> {
    let mut count = 0u32;
    // SAFETY: `device` is a live handle; a null array asks for the count only.
    check("zeDeviceGetCommandQueueGroupProperties", unsafe {
        (funcs.device_get_queue_group_properties)(device, &mut count, null_mut())
    })?;
    let mut groups: Vec<_> = (0..count).map(|_| QueueGroupProperties::new()).collect();
    // SAFETY: `groups` holds `count` typed structures.
    check("zeDeviceGetCommandQueueGroupProperties", unsafe {
        (funcs.device_get_queue_group_properties)(device, &mut count, groups.as_mut_ptr())
    })?;
    groups
        .iter()
        .position(|group| group.flags & ZE_QUEUE_GROUP_FLAG_COMPUTE != 0 && group.num_queues > 0)
        .map(|ordinal| ordinal as u32)
        .ok_or(L0Error::NoComputeQueue)
}

/// The driver's `ZE_extension_graph` table.
fn graph_table(funcs: &Funcs, driver: Handle) -> Result<NonNull<GraphTable>, L0Error> {
    let name = CString::new("ZE_extension_graph").expect("a literal has no interior NUL");
    let mut table: *mut c_void = null_mut();
    // SAFETY: `driver` is a live handle, `name` is NUL-terminated and outlives the call, and `table` receives
    // the driver's table pointer.
    check("zeDriverGetExtensionFunctionAddress", unsafe {
        (funcs.driver_get_extension_function_address)(driver, name.as_ptr(), &mut table)
    })?;
    NonNull::new(table.cast()).ok_or(L0Error::MissingExtension("ZE_extension_graph"))
}
