//! Agent and pool discovery queries.
//!
//! Every `unsafe fn` here has one contract: `funcs` is the table of a loaded HSA runtime between
//! `hsa_init` and `hsa_shut_down`. The runtime validates the handles it is given, so the remaining
//! obligation, stated at each call, is that each info query's value pointer addresses storage of the
//! size and type the header gives that attribute.

use crate::*;

pub use poot_runtime_common::fnv1a;

// Helpers used by the iteration callbacks and the example.

pub(crate) struct GpuPicker<'a> {
    pub(crate) funcs: &'a Funcs,
    pub(crate) picked: Option<hsa_agent_t>,
}

impl GpuPicker<'_> {
    /// # Safety
    ///
    /// The module contract: `self.funcs` belongs to an initialized runtime.
    pub(crate) unsafe fn visit(&mut self, agent: hsa_agent_t) -> hsa_status_t {
        // SAFETY: `HSA_AGENT_INFO_DEVICE` writes a 4-byte `hsa_device_type_t` into `dev_type`.
        unsafe {
            // HSA_AGENT_INFO_DEVICE (= 17) returns an hsa_device_type_t: 0=CPU, 1=GPU, 2=DSP. Do not use
            // HSA_AGENT_INFO_FEATURE (a feature bitmask) for device type.
            //
            // A non-zero status returned from this callback becomes `hsa_iterate_agents`' own return value
            // (the sample ~/refs/ROCR-Runtime/samples/common/hsa_rsrc_factory.cpp::get_gpu_agent always
            // returns SUCCESS and collects). Follow that: always return SUCCESS and pick the first GPU after
            // iteration completes.
            let mut dev_type: u32 = 0;
            if check((self.funcs.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_DEVICE,
                &mut dev_type as *mut _ as *mut c_void,
            ))
            .is_err()
            {
                return 0; // ignore failures, continue iteration
            }
            if dev_type == HSA_DEVICE_TYPE_GPU && self.picked.is_none() {
                self.picked = Some(agent);
            }
            0 // ALWAYS continue - iterate_agents will return SUCCESS
        }
    }
}

pub(crate) struct PoolState<'a> {
    pub(crate) coarse: &'a mut Option<hsa_amd_memory_pool_t>,
    pub(crate) fine: &'a mut Option<hsa_amd_memory_pool_t>,
    pub(crate) funcs: &'a Funcs,
}

impl PoolState<'_> {
    /// Classify a pool, mirroring `clr/rocclr/device/rocm/rocdevice.cpp:800-908`:
    /// - skip non-GLOBAL segment pools
    /// - skip pools that disallow runtime allocation
    /// - prefer FINE_GRAINED + KERNARG_INIT for the kernarg buffer (the loader rejects coarse-grained
    ///   kernarg); coarse-grained GPU pool for device-local scratch
    ///
    /// # Safety
    ///
    /// The module contract: `self.funcs` belongs to an initialized runtime.
    pub(crate) unsafe fn visit(&mut self, pool: hsa_amd_memory_pool_t) -> hsa_status_t {
        // SAFETY: SEGMENT and LOCATION write 4-byte enums, RUNTIME_ALLOC_ALLOWED a C `bool` (0 or 1),
        // and GLOBAL_FLAGS a `uint32_t`, each into a local of that type.
        unsafe {
            let mut segment: u32 = 0;
            if check((self.funcs.hsa_amd_memory_pool_get_info)(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                &mut segment as *mut _ as *mut c_void,
            ))
            .is_err()
            {
                return 1;
            }
            if segment != HSA_AMD_SEGMENT_GLOBAL {
                return 0;
            }
            let mut location: u32 = 0;
            if check((self.funcs.hsa_amd_memory_pool_get_info)(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_LOCATION,
                &mut location as *mut _ as *mut c_void,
            ))
            .is_err()
            {
                return 1;
            }
            let mut runtime_alloc: bool = false;
            if check((self.funcs.hsa_amd_memory_pool_get_info)(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                &mut runtime_alloc as *mut _ as *mut c_void,
            ))
            .is_err()
            {
                return 1;
            }
            if !runtime_alloc {
                return 0;
            }
            let mut flags: u32 = 0;
            if check((self.funcs.hsa_amd_memory_pool_get_info)(
                pool,
                HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                &mut flags as *mut _ as *mut c_void,
            ))
            .is_err()
            {
                return 1;
            }
            let fine = flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED != 0;
            let coarse = flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED != 0;
            let ext_fine = flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_EXTENDED_SCOPE_FINE_GRAINED != 0;
            let _kernarg = flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT != 0;
            // Pick the GPU pool. On dGPUs this is COARSE_GRAINED (the VRAM bucket). On APUs/iGPUs with unified
            // memory the GPU pool is FINE_GRAINED only (no separate VRAM; allocations come from the system
            // pool via HMM/SVM), so accept either flag; FINE_GRAINED is fine for iGPU zero-copy.
            if location == HSA_AMD_MEMORY_POOL_LOCATION_GPU
                && self.coarse.is_none()
                && (coarse || fine)
            {
                *self.coarse = Some(pool);
            }
            // Pick the kernarg / host-mappable fine pool. On Strix Halo the GPU agent's pool is the one
            // documented for kernarg allocation. Accept FINE_GRAINED + KERNARG_INIT, FINE_GRAINED alone as a
            // fallback, or EXTENDED_SCOPE_FINE_GRAINED. (A `kernarg ||` term would be redundant here since
            // `fine || ext_fine` is already required by the outer condition.)
            if self.fine.is_none() && (fine || ext_fine) {
                *self.fine = Some(pool);
            }
            0
        }
    }
}

pub(crate) unsafe fn agent_name(funcs: &Funcs, agent: hsa_agent_t) -> Result<String, RocmError> {
    // SAFETY: `HSA_AGENT_INFO_NAME` writes a NUL-terminated `char[64]` into the 64-byte `name`.
    unsafe {
        let mut name: [c_char; 64] = [0; 64];
        check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_NAME,
            name.as_mut_ptr() as *mut c_void,
        ))?;
        Ok(cstr_to_string(&name))
    }
}

pub(crate) unsafe fn agent_vendor(funcs: &Funcs, agent: hsa_agent_t) -> Result<String, RocmError> {
    // SAFETY: `HSA_AGENT_INFO_VENDOR_NAME` writes a NUL-terminated `char[64]` into the 64-byte `name`.
    unsafe {
        let mut name: [c_char; 64] = [0; 64];
        check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_VENDOR_NAME,
            name.as_mut_ptr() as *mut c_void,
        ))?;
        Ok(cstr_to_string(&name))
    }
}

pub(crate) unsafe fn amd_agent_product_name(funcs: &Funcs, agent: hsa_agent_t) -> Option<String> {
    // SAFETY: `HSA_AMD_AGENT_INFO_PRODUCT_NAME` writes a 64-char string into the 64-byte `name`.
    unsafe {
        let mut name: [c_char; 64] = [0; 64];
        let status = check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AMD_AGENT_INFO_PRODUCT_NAME,
            name.as_mut_ptr() as *mut c_void,
        ));
        match status {
            Ok(()) => Some(cstr_to_string(&name)),
            Err(_) => None,
        }
    }
}

/// `HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES` returns `uint8_t[8]`. APU is bit index
/// [`HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU`] via the `hsa_flag_isset64` convention of rocminfo (the
/// enum value is an index, not a byte mask).
pub(crate) unsafe fn amd_agent_is_apu(
    funcs: &Funcs,
    agent: hsa_agent_t,
) -> Result<bool, RocmError> {
    // SAFETY: `HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES` writes a `uint8_t[8]` into `props`.
    unsafe {
        let mut props: [u8; 8] = [0; 8];
        check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AMD_AGENT_INFO_MEMORY_PROPERTIES,
            props.as_mut_ptr() as *mut c_void,
        ))?;
        Ok(hsa_flag_isset64(
            &props,
            HSA_AMD_MEMORY_PROPERTY_AGENT_IS_APU,
        ))
    }
}

/// Mirror of `hsa_ext_amd.h`'s `hsa_flag_isset64`: `bit` is a bit index into the 64-bit property
/// vector stored as `uint8_t[8]`.
pub(crate) fn hsa_flag_isset64(value: &[u8; 8], bit: u32) -> bool {
    let index = (bit / 8) as usize;
    let sub_bit = bit % 8;
    value
        .get(index)
        .is_some_and(|byte| byte & (1 << sub_bit) != 0)
}

/// # Safety
///
/// The module contract, and `attribute` must be an agent attribute whose value is a `uint32_t`.
pub(crate) unsafe fn amd_agent_u32_info(
    funcs: &Funcs,
    agent: hsa_agent_t,
    attribute: u32,
) -> Option<u32> {
    // SAFETY: `attribute` writes a `uint32_t` (caller contract) into `val`.
    unsafe {
        let mut val: u32 = 0;
        match check((funcs.hsa_agent_get_info)(
            agent,
            attribute,
            &mut val as *mut _ as *mut c_void,
        )) {
            Ok(()) => Some(val),
            Err(_) => None,
        }
    }
}

pub(crate) unsafe fn first_cpu_agent(funcs: &Funcs) -> Result<Option<hsa_agent_t>, RocmError> {
    struct CpuPicker<'a> {
        funcs: &'a Funcs,
        picked: Option<hsa_agent_t>,
    }
    unsafe extern "C" fn visit(agent: hsa_agent_t, data: *mut c_void) -> hsa_status_t {
        // SAFETY: `data` is the `&mut picker` passed to `hsa_iterate_agents` below, which calls back
        // synchronously on this thread while `picker` is otherwise unused; `HSA_AGENT_INFO_DEVICE`
        // writes a 4-byte enum into `dev_type`.
        unsafe {
            let picker = &mut *(data as *mut CpuPicker<'_>);
            let mut dev_type: u32 = 0;
            if check((picker.funcs.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_DEVICE,
                &mut dev_type as *mut _ as *mut c_void,
            ))
            .is_err()
            {
                return 0;
            }
            if dev_type == HSA_DEVICE_TYPE_CPU && picker.picked.is_none() {
                picker.picked = Some(agent);
            }
            0
        }
    }
    let mut picker = CpuPicker {
        funcs,
        picked: None,
    };
    // SAFETY: `visit` matches the callback type and reads `data` only as the `CpuPicker` passed here,
    // which outlives the synchronous iteration.
    unsafe {
        check((funcs.hsa_iterate_agents)(
            Some(visit),
            &mut picker as *mut _ as *mut c_void,
        ))?;
    }
    Ok(picker.picked)
}

/// CPU-agent path to a GPU memory pool: hop count plus each hop's `link_type`.
pub(crate) unsafe fn cpu_to_pool_link_types(
    funcs: &Funcs,
    cpu_agent: hsa_agent_t,
    pool: hsa_amd_memory_pool_t,
) -> Result<(u32, Vec<u32>), RocmError> {
    // SAFETY: NUM_LINK_HOPS writes a `uint32_t` into `num_hops`; LINK_INFO writes `num_hops`
    // `hsa_amd_memory_pool_link_info_t` records (layout mirrored in `bindings`) into `hops`, which
    // holds exactly that many.
    unsafe {
        let mut num_hops: u32 = 0;
        check((funcs.hsa_amd_agent_memory_pool_get_info)(
            cpu_agent,
            pool,
            HSA_AMD_AGENT_MEMORY_POOL_INFO_NUM_LINK_HOPS,
            &mut num_hops as *mut _ as *mut c_void,
        ))?;
        if num_hops == 0 {
            return Ok((0, Vec::new()));
        }
        let mut hops = vec![
            hsa_amd_memory_pool_link_info_t {
                min_latency: 0,
                max_latency: 0,
                min_bandwidth: 0,
                max_bandwidth: 0,
                atomic_support_32bit: false,
                atomic_support_64bit: false,
                coherent_support: false,
                link_type: 0,
                numa_distance: 0,
            };
            num_hops as usize
        ];
        check((funcs.hsa_amd_agent_memory_pool_get_info)(
            cpu_agent,
            pool,
            HSA_AMD_AGENT_MEMORY_POOL_INFO_LINK_INFO,
            hops.as_mut_ptr() as *mut c_void,
        ))?;
        let link_types = hops.iter().map(|hop| hop.link_type).collect();
        Ok((num_hops, link_types))
    }
}

pub(crate) unsafe fn amd_cooperative_queues(funcs: &Funcs, agent: hsa_agent_t) -> Option<u32> {
    let mut val: u32 = 0;
    // SAFETY: `HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES` writes a `uint32_t` into `val`.
    let status = unsafe {
        check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES,
            &mut val as *mut _ as *mut c_void,
        ))
    };
    match status {
        Ok(()) => Some(val),
        Err(_) => None,
    }
}

/// `HSA_AMD_AGENT_INFO_MEMORY_AVAIL` (0xA015): "the amount of memory available in bytes across all
/// global pools owned by the agent" (`hsa_ext_amd.h`), a live driver query. The "free" half of the
/// card 030 VRAM metric; see [`RocmContext::vram_used_bytes`].
pub(crate) unsafe fn amd_agent_memory_avail(
    funcs: &Funcs,
    agent: hsa_agent_t,
) -> Result<u64, RocmError> {
    // SAFETY: `HSA_AMD_AGENT_INFO_MEMORY_AVAIL` writes a `uint64_t` into `val`.
    unsafe {
        let mut val: u64 = 0;
        check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AMD_AGENT_INFO_MEMORY_AVAIL,
            &mut val as *mut _ as *mut c_void,
        ))?;
        Ok(val)
    }
}

/// `HSA_AMD_MEMORY_POOL_INFO_SIZE` (2): "Size of this pool, in bytes" (`hsa_ext_amd.h`; `size_t`,
/// read as `u64`). The "total" half of the card 030 VRAM metric, read once at construction from
/// `gpu_coarse_pool`; see [`RocmContext::new`] for why it is a single pool's size, not a sum.
pub(crate) unsafe fn amd_pool_size(
    funcs: &Funcs,
    pool: hsa_amd_memory_pool_t,
) -> Result<u64, RocmError> {
    // SAFETY: `HSA_AMD_MEMORY_POOL_INFO_SIZE` writes a `size_t`, 8 bytes on the 64-bit targets this
    // crate builds for, into `val`.
    unsafe {
        let mut val: u64 = 0;
        check((funcs.hsa_amd_memory_pool_get_info)(
            pool,
            HSA_AMD_MEMORY_POOL_INFO_SIZE,
            &mut val as *mut _ as *mut c_void,
        ))?;
        Ok(val)
    }
}

pub(crate) unsafe fn agent_wavefront_size(
    funcs: &Funcs,
    agent: hsa_agent_t,
) -> Result<u32, RocmError> {
    // SAFETY: `HSA_AGENT_INFO_WAVEFRONT_SIZE` writes a `uint32_t` into `val`.
    unsafe {
        let mut val: u32 = 0;
        check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_WAVEFRONT_SIZE,
            &mut val as *mut _ as *mut c_void,
        ))?;
        Ok(val)
    }
}

pub(crate) unsafe fn agent_isa_name(
    funcs: &Funcs,
    agent: hsa_agent_t,
) -> Result<String, RocmError> {
    // SAFETY: `HSA_AGENT_INFO_ISA` writes an 8-byte `hsa_isa_t` into `isa_handle`, NAME_LENGTH a
    // `uint32_t` into `name_length`, and NAME `name_length` bytes into `name`, which holds one more.
    unsafe {
        // Query the ISA handle from the agent, then the ISA name from the handle. HSA_AGENT_INFO_ISA
        // returns an hsa_isa_t (same {u64 handle} layout as hsa_agent_t, but a distinct handle class:
        // the wrong one returns HSA_STATUS_ERROR_INVALID_INDEX).
        //
        // Use `hsa_isa_get_info_alt`, not the deprecated `hsa_isa_get_info`, which returns INVALID_INDEX
        // for `HSA_ISA_INFO_NAME` in ROCm 7.2.x (the alt call returns name="amdgcn-amd-amdhsa--gfx1151").
        let mut isa_handle: u64 = 0;
        check((funcs.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_ISA,
            &mut isa_handle as *mut _ as *mut c_void,
        ))?;
        let isa: hsa_isa_t = hsa_isa_t { handle: isa_handle };
        let mut name_length: u32 = 0;
        check((funcs.hsa_isa_get_info_alt)(
            isa,
            HSA_ISA_INFO_NAME_LENGTH,
            &mut name_length as *mut _ as *mut c_void,
        ))?;
        // The name is `name_length` bytes; the extra zero byte terminates it.
        let mut name: Vec<c_char> = vec![0; name_length as usize + 1];
        check((funcs.hsa_isa_get_info_alt)(
            isa,
            HSA_ISA_INFO_NAME,
            name.as_mut_ptr() as *mut c_void,
        ))?;
        Ok(cstr_to_string(&name))
    }
}

pub(crate) fn cstr_to_string(buf: &[c_char]) -> String {
    let nul = buf.iter().position(|&b| b == 0).unwrap_or(buf.len());
    let bytes: Vec<u8> = buf[..nul].iter().map(|&b| b as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}
