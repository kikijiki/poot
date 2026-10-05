//! `rocminfo`-style diagnostic: initializes HSA, enumerates every agent (CPU + GPU) through the raw
//! FFI, and prints the chosen GPU agent's gfx id, queue limits, wavefront size and cooperative-queue
//! capability. With `HSA_OVERRIDE_GFX_VERSION=11.5.1` (set by the flake) the ISA name is `gfx1151` on
//! Strix Halo (spec 063 FR-002).
//!
//! Exits 0 on success; exits 1 with a diagnostic when ROCm is missing or no GPU is visible.

use poot_rocm_runtime::{Funcs, RocmContext, RocmError};
use std::cell::RefCell;
use std::ffi::{CStr, c_void};

fn main() {
    // Surface the tracing lines from RocmContext::new to see where HSA stops (init / iterate / pool
    // walk / queue).
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    println!("== poot-rocm-runtime agent-list (card 106 / M0) ==");

    let ctx = match RocmContext::new() {
        Ok(c) => c,
        Err(RocmError::LibraryNotFound(msg)) => {
            eprintln!("FAIL: libhsa-runtime64.so.1 not loadable: {msg}");
            eprintln!("  hint: install ROCm (libhsa-runtime64.so.1) or run inside `nix develop`");
            std::process::exit(1);
        }
        Err(RocmError::NoGpuAgent) => {
            eprintln!("FAIL: no GPU agent visible to HSA on this box");
            eprintln!("  hint: is /dev/kfd readable? Is the amdgpu driver loaded?");
            std::process::exit(1);
        }
        Err(e) => {
            eprintln!("FAIL: hsa_init: {e}");
            std::process::exit(1);
        }
    };

    let funcs = ctx.funcs();
    let agent = ctx.gpu_agent();

    println!("== chosen GPU agent ==");
    println!("  gfx id:       {}", ctx.isa_name());
    println!("  agent handle: {}", agent.handle);

    print_agent_detail(funcs, agent);

    println!();
    println!("== all HSA agents ==");
    iterate_all_agents(funcs);

    println!();
    println!("OK: chosen gfx id = {}", ctx.isa_name());
}

fn print_agent_detail(funcs: &Funcs, agent: poot_rocm_runtime::bindings::hsa_agent_t) {
    use poot_rocm_runtime::bindings::{
        HSA_AGENT_INFO_NAME, HSA_AGENT_INFO_QUEUE_MAX_SIZE, HSA_AGENT_INFO_VENDOR_NAME,
        HSA_AGENT_INFO_WAVEFRONT_SIZE, HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES,
    };
    let mut name = [0i8; 64];
    let mut vendor = [0i8; 64];
    let mut wavefront: u32 = 0;
    let mut coop_q: u32 = 0;
    let mut max_q: u32 = 0;
    unsafe {
        if (funcs.hsa_agent_get_info)(agent, HSA_AGENT_INFO_NAME, name.as_mut_ptr() as *mut _) == 0
        {
            println!("  name:        {:?}", CStr::from_ptr(name.as_ptr()));
        }
        if (funcs.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_VENDOR_NAME,
            vendor.as_mut_ptr() as *mut _,
        ) == 0
        {
            println!("  vendor:      {:?}", CStr::from_ptr(vendor.as_ptr()));
        }
        if (funcs.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_WAVEFRONT_SIZE,
            &mut wavefront as *mut _ as *mut _,
        ) == 0
        {
            println!("  wavefront:   {wavefront}");
        }
        if (funcs.hsa_agent_get_info)(
            agent,
            HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES,
            &mut coop_q as *mut _ as *mut _,
        ) == 0
        {
            println!("  coop queues: {coop_q}");
        }
        if (funcs.hsa_agent_get_info)(
            agent,
            HSA_AGENT_INFO_QUEUE_MAX_SIZE,
            &mut max_q as *mut _ as *mut _,
        ) == 0
        {
            println!("  queue max:   {max_q}");
        }
    }
}

thread_local! {
    static VISIT_FUNCS: RefCell<*const Funcs> = const { RefCell::new(std::ptr::null()) };
}

struct Counter(usize);

fn iterate_all_agents(funcs: &Funcs) {
    use poot_rocm_runtime::bindings::{
        HSA_AGENT_INFO_DEVICE, HSA_AGENT_INFO_ISA, HSA_AGENT_INFO_NAME, HSA_AGENT_INFO_VENDOR_NAME,
        HSA_DEVICE_TYPE_GPU, HSA_ISA_INFO_NAME,
    };
    VISIT_FUNCS.with(|c| *c.borrow_mut() = funcs as *const _);

    unsafe extern "C" fn visit(
        agent: poot_rocm_runtime::bindings::hsa_agent_t,
        data: *mut c_void,
    ) -> poot_rocm_runtime::bindings::hsa_status_t {
        let counter = unsafe { &mut *(data as *mut Counter) };
        counter.0 += 1;
        let funcs_ptr = VISIT_FUNCS.with(|c| *c.borrow());
        if funcs_ptr.is_null() {
            return 1;
        }
        let funcs = unsafe { &*funcs_ptr };
        let mut name = [0i8; 64];
        let mut vendor = [0i8; 64];
        let mut dev_type: u32 = 0;
        unsafe {
            (funcs.hsa_agent_get_info)(agent, HSA_AGENT_INFO_NAME, name.as_mut_ptr() as *mut _);
            (funcs.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_VENDOR_NAME,
                vendor.as_mut_ptr() as *mut _,
            );
            (funcs.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_DEVICE,
                &mut dev_type as *mut _ as *mut _,
            );
        }
        let kind = if dev_type == HSA_DEVICE_TYPE_GPU {
            "GPU"
        } else {
            "CPU"
        };
        let mut isa = [0i8; 64];
        unsafe {
            let mut isa_handle: u64 = 0;
            (funcs.hsa_agent_get_info)(
                agent,
                HSA_AGENT_INFO_ISA,
                &mut isa_handle as *mut _ as *mut _,
            );
            (funcs.hsa_isa_get_info_alt)(
                poot_rocm_runtime::bindings::hsa_isa_t { handle: isa_handle },
                HSA_ISA_INFO_NAME,
                isa.as_mut_ptr() as *mut _,
            );
        }
        println!(
            "  Agent {:2} - Name: {:?} - Vendor: {:?} - Kind: {} - ISA: {:?}",
            counter.0,
            unsafe { CStr::from_ptr(name.as_ptr()) },
            unsafe { CStr::from_ptr(vendor.as_ptr()) },
            kind,
            unsafe { CStr::from_ptr(isa.as_ptr()) },
        );
        0
    }

    let mut counter = Counter(0);
    unsafe {
        (funcs.hsa_iterate_agents)(Some(visit), &mut counter as *mut _ as *mut _);
    }
}
