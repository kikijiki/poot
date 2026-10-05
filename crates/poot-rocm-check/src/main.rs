//! ROCm/HSA execution check (card 106 M1, 106b): load poot-emitted HSACO via poot-rocm-runtime
//! and verify against the CPU oracle. Analog of `poot-ptx-check` for the AMD path.
//!
//! The `--add` invocation:
//!  1. Compiles `fixtures::add_kernel` to HSACO via `poot-codegen::Target::AmdGcn`.
//!  2. Opens a `RocmContext` (hsa_init + agent pick + queue + memory pools) on the AMD iGPU.
//!  3. Uploads a/b f32 arrays (HMM/SVM: the same pointer is host + device visible).
//!  4. Allocates c + the kernarg buffer.
//!  5. Writes the kernarg layout: 3 pointers + 3 i64 lengths (48 bytes).
//!  6. Loads the HSACO via `hsa_executable_load_agent_code_object`, looks up the `add` symbol.
//!  7. Submits one AQL packet via `hsa_queue_add_write_index_relaxed` + doorbell ring.
//!  8. Waits on the per-launch completion signal.
//!  9. Downloads c, compares element-wise to the CPU oracle `a[i] + b[i]`.
//! 10. Prints `OK: c[0]=4.0 c[255]=259.0 (oracle=cpu)` and exits 0.
//!
//! The `--agent-list` mode lists every HSA agent (CPU + GPU + AIE), like `rocminfo`.
//!
//! The `--pcie-publication-receipt` mode (Card 333) opens the device-ring receipt constructor, asserts a
//! `PcieDeviceMemory` contract on a discrete PCIe GPU, and replays a multi-packet add graph with changing
//! inputs via `replay_graph_batched`. ROCr reads `HSA_ALLOCATE_QUEUE_DEV_MEM=1` once, when the runtime
//! loads, and the receipt constructor refuses to run without it, so this binary sets it as the first
//! statement of `main` in that mode, while the process is still single-threaded. On an APU host the
//! mode prints a SKIP and exits 0 unless ROCm is required.
//!
//! Usage: `poot-rocm-check [--add | --agent-list | --pcie-publication-receipt]`

use std::path::PathBuf;
use std::process::Command;

use poot_codegen::{CompileError, Target, artifact_path, compile, kernel_handle};
use poot_kernel_ir::fixtures;
use poot_rocm_runtime::{
    BufferRole, DeviceStoreFence, QueuePublicationContract, RocmContext, RocmError, fnv1a,
    take_issued_device_store_fence_count,
};
use poot_runtime_common::DeviceBackend;
use poot_target::{AmdArch, ElementKind};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = if args.iter().any(|a| a == "--agent-list") {
        Mode::AgentList
    } else if args.iter().any(|a| a == "--pcie-publication-receipt") {
        Mode::PciePublicationReceipt
    } else {
        Mode::Add
    };
    if matches!(mode, Mode::PciePublicationReceipt) {
        // SAFETY: nothing has spawned a thread yet (no HSA load, no logging, no workers), so no other
        // thread can read or write the environment concurrently.
        unsafe {
            std::env::set_var(DEVICE_RING_ENV, "1");
        }
    }
    let res = match mode {
        Mode::Add => run_add(),
        Mode::AgentList => run_agent_list(),
        Mode::PciePublicationReceipt => run_pcie_publication_receipt(),
    };
    if let Err(e) = res {
        eprintln!("{e}");
        std::process::exit(1);
    }
}

/// ROCr's switch for a device-memory AQL packet ring, read once when the HSA runtime loads.
const DEVICE_RING_ENV: &str = "HSA_ALLOCATE_QUEUE_DEV_MEM";

enum Mode {
    Add,
    AgentList,
    PciePublicationReceipt,
}

fn artifact_dir() -> PathBuf {
    let d = std::env::temp_dir().join("poot-rocm-check");
    let _ = std::fs::create_dir_all(&d);
    d
}

/// Whether the PCIe publication receipt may report a SKIP for `error` instead of failing: only a
/// host without a discrete GPU, and only when the ROCm backend is not required.
fn pcie_receipt_may_skip(error: &RocmError, get: impl Fn(&str) -> Option<String>) -> bool {
    matches!(error, RocmError::ApuNotDiscrete) && !DeviceBackend::Rocm.required(get)
}

/// Compile `fixtures::add_kernel` to HSACO, dispatch it on the AMD iGPU, verify == CPU oracle.
/// SC-001 of spec 063.
fn run_add() -> Result<(), CheckError> {
    let body = fixtures::add_kernel();
    let dir = artifact_dir();
    // Open the context first so codegen targets the device ISA (spec 120), not a constant.
    let ctx = RocmContext::new()?;
    eprintln!(
        "RocmContext ready (gfx={}, upload_strategy={:?})",
        ctx.isa_name(),
        ctx.upload_strategy()
    );
    let arch = AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront()).unwrap_or_else(|_| {
        eprintln!(
            "could not classify ISA {:?}; defaulting to gfx1151",
            ctx.isa_name()
        );
        AmdArch::gfx1151()
    });
    let target = Target::AmdGcn(arch);
    let out = artifact_path(&dir, "add", target);
    compile(&body, target, &out)?;
    let hsaco_bytes = std::fs::read(&out)?;
    eprintln!(
        "compiled `add` to {} ({} bytes)",
        out.display(),
        hsaco_bytes.len()
    );
    let kernel = kernel_handle(&body, target, hsaco_bytes);

    // SC-001 inputs: a = 1..=64, b = 3*1..=64, c = a + b (64 elements, one workgroup).
    //
    // Strix Halo's iGPU only schedules 1 workgroup per AQL packet on the current gfx1151 / ROCm 7.2.3 combo,
    // capped at 64 work-items (2 wavefronts of 32); 106d's cooperative queue is the multi-workgroup path. This
    // exercises one workgroup per dispatch and checks the IR's global-id computation
    // (`workgroup_id * workgroup_size + workitem_id`). Reports
    // `OK: c[0]=4.0 c[63]=67.0 (oracle=cpu, 1 workgroup of 64)`.
    let n: usize = 64;
    let a: Vec<f32> = (1..=n).map(|i| i as f32).collect();
    let b: Vec<f32> = vec![3.0; n];
    let mut c: Vec<f32> = vec![0.0; n];
    let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();

    // Load the HSACO + look up the `add` kernel (ctx opened above).
    let module = ctx.load_hsaco(&kernel)?;
    let kernel = ctx.lookup_kernel(&module, kernel.entry_point())?;
    eprintln!(
        "kernel `add` loaded: kernarg_size={} group_segment={} private_segment={}",
        kernel.kernarg_size(),
        kernel.group_segment_size(),
        kernel.private_segment_size()
    );

    // Upload a + b, allocate c (HMM/SVM: same pointer is host + device).
    let a_buf = ctx.upload_f32(&a, BufferRole::Input)?;
    let b_buf = ctx.upload_f32(&b, BufferRole::Input)?;
    let c_buf = ctx.allocate_f32(n, BufferRole::Output)?;
    let kernarg_buf = ctx.allocate_kernarg(kernel.kernarg_size() as usize)?;

    // Write kernarg layout: ptr a, i64 n_a, ptr b, i64 n_b, ptr c, i64 n_c.
    // 48 bytes total = 3 * (8 byte ptr + 8 byte i64). On aarch64/x86_64 the layouts match.
    #[repr(C)]
    #[derive(Default)]
    struct AddKernarg {
        p_a: *const f32,
        n_a: i64,
        p_b: *const f32,
        n_b: i64,
        p_c: *mut f32,
        n_c: i64,
    }
    let args = AddKernarg {
        p_a: a_buf.device_ptr() as *const f32,
        n_a: n as i64,
        p_b: b_buf.device_ptr() as *const f32,
        n_b: n as i64,
        p_c: c_buf.device_ptr() as *mut f32,
        n_c: n as i64,
    };
    let arg_bytes = unsafe {
        std::slice::from_raw_parts(
            &args as *const AddKernarg as *const u8,
            std::mem::size_of::<AddKernarg>(),
        )
    };
    ctx.write_raw_bytes(&kernarg_buf, 0, arg_bytes)?;

    // Dispatch: 64 work-items in one workgroup (see the SC-001 note above).
    let elements = [a_buf.element(), b_buf.element(), c_buf.element()];
    ctx.dispatch(
        &kernel,
        &kernarg_buf,
        &elements,
        (n as u32, 1, 1),
        (n as u32, 1, 1),
    )?;
    ctx.synchronize()?;

    // Download + oracle compare.
    ctx.download_f32(&c_buf, &mut c)?;
    let mut failures = 0u32;
    for (i, (g, w)) in c.iter().zip(&want).enumerate() {
        if (g - w).abs() > 1e-5 {
            eprintln!("  c[{i}] = {g}, want {w}");
            failures += 1;
        }
    }
    if failures == 0 {
        println!(
            "OK: c[0]={} c[{}]={} (oracle=cpu, 1 workgroup of {})",
            c[0],
            n - 1,
            c[n - 1],
            n
        );
        Ok(())
    } else {
        Err(CheckError::Rocm(RocmError::Hsa(format!(
            "{failures} element(s) disagreed with CPU oracle"
        ))))
    }
}

/// Print every HSA agent visible to the runtime (like `rocminfo`), to check the iGPU is present before
/// `--add`.
fn run_agent_list() -> Result<(), CheckError> {
    let ctx = RocmContext::new()?;
    println!(
        "chosen GPU: {} (handle={:#x}, upload_strategy={:?})",
        ctx.isa_name(),
        ctx.gpu_agent().handle,
        ctx.upload_strategy()
    );
    Ok(())
}

/// Card 333 fail-loud discrete PCIe device-memory-ring publication receipt.
fn run_pcie_publication_receipt() -> Result<(), CheckError> {
    let command = std::env::args().collect::<Vec<_>>().join(" ");
    let commit = git_commit();
    let host_arch = std::env::consts::ARCH;
    let host_kernel = read_text("/proc/version").unwrap_or_else(|| "unknown".into());

    let ctx = match RocmContext::new_for_pcie_device_ring_publication_receipt() {
        Ok(ctx) => ctx,
        Err(error) if pcie_receipt_may_skip(&error, |name| std::env::var(name).ok()) => {
            println!(
                "SKIP: --pcie-publication-receipt requires a discrete PCIe GPU \
                 (local agent reports APU / unified memory)"
            );
            return Ok(());
        }
        Err(e) => return Err(CheckError::Rocm(e)),
    };

    let contract = ctx.queue_publication_contract();
    let provenance = ctx
        .device_ring_receipt_provenance()
        .cloned()
        .ok_or_else(|| {
            CheckError::Rocm(RocmError::Hsa(
                "device-ring receipt constructor did not retain provenance".into(),
            ))
        })?;

    match contract {
        QueuePublicationContract::PcieDeviceMemory {
            fence: DeviceStoreFence::X86StoreFence,
        } => {}
        other => {
            return Err(CheckError::Rocm(RocmError::Hsa(format!(
                "expected PcieDeviceMemory {{ X86StoreFence }}, got {other:?}"
            ))));
        }
    }
    if !provenance.allocate_queue_dev_mem {
        return Err(CheckError::Rocm(RocmError::Hsa(
            "HSA_ALLOCATE_QUEUE_DEV_MEM=1 was not observed at queue creation".into(),
        )));
    }
    if !provenance
        .link_types
        .contains(&poot_rocm_runtime::bindings::HSA_AMD_LINK_INFO_TYPE_PCIE)
    {
        return Err(CheckError::Rocm(RocmError::Hsa(format!(
            "CPU-to-GPU link types {:?} do not include PCIe",
            provenance.link_types
        ))));
    }

    let body = fixtures::add_kernel();
    let dir = artifact_dir();
    let arch = AmdArch::from_isa_name(ctx.isa_name(), ctx.wavefront())
        .unwrap_or_else(|_| AmdArch::new("gfx942", ctx.wavefront().max(64)));
    let target = Target::AmdGcn(arch);
    let out = artifact_path(&dir, "add-pcie-receipt", target);
    compile(&body, target, &out)?;
    let hsaco_bytes = std::fs::read(&out)?;
    let compiled_kernel = kernel_handle(&body, target, hsaco_bytes);
    let module = ctx.load_hsaco(&compiled_kernel)?;
    let add_kernel = ctx.lookup_kernel(&module, compiled_kernel.entry_point())?;

    // Multi-packet batched replay with changing inputs. Prefer more than one chunk when the queue is small
    // enough that a modest packet count crosses half-queue.
    let n: usize = 64;
    let queue_size = provenance.queue_size.max(1) as usize;
    let chunk_size = (queue_size / 2).max(1);
    let packet_count = (chunk_size + 3).clamp(4, 16);
    let rounds = 3usize;

    let mut digest_bytes = Vec::new();
    let mut total_packets = 0usize;
    let mut total_chunks = 0usize;
    let mut total_doorbells = 0usize;
    let _ = take_issued_device_store_fence_count();

    #[repr(C)]
    struct AddKernarg {
        p_a: *const f32,
        n_a: i64,
        p_b: *const f32,
        n_b: i64,
        p_c: *mut f32,
        n_c: i64,
    }

    for round in 0..rounds {
        let mut dispatches = Vec::with_capacity(packet_count);
        let mut c_bufs = Vec::with_capacity(packet_count);
        let mut wants = Vec::with_capacity(packet_count);
        // Keep owned input buffers alive for the batch lifetime.
        let mut owned_a = Vec::with_capacity(packet_count);
        let mut owned_b = Vec::with_capacity(packet_count);
        let mut owned_k = Vec::with_capacity(packet_count);

        for pkt in 0..packet_count {
            let bias = (round * 100 + pkt) as f32;
            let a: Vec<f32> = (1..=n).map(|i| i as f32 + bias).collect();
            let b: Vec<f32> = vec![3.0 + bias * 0.01; n];
            let want: Vec<f32> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
            let a_buf = ctx.upload_f32(&a, BufferRole::Input)?;
            let b_buf = ctx.upload_f32(&b, BufferRole::Input)?;
            let c_buf = ctx.allocate_f32(n, BufferRole::Output)?;
            let kernarg_buf = ctx.allocate_kernarg(add_kernel.kernarg_size() as usize)?;
            let args = AddKernarg {
                p_a: a_buf.device_ptr() as *const f32,
                n_a: n as i64,
                p_b: b_buf.device_ptr() as *const f32,
                n_b: n as i64,
                p_c: c_buf.device_ptr() as *mut f32,
                n_c: n as i64,
            };
            let arg_bytes = unsafe {
                std::slice::from_raw_parts(
                    &args as *const AddKernarg as *const u8,
                    std::mem::size_of::<AddKernarg>(),
                )
            };
            ctx.write_raw_bytes(&kernarg_buf, 0, arg_bytes)?;
            owned_a.push(a_buf);
            owned_b.push(b_buf);
            owned_k.push(kernarg_buf);
            c_bufs.push(c_buf);
            wants.push(want);
        }
        let _keep_inputs = (owned_a, owned_b);
        for kernarg in owned_k {
            dispatches.push((
                add_kernel.clone(),
                kernarg,
                (n as u32, 1, 1),
                (n as u32, 1, 1),
            ));
        }

        let chunks_this = packet_count.div_ceil(chunk_size);
        // All-F32 add kernarg (see `AddKernarg`/the SC-001 dispatch above), one entry per packet.
        let elements = [ElementKind::F32, ElementKind::F32, ElementKind::F32];
        let elements_per_packet: Vec<&[ElementKind]> =
            std::iter::repeat_n(elements.as_slice(), dispatches.len()).collect();
        ctx.replay_graph_batched(&dispatches, &elements_per_packet)?;

        for (c_buf, want) in c_bufs.iter().zip(&wants) {
            let mut got = vec![0.0f32; n];
            ctx.download_f32(c_buf, &mut got)?;
            for (g, w) in got.iter().zip(want) {
                if (g - w).abs() > 1e-5 {
                    return Err(CheckError::Rocm(RocmError::Hsa(format!(
                        "oracle mismatch round={round}: got={g} want={w}"
                    ))));
                }
            }
            for v in &got {
                digest_bytes.extend_from_slice(&v.to_le_bytes());
            }
        }

        total_packets += packet_count;
        total_chunks += chunks_this;
        total_doorbells += chunks_this;
    }

    let fence_count = take_issued_device_store_fence_count();
    if fence_count != total_chunks {
        return Err(CheckError::Rocm(RocmError::Hsa(format!(
            "body-fence count {fence_count} != chunk count {total_chunks}"
        ))));
    }

    let digest = fnv1a(&format!("{:x?}", digest_bytes));
    let rocm_versions = collect_rocm_versions();
    let bdf = format_bdf(provenance.pci_domain, provenance.bdfid);

    println!("=== Card 333 PCIe device-ring publication receipt ===");
    println!("commit: {commit}");
    println!("command: {command}");
    println!("host_arch: {host_arch}");
    println!("host_kernel: {host_kernel}");
    println!("gpu_product: {}", provenance.product_name);
    println!("gpu_agent_name: {}", provenance.agent_name);
    println!("gpu_isa: {}", provenance.isa_name);
    println!("gpu_pci: {bdf}");
    println!("gpu_bdfid: {:?}", provenance.bdfid);
    println!("gpu_pci_domain: {:?}", provenance.pci_domain);
    println!("rocm_versions: {rocm_versions}");
    println!(
        "link_hops: {} link_types: {:?} (PCIE=2)",
        provenance.num_link_hops, provenance.link_types
    );
    println!("queue_api: {}", provenance.queue_api);
    println!(
        "allocate_queue_dev_mem: {}",
        provenance.allocate_queue_dev_mem
    );
    println!(
        "queue_type_requested: {} queue_type_returned: {} features: {} size: {}",
        provenance.queue_type_requested,
        provenance.queue_type_returned,
        provenance.queue_features,
        provenance.queue_size
    );
    // Spec 333 requires base-address class. Public `hsa_queue_t` does not expose the ring
    // allocation flag; provenance is the ROCr `HSA_ALLOCATE_QUEUE_DEV_MEM=1` creation option.
    println!("base_address_class: DeviceMemory (ROCr ALLOCATE_QUEUE_DEV_MEM provenance)");
    println!("publication_contract: {contract:?}");
    println!(
        "packets: {total_packets} chunk_size: {chunk_size} chunks: {total_chunks} \
         body_fences: {fence_count} headers: {total_packets} (reverse-per-chunk) \
         doorbells: {total_doorbells}"
    );
    println!("rounds: {rounds} packets_per_round: {packet_count}");
    println!("output_digest_fnv1a: {digest:#x}");
    println!("completion: ok");
    println!("OK: pcie-publication-receipt");
    Ok(())
}

fn git_commit() -> String {
    // Prefer a live override, then the compile-time bake from build.rs, then a local git probe.
    if let Ok(sha) = std::env::var("POOT_GIT_COMMIT") {
        let trimmed = sha.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    if let Some(sha) = option_env!("POOT_GIT_COMMIT") {
        let trimmed = sha.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                Some(String::from_utf8_lossy(&o.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".into())
}

fn read_text(path: &str) -> Option<String> {
    std::fs::read_to_string(path)
        .ok()
        .map(|s| s.trim().to_string())
}

fn collect_rocm_versions() -> String {
    let mut parts = Vec::new();
    for path in [
        "/opt/rocm/.info/version",
        "/opt/rocm-6.1.0/.info/version",
        "/sys/module/amdgpu/version",
    ] {
        if let Some(v) = read_text(path) {
            parts.push(format!("{path}={v}"));
        }
    }
    if let Ok(out) = Command::new("rocminfo").arg("--version").output()
        && out.status.success()
    {
        let s = String::from_utf8_lossy(&out.stdout);
        let line = s.lines().next().unwrap_or("").trim();
        if !line.is_empty() {
            parts.push(format!("rocminfo={line}"));
        }
    }
    if parts.is_empty() {
        "unknown".into()
    } else {
        parts.join("; ")
    }
}

fn format_bdf(domain: Option<u32>, bdfid: Option<u32>) -> String {
    match (domain, bdfid) {
        (Some(dom), Some(bdf)) => {
            let bus = (bdf >> 8) & 0xff;
            let dev = (bdf >> 3) & 0x1f;
            let func = bdf & 0x7;
            format!("{dom:04x}:{bus:02x}:{dev:02x}.{func}")
        }
        (_, Some(bdf)) => format!("bdfid={bdf:#x}"),
        _ => "unknown".into(),
    }
}

#[derive(Debug, thiserror::Error)]
enum CheckError {
    #[error(transparent)]
    Codegen(#[from] CompileError),
    #[error(transparent)]
    Rocm(#[from] RocmError),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The discrete-GPU receipt skips on an APU host only while ROCm is not required, and no other
    /// backend's variable makes it fail.
    #[test]
    fn pcie_receipt_skip_is_decided_by_the_rocm_requirement_only() {
        let env = |variable: &'static str| move |name: &str| (name == variable).then(|| "1".into());
        assert!(pcie_receipt_may_skip(&RocmError::ApuNotDiscrete, |_| None));
        assert!(!pcie_receipt_may_skip(
            &RocmError::ApuNotDiscrete,
            env(DeviceBackend::Rocm.variable())
        ));
        assert!(
            pcie_receipt_may_skip(&RocmError::ApuNotDiscrete, env("POOT_REQUIRE_GPU")),
            "the retired all-backend switch must not require the ROCm receipt"
        );
        for other in DeviceBackend::ALL
            .into_iter()
            .filter(|other| *other != DeviceBackend::Rocm)
        {
            assert!(
                pcie_receipt_may_skip(&RocmError::ApuNotDiscrete, env(other.variable())),
                "{other:?} must not require the ROCm receipt"
            );
        }
        // A real failure is never a skip, required or not.
        assert!(!pcie_receipt_may_skip(&RocmError::NoGpuAgent, |_| None));
    }
}
