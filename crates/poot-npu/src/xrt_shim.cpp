// extern "C" shim over the XRT native C++ API for poot-npu (POOT-641).
//
// XRT's native API is C++ only (xrt::device, xrt::bo, xrt::hw_context, xrt::kernel, xrt::run), so the Rust
// FFI talks to this C surface. Every function catches every C++ exception and reports it as a message in the
// caller's error buffer: no exception crosses the FFI boundary. Each handle is a heap object the Rust RAII
// wrapper (src/xrt.rs) frees exactly once.
//
// The dispatch pattern follows mlir-aie's host runtime (python/aie/utils/hostruntime.py, retrieved
// 2026-10-11, https://github.com/Xilinx/mlir-aie): register the xclbin, open a hardware context, open the
// kernel, hand it opcode 3, the instruction stream BO, its byte length and the data BOs, and wait. The
// pre-POOT-633 poot-npu shim (git 616c2181f) proved the same sequence on this hardware.

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_hw_context.h>
#include <xrt/xrt_kernel.h>

namespace {

// The error buffer the Rust side passes is this long, NUL included.
constexpr std::size_t kErrLen = 512;

void set_err(char* out, const std::string& message) {
    if (out == nullptr) return;
    const std::size_t n = message.size() < kErrLen - 1 ? message.size() : kErrLen - 1;
    std::memcpy(out, message.data(), n);
    out[n] = '\0';
}

struct Device {
    xrt::device device;
};

struct Bo {
    xrt::ext::bo bo;
    void* map;
    std::size_t size;
};

struct Kernel {
    xrt::xclbin xclbin;
    xrt::hw_context context;
    xrt::kernel kernel;
    xrt::bo instructions;
    std::uint32_t instruction_bytes;
    std::uint32_t arg_count;
    // Set when a run was abandoned unfinished: XRT aborts the process if an `xrt::run` is destroyed while its
    // command is still in progress, and a hardware context is not known idle, so such a kernel is never
    // destroyed (see `poot_xrt_kernel_free`).
    bool wedged = false;
};

}  // namespace

extern "C" {

void* poot_xrt_device_open(unsigned index, char* err) {
    if (err) err[0] = '\0';
    try {
        return new Device{xrt::device(index)};
    } catch (const std::exception& e) {
        set_err(err, std::string("device open: ") + e.what());
    } catch (...) {
        set_err(err, "device open: unknown C++ exception");
    }
    return nullptr;
}

void poot_xrt_device_free(void* handle) { delete static_cast<Device*>(handle); }

// A zero-filled, host-mapped buffer object of `bytes` bytes (at least one byte is allocated).
void* poot_xrt_bo_alloc(void* device, std::size_t bytes, char* err) {
    if (err) err[0] = '\0';
    try {
        auto* dev = static_cast<Device*>(device);
        const std::size_t size = bytes == 0 ? 1 : bytes;
        xrt::ext::bo bo(dev->device, size);
        void* map = bo.map();
        std::memset(map, 0, size);
        bo.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        return new Bo{std::move(bo), map, size};
    } catch (const std::exception& e) {
        set_err(err, std::string("bo alloc: ") + e.what());
    } catch (...) {
        set_err(err, "bo alloc: unknown C++ exception");
    }
    return nullptr;
}

void poot_xrt_bo_free(void* handle) { delete static_cast<Bo*>(handle); }

// The host address of the buffer's mapping; valid until the buffer is freed.
void* poot_xrt_bo_map(void* handle) { return static_cast<Bo*>(handle)->map; }

// `to_device` != 0: make host writes visible to the device; else make device writes visible to the host.
int poot_xrt_bo_sync(void* handle, int to_device, std::size_t size, std::size_t offset, char* err) {
    if (err) err[0] = '\0';
    try {
        auto* bo = static_cast<Bo*>(handle);
        bo->bo.sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, size, offset);
        return 0;
    } catch (const std::exception& e) {
        set_err(err, std::string("bo sync: ") + e.what());
    } catch (...) {
        set_err(err, "bo sync: unknown C++ exception");
    }
    return -1;
}

// Register `xclbin`, open a hardware context and the kernel `name`, and upload `instructions` to a
// device-visible instruction buffer. `arg_count` receives the kernel's declared argument count.
void* poot_xrt_kernel_load(void* device, const std::uint8_t* xclbin, std::size_t xclbin_bytes,
                           const char* name, const std::uint8_t* instructions,
                           std::size_t instruction_bytes, std::uint32_t* arg_count, char* err) {
    if (err) err[0] = '\0';
    try {
        auto* dev = static_cast<Device*>(device);
        std::vector<char> image(reinterpret_cast<const char*>(xclbin),
                                reinterpret_cast<const char*>(xclbin) + xclbin_bytes);
        xrt::xclbin xclb(image);
        dev->device.register_xclbin(xclb);
        xrt::hw_context context(dev->device, xclb.get_uuid());
        xrt::kernel kernel(context, std::string(name));
        // Argument 1 is the instruction buffer's bank (mlir-aie hostruntime: `kernel.group_id(1)`).
        xrt::bo insts(dev->device, instruction_bytes, XRT_BO_FLAGS_CACHEABLE,
                      static_cast<xrt::memory_group>(kernel.group_id(1)));
        insts.write(instructions, instruction_bytes, 0);
        insts.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        *arg_count = static_cast<std::uint32_t>(xclb.get_kernel(std::string(name)).get_args().size());
        return new Kernel{std::move(xclb),
                          std::move(context),
                          std::move(kernel),
                          std::move(insts),
                          static_cast<std::uint32_t>(instruction_bytes),
                          *arg_count};
    } catch (const std::exception& e) {
        set_err(err, std::string("kernel load: ") + e.what());
    } catch (...) {
        set_err(err, "kernel load: unknown C++ exception");
    }
    return nullptr;
}

// A kernel that abandoned a run is leaked on purpose: destroying its context or the in-flight command would
// abort the process, and the process teardown reclaims the context.
void poot_xrt_kernel_free(void* handle) {
    auto* k = static_cast<Kernel*>(handle);
    if (k != nullptr && !k->wedged) delete k;
}

// Run the loaded kernel once over `n` data buffers (argument slots 3..3+n) and wait up to `timeout_us`
// microseconds for a successful completion. A timeout or any other terminal state is a failure (-1). The wait
// polls the run's state: XRT's own wait has millisecond resolution, and a finite AIE stream finishes in
// well under one.
int poot_xrt_kernel_run(void* handle, void* const* bos, std::size_t n, std::uint64_t timeout_us,
                        char* err) {
    if (err) err[0] = '\0';
    try {
        auto* k = static_cast<Kernel*>(handle);
        xrt::run run(k->kernel);
        run.set_arg(0, static_cast<std::uint64_t>(3));
        run.set_arg(1, k->instructions);
        run.set_arg(2, k->instruction_bytes);
        for (std::size_t i = 0; i < n; ++i) {
            run.set_arg(static_cast<int>(3 + i), static_cast<Bo*>(bos[i])->bo);
        }
        run.start();
        // From here the command is in flight: whatever ends this scope without a terminal state, a timeout or
        // an exception from the state query or the sleep, abandons the run the same way. XRT aborts the
        // process when an `xrt::run` is destroyed in progress, so a copy of the (reference-counted) run is
        // leaked and the kernel marked wedged.
        auto abandon = [&](const std::string& why) {
            new xrt::run(run);
            k->wedged = true;
            set_err(err, why);
            return -1;
        };
        try {
            const auto deadline =
                std::chrono::steady_clock::now() + std::chrono::microseconds(timeout_us);
            for (;;) {
                const ert_cmd_state state = run.state();
                if (state == ERT_CMD_STATE_COMPLETED) return 0;
                if (state != ERT_CMD_STATE_NEW && state != ERT_CMD_STATE_QUEUED &&
                    state != ERT_CMD_STATE_RUNNING && state != ERT_CMD_STATE_SUBMITTED) {
                    set_err(err, "kernel run: the command ended in state " + std::to_string(state));
                    return -1;
                }
                if (std::chrono::steady_clock::now() >= deadline) {
                    return abandon("kernel run: timed out after " + std::to_string(timeout_us) +
                                   " us");
                }
                std::this_thread::sleep_for(std::chrono::microseconds(20));
            }
        } catch (const std::exception& e) {
            return abandon(std::string("kernel run: ") + e.what());
        } catch (...) {
            return abandon("kernel run: unknown C++ exception while the run was in flight");
        }
    } catch (const std::exception& e) {
        set_err(err, std::string("kernel run: ") + e.what());
    } catch (...) {
        set_err(err, "kernel run: unknown C++ exception");
    }
    return -1;
}

}  // extern "C"
