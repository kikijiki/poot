target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10"
target triple = "spirv-unknown-vulkan1.3-compute"

; spec 134 P0.75 case 2 FALLBACK: since C1 (one binding, two element types) is INVALID SPIR-V (see that
; fixture's header), this is the viable alternative - the SAME underlying GPU buffer bound at TWO DISTINCT
; binding indices, binding 0 as a scalar `[0 x float]` resource and binding 1 as a vec4 `[0 x <4 x float>]`
; resource. Each binding gets its own `OpVariable` (no dedup collision), so this is expected to validate
; cleanly - and does (confirmed via `llc` + `spirv-val`). The GPU-dispatch half (does RADV actually see the
; SAME memory through both bindings) is the poot-runtime/tests/dispatch_probe.rs
; `mixed_scalar_vec4_two_bindings_alias_one_buffer` test: it calls `Context::dispatch_dev` with the SAME
; `DeviceBuffer` passed as both the sole `ins` entry (binding 0) and `out` (binding 1) - proving
; poot_runtime's EXISTING binding API already supports this fallback with no runtime-side change; only
; poot-codegen needs a per-param dtype-view mechanism to emit it for the P1 dequant kernels.
;
; Layout (8 floats): [0..4) = vec4 "body", [4] = scalar "tail", [5] = written result (vsum+tail).
@.pn0 = private unnamed_addr constant [5 x i8] c"buf0\00", align 1
@.pn1 = private unnamed_addr constant [5 x i8] c"buf1\00", align 1

define void @main() #0 {
entry:
  %h0 = tail call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.pn0)
  %h1 = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32 0, i32 1, i32 1, i32 0, ptr nonnull @.pn1)
  br label %bb0

bb0:
  ; vec4-load the "body" (elements [0,4)) through binding 1.
  %pv = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) %h1, i32 0)
  %v = load <4 x float>, ptr addrspace(11) %pv, align 16
  %v0 = extractelement <4 x float> %v, i32 0
  %v1 = extractelement <4 x float> %v, i32 1
  %v2 = extractelement <4 x float> %v, i32 2
  %v3 = extractelement <4 x float> %v, i32 3
  %s01 = fadd float %v0, %v1
  %s23 = fadd float %v2, %v3
  %vsum = fadd float %s01, %s23

  ; scalar-load the "tail" (element 4) through binding 0 - the SAME physical buffer as binding 1.
  %ps_tail = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %h0, i32 4)
  %tail = load float, ptr addrspace(11) %ps_tail, align 4
  %total = fadd float %vsum, %tail

  ; write the combined result back through binding 0 at element 5.
  %ps_out = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %h0, i32 5)
  store float %total, ptr addrspace(11) %ps_out, align 4
  ret void
}

declare target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32, i32, i32, i32, ptr)
declare target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1), i32)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1), i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
