target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10"
target triple = "spirv-unknown-vulkan1.3-compute"

; review S1 / spec 134 P0.75 case 2: the SAME buffer (descriptor set 0, binding 0) is bound TWICE, once
; as a scalar `[0 x float]` resource and once as a vec4 `[0 x <4 x float>]` resource - two distinct SPIR-V
; resource variables at one binding index. This is the mixed access every dequant kernel needs (vec4-load
; the aligned body, scalar-load the ragged tail).
;
; RESULT (confirmed 2026-07, static-only, no GPU needed): INVALID. `llc` succeeds but the LLVM SPIR-V
; backend deduplicates `handlefrombinding` calls that share (set, binding) down to ONE `OpVariable`,
; keeping whichever element type it saw FIRST (order-independent: swapping which handle is declared/
; called first just swaps which type "wins", confirmed by hand) and emits an illegal
; `OpCopyObject %other_type %buf0` for the other-typed handle - `OpCopyObject` requires its result type
; to equal its operand type, so `spirv-val` rejects it ("Expected Result Type and Operand type to be the
; same"). This is a REJECTION AT llc/spirv-val TIME, before any Vulkan pipeline creation or GPU dispatch -
; poot-runtime/tests/dispatch_probe.rs asserts this compile-time failure (a regression guard), it does NOT
; attempt to dispatch this module (there is no valid SPIR-V to dispatch).
;
; CONCLUSION: one binding index cannot carry two differently-typed resource views through poot's llc
; pipeline. See `C2_mixed_scalar_vec4_two_bindings_WORKS.ll` for the viable fallback - the SAME underlying
; GPU buffer aliased at TWO DISTINCT binding indices (one scalar-typed, one vec4-typed), which compiles,
; validates, AND dispatches correctly (poot_runtime's `dispatch_dev` already supports this: pass the same
; `DeviceBuffer` as both an `ins` entry and `out`). The P1 vec4 dequant adoption needs a per-param
; dtype-view mechanism in poot-codegen (bind the same logical buffer twice, at two binding indices), not
; a single-binding dual-type declaration.
@.pn0s = private unnamed_addr constant [5 x i8] c"buf0\00", align 1
@.pn0v = private unnamed_addr constant [5 x i8] c"buf0\00", align 1

define void @main() #0 {
entry:
  %hs = tail call target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.pn0s)
  %hv = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.pn0v)
  br label %bb0

bb0:
  ; vec4-load the "body": elements [0,4) via the vec handle at vector-index 0.
  %pv = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) %hv, i32 0)
  %v = load <4 x float>, ptr addrspace(11) %pv, align 16
  %v0 = extractelement <4 x float> %v, i32 0
  %v1 = extractelement <4 x float> %v, i32 1
  %v2 = extractelement <4 x float> %v, i32 2
  %v3 = extractelement <4 x float> %v, i32 3
  %s01 = fadd float %v0, %v1
  %s23 = fadd float %v2, %v3
  %vsum = fadd float %s01, %s23

  ; scalar-load the "tail": element 4, via the SCALAR handle over the SAME binding.
  %ps_tail = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %hs, i32 4)
  %tail = load float, ptr addrspace(11) %ps_tail, align 4

  %total = fadd float %vsum, %tail

  ; write the combined result back through the SCALAR handle at element 5, and echo the raw tail value
  ; (read via the scalar handle) at element 6 through the VEC handle's underlying byte range would need a
  ; 4-wide store; keep this probe to the two reads + one scalar write that mirrors the dequant shape.
  %ps_out = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %hs, i32 5)
  store float %total, ptr addrspace(11) %ps_out, align 4

  ; also store the vec4 sum ALONE at element 6, and the raw tail alone at element 7, via the scalar
  ; handle, so a dispatch test can assert each partial value independently (not just the total).
  %ps_out2 = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %hs, i32 6)
  store float %vsum, ptr addrspace(11) %ps_out2, align 4
  %ps_out3 = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1) %hs, i32 7)
  store float %tail, ptr addrspace(11) %ps_out3, align 4
  ret void
}

declare target("spirv.VulkanBuffer", [0 x float], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0f32_12_1t(i32, i32, i32, i32, ptr)
declare target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0f32_12_1t(target("spirv.VulkanBuffer", [0 x float], 12, 1), i32)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1), i32)

attributes #0 = { "hlsl.numthreads"="1,1,1" "hlsl.shader"="compute" }
