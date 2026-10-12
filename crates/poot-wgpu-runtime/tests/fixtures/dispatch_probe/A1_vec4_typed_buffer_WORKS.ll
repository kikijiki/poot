target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G10"
target triple = "spirv-unknown-vulkan1.3-compute"

@.pn0 = private unnamed_addr constant [5 x i8] c"buf0\00", align 1

define void @main() #0 {
entry:
  %h1 = tail call target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32 0, i32 0, i32 1, i32 0, ptr nonnull @.pn0)
  %l2 = alloca i32
  br label %bb0

bb0:
  %t0 = tail call i32 @llvm.spv.thread.id.i32(i32 0)
  store i32 %t0, ptr %l2
  br label %bb1

bb1:
  %idx = load i32, ptr %l2
  %p1 = tail call ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) %h1, i32 %idx)
  %v1 = load <4 x float>, ptr addrspace(11) %p1, align 16
  %v2 = fadd <4 x float> %v1, <float 1.0, float 1.0, float 1.0, float 1.0>
  store <4 x float> %v2, ptr addrspace(11) %p1, align 16
  ret void
}

declare target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1) @llvm.spv.resource.handlefrombinding.tspirv.VulkanBuffer_a0v4f32_12_1t(i32, i32, i32, i32, ptr)
declare ptr addrspace(11) @llvm.spv.resource.getpointer.p11.tspirv.VulkanBuffer_a0v4f32_12_1t(target("spirv.VulkanBuffer", [0 x <4 x float>], 12, 1), i32)
declare i32 @llvm.spv.thread.id.i32(i32)

attributes #0 = { "hlsl.numthreads"="64,1,1" "hlsl.shader"="compute" }
