//! Post-process the SPIR-V binary that `llc` emits, to fix Vulkan-invalid memory semantics and coopmat
//! builtin calls.
//!
//! LLVM 22's SPIR-V backend has one barrier intrinsic (`@llvm.spv.group.memory.barrier.with.group.sync`, the
//! HLSL `GroupMemoryBarrierWithGroupSync`) and hardcodes `OpControlBarrier`'s memory semantics to
//! `SequentiallyConsistent` (0x10). Vulkan forbids that (VUID-StandaloneSpirv-MemorySemantics-10866), so
//! every poot barrier kernel is technically invalid SPIR-V that RADV tolerates. No LLVM intrinsic emits other
//! semantics (checked against libLLVM.so), so the fix is a binary rewrite: repoint each `OpControlBarrier`'s
//! memory-semantics operand at a constant for `AcquireRelease | WorkgroupMemory` (0x8 | 0x100 = 0x108 = 264),
//! what glslang's `barrier()` emits. That is weaker than SeqCst but is what a workgroup barrier means, so
//! behavior is preserved (checked by the executor-equivalence corpus).
//!
//! Everything else is copied verbatim.

const MAGIC: u32 = 0x0723_0203;
const OP_TYPE_INT: u16 = 21;
const OP_CONSTANT: u16 = 43;
const OP_FUNCTION: u16 = 54;
const OP_CONTROL_BARRIER: u16 = 224;
/// AcquireRelease (0x8) | WorkgroupMemory (0x100).
const VULKAN_WG_BARRIER_SEMANTICS: u32 = 0x108;

/// Rewrite `OpControlBarrier` memory semantics from SequentiallyConsistent to AcquireRelease |
/// WorkgroupMemory so the module is valid Vulkan SPIR-V. No-op if there are no barriers; non-SPIR-V input is
/// returned untouched.
pub fn fix_barrier_semantics(words: Vec<u32>) -> Vec<u32> {
    if words.len() < 5 || words[0] != MAGIC {
        return words;
    }

    // Single scan: the 32-bit unsigned int type, any existing `264` constant of it, the first OpFunction
    // (insertion point for a new constant), and every OpControlBarrier's semantics-operand index.
    // Instruction layout: header word = (wordcount << 16) | opcode, then operands.
    let mut uint_type_id: Option<u32> = None;
    let mut existing_264: Option<u32> = None;
    let mut first_function_at: Option<usize> = None;
    let mut barrier_sem_operand_idxs: Vec<usize> = Vec::new();

    let mut i = 5;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        let op = (header & 0xFFFF) as u16;
        if wc == 0 {
            break; // malformed; bail out without changing anything
        }
        match op {
            OP_TYPE_INT
                if wc >= 4
                // OpTypeInt %id <width> <signedness>
                && words[i + 2] == 32 && words[i + 3] == 0 =>
            {
                uint_type_id = Some(words[i + 1]);
            }
            OP_CONSTANT
                if wc == 4
                // OpConstant <type-id> %id <value>
                && Some(words[i + 1]) == uint_type_id && words[i + 3] == VULKAN_WG_BARRIER_SEMANTICS =>
            {
                existing_264 = Some(words[i + 2]);
            }
            OP_FUNCTION if first_function_at.is_none() => {
                first_function_at = Some(i);
            }
            OP_CONTROL_BARRIER if wc == 4 => {
                // OpControlBarrier <exec-scope> <mem-scope> <semantics> ; semantics is operand 3.
                barrier_sem_operand_idxs.push(i + 3);
            }
            _ => {}
        }
        i += wc;
    }

    if barrier_sem_operand_idxs.is_empty() {
        return words; // no barriers, nothing to fix
    }
    let (uint_type_id, insert_at) = match (uint_type_id, first_function_at) {
        (Some(t), Some(f)) => (t, f),
        // A barrier implies a uint type and a kernel implies a function; if either is absent the module
        // is unexpected, so leave it untouched.
        _ => return words,
    };

    // Determine the target semantics constant id, creating one if needed.
    let mut bound = words[3];
    let (sem_id, new_const): (u32, Option<[u32; 4]>) = match existing_264 {
        Some(id) => (id, None),
        None => {
            let id = bound;
            bound += 1;
            // OpConstant (opcode 43, wordcount 4): header, type-id, result-id, value.
            let header = (4u32 << 16) | OP_CONSTANT as u32;
            (
                id,
                Some([header, uint_type_id, id, VULKAN_WG_BARRIER_SEMANTICS]),
            )
        }
    };

    // Rebuild: copy verbatim, inserting the new constant before the first function and repointing barrier
    // semantics operands to `sem_id`.
    let mut out = Vec::with_capacity(words.len() + 4);
    let barrier_set: std::collections::HashSet<usize> =
        barrier_sem_operand_idxs.into_iter().collect();
    for (idx, &w) in words.iter().enumerate() {
        if idx == insert_at
            && let Some(c) = new_const
        {
            out.extend_from_slice(&c);
        }
        if idx == 3 {
            out.push(bound); // updated id bound
        } else if barrier_set.contains(&idx) {
            out.push(sem_id); // repointed semantics operand
        } else {
            out.push(w);
        }
    }
    out
}

/// Rewrite `OpMemoryBarrier`'s Semantics operand to carry a Vulkan-required storage-class bit (spec 138
/// Phase 2). `Statement::Fence` lowers to a generic LLVM `fence`, which the SPIR-V backend turns into
/// `OpMemoryBarrier` with only ordering bits (`Acquire`=0x2 / `Release`=0x4 / `AcquireRelease`=0x8) and no
/// storage-class bit, which `spirv-val` rejects (`VUID-StandaloneSpirv-MemorySemantics-10870`: a
/// non-relaxed order needs one of `UniformMemory`/`WorkgroupMemory`/`ImageMemory`/`OutputMemory`). This adds
/// a missing bit, where the barrier fix above strips a forbidden one; LLVM's generic lowering cannot name a
/// SPIR-V storage class.
///
/// `Emitter::fence_syncscope` already emits a correct Memory Scope operand (`Device`(1)/`Workgroup`(2)), so
/// the class bit is chosen from it: `Workgroup` -> `WorkgroupMemory` (workgroup-scope fences only guard LDS
/// here); `Device` -> `UniformMemory` (Vulkan overloads it for storage-buffer memory; glslang's
/// `memoryBarrierBuffer()` does the same). An unresolved or `CrossDevice`(0) scope gets both bits (safe
/// over-fencing). No-op if the Semantics constant already has a storage-class bit, so it is idempotent and
/// leaves hand-authored modules alone. Everything else is copied verbatim.
pub fn fix_fence_semantics(words: Vec<u32>) -> Vec<u32> {
    if words.len() < 5 || words[0] != MAGIC {
        return words;
    }

    const OP_MEMORY_BARRIER: u16 = 225;
    const OP_CONSTANT_NULL: u16 = 46;
    const CLASS_UNIFORM_MEMORY: u32 = 0x40;
    const CLASS_WORKGROUP_MEMORY: u32 = 0x100;

    // Single scan: the uint type id, every uint-typed `OpConstant`'s (id -> value), every uint-typed
    // `OpConstantNull` id (value 0, i.e. CrossDevice), the first `OpFunction` (insertion point for new
    // constants), and every `OpMemoryBarrier`'s (scope, semantics) operand indices (layout
    // `OpMemoryBarrier <scope> <semantics>`, wordcount 3).
    let mut uint_type_id: Option<u32> = None;
    let mut const_values: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut null_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut first_function_at: Option<usize> = None;
    let mut barriers: Vec<(usize, usize)> = Vec::new();

    let mut i = 5;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        let op = (header & 0xFFFF) as u16;
        if wc == 0 {
            break; // malformed; bail out without changing anything
        }
        match op {
            OP_TYPE_INT if wc >= 4 && words[i + 2] == 32 && words[i + 3] == 0 => {
                uint_type_id = Some(words[i + 1]);
            }
            OP_CONSTANT if wc == 4 && Some(words[i + 1]) == uint_type_id => {
                const_values.insert(words[i + 2], words[i + 3]);
            }
            OP_CONSTANT_NULL if wc == 3 && Some(words[i + 1]) == uint_type_id => {
                null_ids.insert(words[i + 2]);
            }
            OP_FUNCTION if first_function_at.is_none() => {
                first_function_at = Some(i);
            }
            OP_MEMORY_BARRIER if wc == 3 => {
                barriers.push((i + 1, i + 2)); // (scope-operand idx, semantics-operand idx)
            }
            _ => {}
        }
        i += wc;
    }

    if barriers.is_empty() {
        return words; // no memory barriers, nothing to fix
    }
    let (uint_type_id, insert_at) = match (uint_type_id, first_function_at) {
        (Some(t), Some(f)) => (t, f),
        _ => return words, // unexpected shape; leave untouched rather than corrupt it
    };

    let value_of = |id: u32| -> Option<u32> {
        const_values
            .get(&id)
            .copied()
            .or(if null_ids.contains(&id) {
                Some(0)
            } else {
                None
            })
    };

    // Reuse an existing constant with the needed value (avoids duplicates when barriers share semantics).
    let mut value_to_id: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    for (&id, &v) in &const_values {
        value_to_id.entry(v).or_insert(id);
    }

    let mut bound = words[3];
    let mut new_consts: Vec<[u32; 4]> = Vec::new();
    let mut repoint: std::collections::HashMap<usize, u32> = std::collections::HashMap::new();
    for (scope_idx, sem_idx) in barriers {
        let scope_val = value_of(words[scope_idx]);
        let sem_val = value_of(words[sem_idx]).unwrap_or(0);
        let needed_class = match scope_val {
            Some(2) => CLASS_WORKGROUP_MEMORY, // Workgroup scope -> LDS
            Some(1) => CLASS_UNIFORM_MEMORY,   // Device scope -> storage-buffer memory
            // CrossDevice(0) or unresolved: over-fence with both bits.
            _ => CLASS_UNIFORM_MEMORY | CLASS_WORKGROUP_MEMORY,
        };
        let needed = sem_val | needed_class;
        if needed == sem_val {
            continue; // already carries a storage-class bit; no-op
        }
        let target_id = *value_to_id.entry(needed).or_insert_with(|| {
            let id = bound;
            bound += 1;
            let header = (4u32 << 16) | OP_CONSTANT as u32;
            new_consts.push([header, uint_type_id, id, needed]);
            id
        });
        repoint.insert(sem_idx, target_id);
    }

    if repoint.is_empty() {
        return words; // every barrier already had a storage-class bit set
    }

    let mut out = Vec::with_capacity(words.len() + new_consts.len() * 4);
    for (idx, &w) in words.iter().enumerate() {
        if idx == insert_at {
            for c in &new_consts {
                out.extend_from_slice(c);
            }
        }
        if idx == 3 {
            out.push(bound); // updated id bound
        } else if let Some(&target) = repoint.get(&idx) {
            out.push(target); // repointed semantics operand
        } else {
            out.push(w);
        }
    }
    out
}

/// Decorate the compiled module's marked float binops `NoContraction` (card 628):
/// `ordinals` are 0-indexed positions into the instruction-stream order of every `OpFAdd`/`OpFSub`/`OpFMul`
/// the module contains, as `Emitter::float_binop_ordinal` counted them while walking the source `Body`
/// (`emit_llvm_ir_marked`; every `Rvalue::BinaryOp` and `Rvalue::BinaryOpNoContract` float Add/Sub/Mul
/// counts, so the ordinals line up with the compiled stream one for one).
///
/// LLVM 22.1.5's SPIR-V backend does not translate any LLVM-level marker (a fast-math flag, a constrained
/// intrinsic) into a `NoContraction` decoration - checked directly: a `contract`-flagged and a plain
/// `fmul`/`fsub` compile to byte-identical SPIR-V, neither decorated - so the decoration has to be added to
/// the compiled binary instead. That only works because `SpirvVulkan` always compiles at `-O0` (its only
/// `llc_args` opt level): with no CSE, reassociation, dead-code elimination or instruction reordering, the
/// Nth float `OpFAdd`/`OpFSub`/`OpFMul` in the compiled module is exactly the op `emit_llvm_ir_marked`
/// counted as the Nth one in the source body - checked directly (`add_kernel`/three-op probe fixtures
/// compile to the same op count and order as their source statements).
///
/// `OpDecorate` must sit in the module's dedicated annotation section - after the debug instructions
/// (`OpName` etc.) and before the first type/constant/global-variable declaration (SPIR-V's logical layout
/// rule; `spirv-val` rejects a `OpDecorate` placed later, e.g. right before `OpFunction`, as "in an invalid
/// layout section"). Every poot module llc emits already carries at least the buffer binding decorations
/// (`OpDecorate %bufN DescriptorSet`/`Binding`) whenever it has a float op to decorate at all (both need a
/// buffer operand), so inserting right after the last existing `OpDecorate`/`OpMemberDecorate` keeps every
/// decoration contiguous and lands well inside the annotation section. `Err` when the compiled module has
/// no existing decoration to anchor on, or does not contain as many float binops as `ordinals` names (an
/// emitter/postprocess ordinal drift - a silently-missing decoration is the exact bug this function exists
/// to prevent, so a mismatch is a loud, typed failure rather than a best-effort no-op like the barrier/fence
/// fixups above). No-op for an empty `ordinals` or non-SPIR-V input.
/// Why [`fix_no_contraction`] could not place every requested decoration: an emitter/postprocess ordinal
/// drift (card 628), not a malformed body (that fails at `body.verify()` earlier).
#[derive(Debug, thiserror::Error)]
pub enum NoContractionError {
    #[error("no existing OpDecorate/OpMemberDecorate to anchor the new decoration after")]
    NoAnchor,
    #[error(
        "{requested} no-contraction ordinal(s) requested, only {found} float Add/Sub/Mul op(s) found in \
         the compiled module ({total} total)"
    )]
    OrdinalMismatch {
        requested: usize,
        found: usize,
        total: usize,
    },
}

pub fn fix_no_contraction(
    words: Vec<u32>,
    ordinals: &[usize],
) -> Result<Vec<u32>, NoContractionError> {
    if ordinals.is_empty() || words.len() < 5 || words[0] != MAGIC {
        return Ok(words);
    }

    const OP_FADD: u16 = 129;
    const OP_FSUB: u16 = 131;
    const OP_FMUL: u16 = 133;
    const OP_DECORATE: u16 = 71;
    const OP_MEMBER_DECORATE: u16 = 72;
    const DECORATION_NO_CONTRACTION: u32 = 42;

    let wanted: std::collections::HashSet<usize> = ordinals.iter().copied().collect();

    let mut i = 5;
    let mut ordinal = 0usize;
    let mut target_ids: Vec<u32> = Vec::new();
    let mut last_decoration_end: Option<usize> = None;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        let op = (header & 0xFFFF) as u16;
        if wc == 0 {
            break; // malformed; bail out without changing anything
        }
        match op {
            OP_DECORATE | OP_MEMBER_DECORATE => {
                last_decoration_end = Some(i + wc);
            }
            OP_FADD | OP_FSUB | OP_FMUL if wc == 5 => {
                // OpFAdd/OpFSub/OpFMul <type-id> <result-id> <operand1> <operand2> (wordcount 5: the
                // header word, then those four).
                if wanted.contains(&ordinal) {
                    target_ids.push(words[i + 2]);
                }
                ordinal += 1;
            }
            _ => {}
        }
        i += wc;
    }

    let Some(insert_at) = last_decoration_end else {
        return Err(NoContractionError::NoAnchor);
    };
    if target_ids.len() != wanted.len() {
        return Err(NoContractionError::OrdinalMismatch {
            requested: wanted.len(),
            found: target_ids.len(),
            total: ordinal,
        });
    }

    let mut new_decorations: Vec<u32> = Vec::with_capacity(target_ids.len() * 3);
    for id in &target_ids {
        new_decorations.push((3u32 << 16) | OP_DECORATE as u32);
        new_decorations.push(*id);
        new_decorations.push(DECORATION_NO_CONTRACTION);
    }

    let mut out = Vec::with_capacity(words.len() + new_decorations.len());
    for (idx, &w) in words.iter().enumerate() {
        if idx == insert_at {
            out.extend_from_slice(&new_decorations);
        }
        out.push(w);
    }
    Ok(out)
}

/// Rewrite Import-linkage `OpFunctionCall`s to the LLVM SPIR-V backend's
/// `__spirv_CooperativeMatrix{Load,Store,MulAdd}KHR`/`__spirv_CompositeConstruct` builtin stubs into real
/// `OpCooperativeMatrix{Load,Store,MulAdd}KHR`/`OpCompositeConstruct` instructions, then strip the dead
/// stub declarations, their `Import` decorations and names, and (if nothing else needs it) the `Linkage`
/// capability (card 154).
///
/// LLVM 22.1.5's SPIR-V backend emits correct `OpTypeCooperativeMatrixKHR` types plus
/// `OpCapability CooperativeMatrixKHR` / `OpExtension "SPV_KHR_cooperative_matrix"` under both the
/// Kernel/OpenCL and Vulkan/Shader models (`-mtriple=spirv64-unknown-unknown` vs
/// `-mtriple=spirv-unknown-vulkan1.3-compute`, with `--spirv-ext=+SPV_KHR_cooperative_matrix`), but lowers
/// the builtin calls to real instructions only under Kernel/OpenCL. Under Vulkan/Shader (poot's target) it
/// leaves each `__spirv_CooperativeMatrix*KHR` call as an `OpFunctionCall` to a declaration-only stub
/// decorated `LinkageAttributes "..." Import`, and emits `OpCapability Linkage` to license that decoration.
/// Vulkan forbids `Linkage` (`spirv-val --target-env vulkan1.3`: "Capability Linkage is not allowed by
/// Vulkan 1.3 specification (or requires extension)"), the only validation error on the unrewritten module
/// (see `specs/113-vulkan-coopmat/probe-fixtures/`).
///
/// The rewrite is mechanical: `OpFunctionCall`'s operand layout (ResultType, ResultId, Function,
/// Arg0..ArgN) matches the real instructions' argument-for-argument (Load/MulAdd: ResultType, ResultId,
/// Arg0..ArgN; Store: Arg0..ArgN, no result). So drop the `Function` operand (and, for Store, the unused
/// void ResultType/ResultId pair), repoint the opcode, and delete the callee's declaration, decoration, and
/// name once nothing calls it. No-op if no recognized coopmat builtin call is present.
///
/// `__spirv_CompositeConstruct`: the SPIR-V spec extends `OpCompositeConstruct` to build a
/// cooperative-matrix value from a single scalar Constituent broadcast into every component (how a zero
/// accumulator is seeded; GLSL's `coopmat(0.0)` lowers to it). LLVM cannot construct an opaque
/// target-extension-type value from IR, so poot's emitter calls this builtin like the other three and hits
/// the same Import-stub gap. It has no trailing bitmask operand: `mask_arg_count` returns a sentinel it can
/// never match, so its calls take the "no trailing mask" branch, and it has Load/MulAdd's has-result shape
/// (ResultType, ResultId, then args verbatim).
///
/// Called from `fixup_spirv_barriers` in `lib.rs`, so every SpirvVulkan compile gets it; a no-op unless the
/// IR contains a coopmat builtin call (see `wmma_check_target`/`emit_wmma_*_coopmat` in `emit.rs`).
pub fn fix_coopmat_calls(words: Vec<u32>) -> Vec<u32> {
    if words.len() < 5 || words[0] != MAGIC {
        return words;
    }

    const OP_NAME: u16 = 5;
    const OP_EXTENSION: u16 = 10;
    const OP_MEMORY_MODEL: u16 = 14;
    const OP_CAPABILITY: u16 = 17;
    const OP_FUNCTION_PARAMETER: u16 = 55;
    const OP_FUNCTION_END: u16 = 56;
    const OP_FUNCTION_CALL: u16 = 57;
    const OP_DECORATE: u16 = 71;
    const DECORATION_LINKAGE_ATTRIBUTES: u32 = 41;
    const LINKAGE_TYPE_IMPORT: u32 = 1;
    const CAPABILITY_LINKAGE: u32 = 5;
    const CAPABILITY_VULKAN_MEMORY_MODEL: u32 = 5345;
    const MEMORY_MODEL_VULKAN: u32 = 3;
    const OP_COOP_LOAD: u16 = 4457;
    const OP_COOP_STORE: u16 = 4458;
    const OP_COOP_MULADD: u16 = 4459;
    const OP_COMPOSITE_CONSTRUCT: u16 = 80;

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum Kind {
        Load,
        Store,
        MulAdd,
        CompositeConstruct,
    }

    // Decode the null-terminated `LiteralString` packed 4 bytes/word (little-endian) from `words[start..end]`.
    fn decode_string(words: &[u32], start: usize, end: usize) -> String {
        let mut s = String::new();
        'outer: for &w in &words[start..end] {
            for shift in [0, 8, 16, 24] {
                let b = ((w >> shift) & 0xFF) as u8;
                if b == 0 {
                    break 'outer;
                }
                s.push(b as char);
            }
        }
        s
    }

    fn classify(name: &str) -> Option<Kind> {
        if name.contains("CooperativeMatrixLoadKHR") {
            Some(Kind::Load)
        } else if name.contains("CooperativeMatrixStoreKHR") {
            Some(Kind::Store)
        } else if name.contains("CooperativeMatrixMulAddKHR") {
            Some(Kind::MulAdd)
        } else if name.contains("CompositeConstruct") {
            // Only the coopmat zero-accumulator constructor (see the doc comment) emits this: LLVM builds real
            // aggregates with insertvalue/extractvalue, never a "__spirv_" builtin call.
            Some(Kind::CompositeConstruct)
        } else {
            None
        }
    }

    // Encode one instruction (word count from `operands.len()`, no result/type).
    fn instr(opcode: u16, operands: &[u32]) -> Vec<u32> {
        let wc = (1 + operands.len()) as u32;
        let mut v = vec![(wc << 16) | opcode as u32];
        v.extend_from_slice(operands);
        v
    }

    // Pack a null-terminated ASCII string as a 4-bytes/word `LiteralString`.
    fn pack_str(s: &str) -> Vec<u32> {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // How many `Arg`s an `OpFunctionCall` of this kind carries when its trailing bitmask operand (`Memory
    // Operand` for Load/Store, `Cooperative Matrix Operands` for MulAdd) is present. The builtin includes
    // an optional trailing operand only if all earlier operands are present (SPIR-V's in-order rule), so
    // arg count alone tells whether the last arg is that bitmask. Baseline counts without it: Load 2
    // (Pointer, MemoryLayout), Store 3 (Pointer, Object, MemoryLayout), MulAdd 3 (A, B, C).
    fn mask_arg_count(kind: Kind) -> usize {
        match kind {
            Kind::Load => 4,   // Pointer, MemoryLayout, Stride, Memory Operand
            Kind::Store => 5,  // Pointer, Object, MemoryLayout, Stride, Memory Operand
            Kind::MulAdd => 4, // A, B, C, Cooperative Matrix Operands
            // No optional trailing bitmask (just one scalar Constituent): a sentinel no real call reaches.
            Kind::CompositeConstruct => usize::MAX,
        }
    }

    // Pass 1 collects: every 32-bit `OpConstant`'s value (to resolve the trailing bitmask operand); every
    // `OpDecorate <id> LinkageAttributes "<name>" Import`, classified by name and tracked per id so a
    // later pass can keep a declaration that turns out not to be safely rewritable; every such
    // decoration's span (`all_linkage_decorate_spans`), so the final pass can tell whether `Linkage` is
    // still needed; and the `OpCapability Linkage` span, if present.
    let mut consts: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    let mut builtin_targets: std::collections::HashMap<u32, Kind> =
        std::collections::HashMap::new();
    let mut decorate_span: std::collections::HashMap<u32, (usize, usize)> =
        std::collections::HashMap::new();
    let mut all_linkage_decorate_spans: Vec<(usize, usize)> = Vec::new();
    let mut linkage_capability_span: Option<(usize, usize)> = None;
    // Insertion point for the `VulkanMemoryModel` capability/extension this rewrite requires (see below):
    // after the last `OpCapability`, or after the header if there are none. `has_vulkan_memory_model` /
    // `has_vulkan_memory_model_ext` skip the insert if the module already carries them.
    let mut after_last_capability = 5;
    let mut has_vulkan_memory_model = false;
    let mut has_vulkan_memory_model_ext = false;
    let mut memory_model_operand_idx: Option<usize> = None;

    let mut i = 5;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        let op = (header & 0xFFFF) as u16;
        if wc == 0 {
            break; // malformed; bail out without changing anything
        }
        match op {
            OP_CONSTANT if wc == 4 => {
                consts.insert(words[i + 2], words[i + 3]);
            }
            OP_CAPABILITY if wc == 2 => {
                after_last_capability = i + wc;
                if words[i + 1] == CAPABILITY_LINKAGE {
                    linkage_capability_span = Some((i, i + wc));
                } else if words[i + 1] == CAPABILITY_VULKAN_MEMORY_MODEL {
                    has_vulkan_memory_model = true;
                }
            }
            OP_EXTENSION
                if wc >= 2
                    && decode_string(&words, i + 1, i + wc) == "SPV_KHR_vulkan_memory_model" =>
            {
                has_vulkan_memory_model_ext = true;
            }
            OP_MEMORY_MODEL if wc == 3 => {
                memory_model_operand_idx = Some(i + 2);
            }
            OP_DECORATE
                if wc >= 5
                    && words[i + 2] == DECORATION_LINKAGE_ATTRIBUTES
                    && words[i + wc - 1] == LINKAGE_TYPE_IMPORT =>
            {
                let span = (i, i + wc);
                all_linkage_decorate_spans.push(span);
                let name = decode_string(&words, i + 3, i + wc - 1);
                if let Some(kind) = classify(&name) {
                    builtin_targets.insert(words[i + 1], kind);
                    decorate_span.insert(words[i + 1], span);
                }
            }
            _ => {}
        }
        i += wc;
    }

    if builtin_targets.is_empty() {
        return words; // no recognized coopmat builtin import found
    }

    // Pass 2 collects: the matched builtins' `OpName`s, their declaration-only `OpFunction..OpFunctionEnd`
    // stubs, and every `OpFunctionCall` invoking one. A trailing bitmask operand is resolved from an `<id>`
    // to the literal the real instruction needs: `MemoryAccess`/`CooperativeMatrixOperands` are literals,
    // not `IdRef`s, the one operand the builtin-call form and the real instruction encode differently.
    let mut name_span: std::collections::HashMap<u32, (usize, usize)> =
        std::collections::HashMap::new();
    let mut function_span: std::collections::HashMap<u32, (usize, usize)> =
        std::collections::HashMap::new();
    // (instr start, wordcount, kind, resolved trailing-mask literal if this call has one)
    let mut call_candidates: Vec<(usize, usize, u32, Kind, Option<u32>)> = Vec::new();
    let mut unrewritable_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    i = 5;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        let op = (header & 0xFFFF) as u16;
        if wc == 0 {
            break;
        }
        match op {
            OP_NAME if wc >= 2 && builtin_targets.contains_key(&words[i + 1]) => {
                name_span.insert(words[i + 1], (i, i + wc));
            }
            OP_FUNCTION if wc == 5 && builtin_targets.contains_key(&words[i + 2]) => {
                // Declaration-only stub: zero or more `OpFunctionParameter`, then `OpFunctionEnd` with no
                // basic block. If a real body follows, this id is not an import stub; leave it alone.
                let mut j = i + wc;
                let mut end = None;
                while j < words.len() {
                    let h2 = words[j];
                    let wc2 = (h2 >> 16) as usize;
                    let op2 = (h2 & 0xFFFF) as u16;
                    if wc2 == 0 {
                        break;
                    }
                    match op2 {
                        OP_FUNCTION_PARAMETER => j += wc2,
                        OP_FUNCTION_END => {
                            end = Some(j + wc2);
                            break;
                        }
                        _ => break,
                    }
                }
                if let Some(end) = end {
                    function_span.insert(words[i + 2], (i, end));
                }
            }
            OP_FUNCTION_CALL if wc >= 4 => {
                let target = words[i + 3];
                if let Some(&kind) = builtin_targets.get(&target) {
                    let num_args = wc - 4;
                    if num_args == mask_arg_count(kind) {
                        let mask_id = words[i + wc - 1];
                        match consts.get(&mask_id) {
                            Some(&v) => call_candidates.push((i, wc, target, kind, Some(v))),
                            // The bitmask arg is not a resolvable constant: leave the call and its stub alone.
                            None => {
                                unrewritable_ids.insert(target);
                            }
                        }
                    } else {
                        call_candidates.push((i, wc, target, kind, None));
                    }
                }
            }
            _ => {}
        }
        i += wc;
    }

    let calls: Vec<(usize, usize, Kind, Option<u32>)> = call_candidates
        .into_iter()
        .filter(|&(_, _, target, ..)| !unrewritable_ids.contains(&target))
        .map(|(s, wc, _, k, m)| (s, wc, k, m))
        .collect();
    if calls.is_empty() {
        return words; // matched decorations were dead weight, or nothing was safely rewritable
    }
    let rewritable_ids: std::collections::HashSet<u32> = builtin_targets
        .keys()
        .filter(|id| !unrewritable_ids.contains(id))
        .copied()
        .collect();

    let mut spans_to_delete: Vec<(usize, usize)> = Vec::new();
    for &id in &rewritable_ids {
        if let Some(&s) = decorate_span.get(&id) {
            spans_to_delete.push(s);
        }
        if let Some(&s) = name_span.get(&id) {
            spans_to_delete.push(s);
        }
        if let Some(&s) = function_span.get(&id) {
            spans_to_delete.push(s);
        }
    }

    // `Linkage` is required by any `Import`/`Export` decoration; drop it only if every remaining
    // `LinkageAttributes` decoration is one this pass removes.
    if let Some(cap_span) = linkage_capability_span
        && all_linkage_decorate_spans
            .iter()
            .all(|s| spans_to_delete.contains(s))
    {
        spans_to_delete.push(cap_span);
    }

    let in_deleted_range = |idx: usize| spans_to_delete.iter().any(|&(s, e)| idx >= s && idx < e);
    let call_at: std::collections::HashMap<usize, (usize, Kind, Option<u32>)> = calls
        .into_iter()
        .map(|(s, wc, k, m)| (s, (wc, k, m)))
        .collect();

    // Real coopmat instructions are legal under Vulkan only when `VulkanMemoryModel` is declared and
    // `OpMemoryModel` selects it (SPV_KHR_cooperative_matrix: Shader + CooperativeMatrixKHR requires it).
    // This is separate from the Import/Linkage gap, and the LLVM SPIR-V backend does not infer it for a
    // plain coopmat kernel: even with `--spirv-ext=+SPV_KHR_vulkan_memory_model` it emits `OpMemoryModel
    // Logical GLSL450` unless the IR needs release/acquire semantics. This pass creates the real coopmat
    // instructions, so it patches this too.
    let mut capability_prelude: Vec<u32> = Vec::new();
    if !has_vulkan_memory_model {
        capability_prelude
            .extend_from_slice(&instr(OP_CAPABILITY, &[CAPABILITY_VULKAN_MEMORY_MODEL]));
    }
    if !has_vulkan_memory_model_ext {
        let mut operands = Vec::new();
        operands.extend(pack_str("SPV_KHR_vulkan_memory_model"));
        capability_prelude.extend_from_slice(&instr(OP_EXTENSION, &operands));
    }
    let memory_model_patch_idx = memory_model_operand_idx;

    let mut out = Vec::with_capacity(words.len() + capability_prelude.len());
    out.extend_from_slice(&words[..5]);
    i = 5;
    while i < words.len() {
        let header = words[i];
        let wc = (header >> 16) as usize;
        if wc == 0 {
            out.extend_from_slice(&words[i..]);
            break;
        }
        if i == after_last_capability {
            out.extend_from_slice(&capability_prelude);
        }
        if let Some(mm_idx) = memory_model_patch_idx
            && i == mm_idx - 2
        {
            out.push(words[i]); // header (unchanged: OpMemoryModel, wc 3)
            out.push(words[i + 1]); // addressing model (unchanged)
            out.push(MEMORY_MODEL_VULKAN);
        } else if let Some(&(call_wc, kind, mask)) = call_at.get(&i) {
            debug_assert_eq!(call_wc, wc);
            // Args verbatim, except a trailing bitmask arg becomes the literal its `<id>` constant-folds to
            // (`OpCooperativeMatrix{Load,Store,MulAdd}KHR` encode it as a literal, not an IdRef).
            let mut args: Vec<u32> = words[i + 4..i + wc].to_vec();
            if let Some(literal) = mask {
                *args.last_mut().unwrap() = literal;
            }
            match kind {
                Kind::Load | Kind::MulAdd | Kind::CompositeConstruct => {
                    let opcode = match kind {
                        Kind::Load => OP_COOP_LOAD,
                        Kind::MulAdd => OP_COOP_MULADD,
                        Kind::CompositeConstruct => OP_COMPOSITE_CONSTRUCT,
                        Kind::Store => unreachable!(),
                    };
                    let new_wc = (wc - 1) as u32;
                    out.push((new_wc << 16) | opcode as u32);
                    out.push(words[i + 1]); // result type
                    out.push(words[i + 2]); // result id
                    out.extend_from_slice(&args);
                }
                Kind::Store => {
                    let new_wc = (wc - 3) as u32;
                    out.push((new_wc << 16) | OP_COOP_STORE as u32);
                    out.extend_from_slice(&args);
                }
            }
        } else if !in_deleted_range(i) {
            out.extend_from_slice(&words[i..i + wc]);
        }
        i += wc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal hand-built SPIR-V module: header + uint type + SeqCst(16) const + a function with one
    // OpControlBarrier using the 16 const as semantics. Verifies the rewrite repoints to a new 264 const.
    fn module() -> Vec<u32> {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 7 /*bound*/, 0];
        // %1 = OpTypeInt 32 0
        w.extend_from_slice(&[(4 << 16) | OP_TYPE_INT as u32, 1, 32, 0]);
        // %2 = OpConstant %1 2  (a scope const, unrelated)
        w.extend_from_slice(&[(4 << 16) | OP_CONSTANT as u32, 1, 2, 2]);
        // %3 = OpConstant %1 16 (SeqCst semantics)
        w.extend_from_slice(&[(4 << 16) | OP_CONSTANT as u32, 1, 3, 16]);
        // %4 = OpFunction ... (3 operands -> wc 5: header, result-type, result-id, function-control, fn-type)
        w.extend_from_slice(&[(5 << 16) | OP_FUNCTION as u32, 5, 4, 0, 6]);
        // OpControlBarrier %2 %2 %3
        w.extend_from_slice(&[(4 << 16) | OP_CONTROL_BARRIER as u32, 2, 2, 3]);
        w
    }

    #[test]
    fn repoints_barrier_to_a_264_constant() {
        let out = fix_barrier_semantics(module());
        // bound grew by one (a new constant id 7 was added).
        assert_eq!(out[3], 8);
        // a new OpConstant %1 264 was inserted with id 7.
        let const_hdr = (4u32 << 16) | OP_CONSTANT as u32;
        let mut found_264 = false;
        let mut i = 5;
        while i < out.len() {
            let wc = (out[i] >> 16) as usize;
            if out[i] == const_hdr && out[i + 1] == 1 && out[i + 3] == 264 {
                found_264 = true;
                assert_eq!(out[i + 2], 7, "new const got id == old bound");
            }
            i += wc.max(1);
        }
        assert!(found_264, "a 264 semantics constant was inserted");
        // the OpControlBarrier semantics operand now points at id 7, not 3.
        let bar_hdr = (4u32 << 16) | OP_CONTROL_BARRIER as u32;
        let pos = out.iter().position(|&x| x == bar_hdr).unwrap();
        assert_eq!(
            out[pos + 3],
            7,
            "barrier semantics repointed to the 264 const"
        );
    }

    #[test]
    fn no_barrier_is_unchanged() {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 3, 0];
        w.extend_from_slice(&[(4 << 16) | OP_TYPE_INT as u32, 1, 32, 0]);
        let before = w.clone();
        assert_eq!(fix_barrier_semantics(w), before);
    }

    #[test]
    fn non_spirv_untouched() {
        let junk = vec![1u32, 2, 3];
        assert_eq!(fix_barrier_semantics(junk.clone()), junk);
    }

    const OP_MEMORY_BARRIER: u32 = 225;

    // A hand-built module with one `OpMemoryBarrier %scope_id %sem_id`, where `scope_id`'s constant has
    // value `scope_val` and `sem_id`'s has `sem_val` (distinct ids, so the test does not depend on the
    // emitter reusing one id for both operands).
    fn fence_module(scope_val: u32, sem_val: u32) -> Vec<u32> {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 6 /*bound*/, 0];
        // %1 = OpTypeInt 32 0
        w.extend_from_slice(&[(4 << 16) | OP_TYPE_INT as u32, 1, 32, 0]);
        // %2 = OpConstant %1 <scope_val>
        w.extend_from_slice(&[(4 << 16) | OP_CONSTANT as u32, 1, 2, scope_val]);
        // %3 = OpConstant %1 <sem_val>
        w.extend_from_slice(&[(4 << 16) | OP_CONSTANT as u32, 1, 3, sem_val]);
        // %4 = OpFunction ...
        w.extend_from_slice(&[(5 << 16) | OP_FUNCTION as u32, 5, 4, 0, 6]);
        // OpMemoryBarrier %2 %3
        w.extend_from_slice(&[(3 << 16) | OP_MEMORY_BARRIER, 2, 3]);
        w
    }

    // The value of the OpMemoryBarrier's semantics operand (word[+2]), resolved through the constant table.
    fn semantics_value(words: &[u32]) -> u32 {
        let bar_hdr = (3u32 << 16) | OP_MEMORY_BARRIER;
        let pos = words.iter().position(|&x| x == bar_hdr).unwrap();
        let sem_id = words[pos + 2];
        let const_hdr = (4u32 << 16) | OP_CONSTANT as u32;
        let mut i = 5;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            if words[i] == const_hdr && words[i + 2] == sem_id {
                return words[i + 3];
            }
            i += wc.max(1);
        }
        panic!("semantics constant {sem_id} not found");
    }

    #[test]
    fn workgroup_scope_barrier_gets_workgroup_memory_bit() {
        // scope=Workgroup(2), semantics=AcquireRelease(8) only: WorkgroupMemory(0x100) is added, giving
        // the same 264 (0x108) as the control-barrier fix (the canonical workgroup acquire-release value).
        let out = fix_fence_semantics(fence_module(2, 8));
        assert_eq!(semantics_value(&out), 0x108);
        assert_eq!(out[3], 7, "bound grew by one for the new constant");
    }

    #[test]
    fn device_scope_barrier_gets_uniform_memory_bit() {
        // scope=Device(1), semantics=Acquire(2) only -> needs UniformMemory(0x40) added.
        let out = fix_fence_semantics(fence_module(1, 2));
        assert_eq!(semantics_value(&out), 0x42);
    }

    #[test]
    fn semantics_already_carrying_a_class_bit_is_unchanged() {
        // Semantics already has WorkgroupMemory(0x100) (0x102): no-op, byte-identical, no bound growth.
        let m = fence_module(2, 0x102);
        let before = m.clone();
        assert_eq!(fix_fence_semantics(m), before);
    }

    #[test]
    fn no_memory_barrier_is_unchanged() {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 3, 0];
        w.extend_from_slice(&[(4 << 16) | OP_TYPE_INT as u32, 1, 32, 0]);
        let before = w.clone();
        assert_eq!(fix_fence_semantics(w), before);
    }

    #[test]
    fn non_spirv_untouched_fence() {
        let junk = vec![1u32, 2, 3];
        assert_eq!(fix_fence_semantics(junk.clone()), junk);
    }

    // --- fix_coopmat_calls -------------------------------------------------------------------

    const OP_CAPABILITY: u16 = 17;
    const OP_DECORATE: u16 = 71;
    const OP_NAME: u16 = 5;
    const OP_FUNCTION_END: u16 = 56;
    const OP_FUNCTION_CALL: u16 = 57;

    fn instr(opcode: u16, operands: &[u32]) -> Vec<u32> {
        let wc = (1 + operands.len()) as u32;
        let mut v = vec![(wc << 16) | opcode as u32];
        v.extend_from_slice(operands);
        v
    }

    /// Pack a null-terminated ASCII string into SPIR-V's 4-bytes/word `LiteralString` encoding.
    fn pack_str(s: &str) -> Vec<u32> {
        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn decorate_import(target: u32, name: &str) -> Vec<u32> {
        let mut operands = vec![target, 41 /* LinkageAttributes */];
        operands.extend(pack_str(name));
        operands.push(1 /* Import */);
        instr(OP_DECORATE, &operands)
    }

    fn name_instr(target: u32, name: &str) -> Vec<u32> {
        let mut operands = vec![target];
        operands.extend(pack_str(name));
        instr(OP_NAME, &operands)
    }

    /// A hand-built module mirroring the real `llc -mtriple=spirv-unknown-vulkan1.3-compute
    /// --spirv-ext=+SPV_KHR_cooperative_matrix` output shape (see `specs/113-vulkan-coopmat/probe-fixtures/`):
    /// `OpCapability Linkage`, one `Import`-decorated declaration-only stub per builtin (Load/MulAdd/Store),
    /// each called once via `OpFunctionCall`, plus an unrelated `main` that must survive. Type ids are
    /// placeholders (`fix_coopmat_calls` inspects only ids and opcodes).
    fn coopmat_vulkan_module() -> Vec<u32> {
        // ids: 1=void, 2=dummy-coop-result-type, 10=LoadFn, 11=MulAddFn, 12=StoreFn, 20=main,
        // 30=load-result, 31=muladd-result, 32=store-call-result(unused, void)
        let mut w = vec![
            MAGIC,
            0x0001_0300,
            0,
            210, /* bound (ids run up to 206) */
            0,
        ];
        // %204 = OpConstant %6 12: the MulAdd call's trailing `Cooperative Matrix Operands` mask
        // (MatrixCSignedComponentsKHR|MatrixResultSignedComponentsKHR = 4|8 = 12), passed as an `<id>` in
        // the builtin-call form; the real instruction needs a literal (see `mask_arg_count`), so the
        // rewrite resolves id 204 -> 12.
        w.extend_from_slice(&instr(OP_CONSTANT, &[6, 204, 12]));
        w.extend_from_slice(&instr(OP_CAPABILITY, &[5 /* Linkage */]));
        w.extend_from_slice(&decorate_import(10, "__spirv_CooperativeMatrixLoadKHR_1"));
        w.extend_from_slice(&decorate_import(11, "__spirv_CooperativeMatrixMulAddKHR"));
        w.extend_from_slice(&decorate_import(12, "__spirv_CooperativeMatrixStoreKHR"));
        w.extend_from_slice(&name_instr(10, "load"));
        w.extend_from_slice(&name_instr(11, "muladd"));
        w.extend_from_slice(&name_instr(12, "store"));
        // declaration-only stubs: OpFunction (wc 5) directly followed by OpFunctionEnd.
        w.extend_from_slice(&instr(OP_FUNCTION, &[2, 10, 0, 100]));
        w.extend_from_slice(&instr(OP_FUNCTION_END, &[]));
        w.extend_from_slice(&instr(OP_FUNCTION, &[2, 11, 0, 101]));
        w.extend_from_slice(&instr(OP_FUNCTION_END, &[]));
        w.extend_from_slice(&instr(OP_FUNCTION, &[1, 12, 0, 102]));
        w.extend_from_slice(&instr(OP_FUNCTION_END, &[]));
        // main: OpFunction, one real basic block, the three builtin calls, OpReturn, OpFunctionEnd.
        w.extend_from_slice(&instr(OP_FUNCTION, &[1, 20, 0, 103]));
        w.extend_from_slice(&instr(OP_FUNCTION_CALL, &[2, 30, 10, 200, 201]));
        w.extend_from_slice(&instr(OP_FUNCTION_CALL, &[2, 31, 11, 30, 202, 203, 204]));
        w.extend_from_slice(&instr(OP_FUNCTION_CALL, &[1, 32, 12, 205, 31, 206]));
        w.extend_from_slice(&instr(253 /* OpReturn */, &[]));
        w.extend_from_slice(&instr(OP_FUNCTION_END, &[]));
        w
    }

    fn find_opcode(words: &[u32], opcode: u16) -> Vec<usize> {
        let mut out = Vec::new();
        let mut i = 5;
        while i < words.len() {
            let wc = (words[i] >> 16) as usize;
            if wc == 0 {
                break;
            }
            if (words[i] & 0xFFFF) as u16 == opcode {
                out.push(i);
            }
            i += wc;
        }
        out
    }

    #[test]
    fn rewrites_import_calls_to_real_coopmat_instructions() {
        let before = coopmat_vulkan_module();
        let after = fix_coopmat_calls(before);

        // No OpFunctionCall survives (all three builtins were rewritten).
        assert!(
            find_opcode(&after, OP_FUNCTION_CALL).is_empty(),
            "an OpFunctionCall remains: {after:?}"
        );

        // OpCooperativeMatrixLoadKHR, MulAddKHR, StoreKHR each appear once, callee operand dropped and
        // the rest preserved in order.
        let load = find_opcode(&after, 4457);
        assert_eq!(load.len(), 1, "expected exactly one real Load");
        let li = load[0];
        assert_eq!(
            (after[li] >> 16) as usize,
            5,
            "Load: result-type, result-id, 2 args -> wc 5"
        );
        assert_eq!(&after[li + 1..li + 5], &[2, 30, 200, 201]);

        let muladd = find_opcode(&after, 4459);
        assert_eq!(muladd.len(), 1, "expected exactly one real MulAdd");
        let mi = muladd[0];
        // Last arg (204) resolves through the OpConstant table to the literal 12.
        assert_eq!(&after[mi + 1..mi + 7], &[2, 31, 30, 202, 203, 12]);

        let store = find_opcode(&after, 4458);
        assert_eq!(store.len(), 1, "expected exactly one real Store");
        let si = store[0];
        assert_eq!(
            (after[si] >> 16) as usize,
            4,
            "Store: no result, 3 args -> wc 4"
        );
        assert_eq!(&after[si + 1..si + 4], &[205, 31, 206]);

        // The now-dead stub declarations, their Import decorations/names, and the now-unused
        // Linkage capability are all gone.
        // `Linkage` is gone; `VulkanMemoryModel` was added (required for real coopmat instructions under
        // Vulkan; see `fix_coopmat_calls`).
        let capabilities: Vec<u32> = find_opcode(&after, OP_CAPABILITY)
            .into_iter()
            .map(|i| after[i + 1])
            .collect();
        assert_eq!(capabilities, vec![5345], "expected only VulkanMemoryModel");
        let decorate_targets: Vec<u32> = find_opcode(&after, OP_DECORATE)
            .into_iter()
            .map(|i| after[i + 1])
            .collect();
        assert!(!decorate_targets.contains(&10));
        assert!(!decorate_targets.contains(&11));
        assert!(!decorate_targets.contains(&12));
        let name_targets: Vec<u32> = find_opcode(&after, OP_NAME)
            .into_iter()
            .map(|i| after[i + 1])
            .collect();
        assert!(!name_targets.contains(&10));
        assert!(!name_targets.contains(&11));
        assert!(!name_targets.contains(&12));

        // `main` (id 20) is untouched: one OpFunction with a real body (OpReturn), not treated as a stub
        // since it was never Import-decorated.
        let functions = find_opcode(&after, OP_FUNCTION);
        assert!(functions.iter().any(|&i| after[i + 2] == 20));
        assert_eq!(functions.len(), 1, "only main's OpFunction should remain");
        assert_eq!(find_opcode(&after, 253 /* OpReturn */).len(), 1);
    }

    #[test]
    fn no_coopmat_import_is_unchanged() {
        let mut w = vec![MAGIC, 0x0001_0300, 0, 2, 0];
        w.extend_from_slice(&instr(OP_FUNCTION, &[1, 20, 0, 100]));
        w.extend_from_slice(&instr(OP_FUNCTION_END, &[]));
        let before = w.clone();
        assert_eq!(fix_coopmat_calls(w), before);
    }

    #[test]
    fn non_spirv_untouched_coopmat() {
        let junk = vec![1u32, 2, 3];
        assert_eq!(fix_coopmat_calls(junk.clone()), junk);
    }

    /// `__spirv_CompositeConstruct` (the coopmat zero-accumulator seed) rewrites like the other three
    /// builtins, but has no trailing bitmask operand: its single scalar arg passes through unchanged.
    #[test]
    fn rewrites_composite_construct_to_real_instruction() {
        // ids: 1=dummy result type, 10=ConstructFn, 20=main, 30=construct-result.
        let mut w = vec![MAGIC, 0x0001_0300, 0, 40, 0];
        w.extend_from_slice(&instr(OP_CAPABILITY, &[5 /* Linkage */]));
        w.extend_from_slice(&decorate_import(10, "__spirv_CompositeConstruct"));
        w.extend_from_slice(&name_instr(10, "ctor"));
        w.extend_from_slice(&instr(OP_FUNCTION, &[1, 10, 0, 100]));
        w.extend_from_slice(&instr(OP_FUNCTION_END, &[]));
        w.extend_from_slice(&instr(OP_FUNCTION, &[1, 20, 0, 101]));
        w.extend_from_slice(&instr(OP_FUNCTION_CALL, &[1, 30, 10, 200 /* float 0.0 */]));
        w.extend_from_slice(&instr(253 /* OpReturn */, &[]));
        w.extend_from_slice(&instr(OP_FUNCTION_END, &[]));

        let after = fix_coopmat_calls(w);
        assert!(find_opcode(&after, OP_FUNCTION_CALL).is_empty());
        let ctor = find_opcode(&after, 80 /* OpCompositeConstruct */);
        assert_eq!(
            ctor.len(),
            1,
            "expected exactly one real CompositeConstruct"
        );
        let ci = ctor[0];
        assert_eq!(
            (after[ci] >> 16) as usize,
            4,
            "ResultType, ResultId, 1 arg -> wc 4"
        );
        assert_eq!(&after[ci + 1..ci + 4], &[1, 30, 200]);
        // Linkage dropped (nothing else needs it), same as the other three builtins.
        let capabilities: Vec<u32> = find_opcode(&after, OP_CAPABILITY)
            .into_iter()
            .map(|i| after[i + 1])
            .collect();
        assert_eq!(capabilities, vec![5345], "expected only VulkanMemoryModel");
    }

    /// End to end: compile the builtin-call IR through the real `llc` with poot's Vulkan triple, confirm
    /// `spirv-val --target-env vulkan1.3` rejects the raw output (the Import/Linkage shape) and accepts the
    /// output of `fix_coopmat_calls`. Needs `llc` and `spirv-val` on PATH (`nix develop`); skips otherwise,
    /// as `crates/poot-codegen/tests/llc.rs`'s `add_spirv_validates` does. The `.ll` fixture matches
    /// `specs/113-vulkan-coopmat/probe-fixtures/02_vulkan_env_load_muladd_store_IMPORT_LINKAGE.ll`.
    #[test]
    fn real_llc_output_before_fails_validation_after_fix_passes() {
        fn have(bin: &str) -> bool {
            std::process::Command::new(bin)
                .arg("--version")
                .output()
                .is_ok()
        }
        if !have("llc") || !have("spirv-val") {
            eprintln!("llc/spirv-val not on PATH; skipping coopmat route-1 round-trip probe");
            return;
        }

        let dir = std::env::temp_dir().join("poot-codegen-coopmat-probe");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("coopmat_vulkan_env.ll");
        let spv = dir.join("coopmat_vulkan_env.spv");
        std::fs::write(&ll, COOPMAT_VULKAN_ENV_LL).unwrap();

        let c = std::process::Command::new("llc")
            .args([
                "-O0",
                "-filetype=obj",
                "-mtriple=spirv-unknown-vulkan1.3-compute",
                "--spirv-ext=+SPV_KHR_cooperative_matrix",
            ])
            .arg(&ll)
            .arg("-o")
            .arg(&spv)
            .output()
            .unwrap();
        assert!(
            c.status.success(),
            "llc failed:\n{}",
            String::from_utf8_lossy(&c.stderr)
        );

        let before_bytes = std::fs::read(&spv).unwrap();
        let before_val = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&spv)
            .output()
            .unwrap();
        assert!(
            !before_val.status.success(),
            "expected raw llc output to FAIL spirv-val (Linkage capability); it passed instead - \
             the premise this rewrite is built on no longer holds, re-probe before trusting the fix"
        );
        assert!(
            String::from_utf8_lossy(&before_val.stderr).contains("Linkage"),
            "expected the Linkage-capability error specifically, got:\n{}",
            String::from_utf8_lossy(&before_val.stderr)
        );

        let before_words: Vec<u32> = before_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let after_words = fix_coopmat_calls(before_words);
        let after_path = dir.join("coopmat_vulkan_env.after.spv");
        let mut after_bytes = Vec::with_capacity(after_words.len() * 4);
        for w in after_words {
            after_bytes.extend_from_slice(&w.to_le_bytes());
        }
        std::fs::write(&after_path, after_bytes).unwrap();

        let after_val = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&after_path)
            .output()
            .unwrap();
        assert!(
            after_val.status.success(),
            "spirv-val failed on the rewritten module:\n{}",
            String::from_utf8_lossy(&after_val.stderr)
        );
    }

    /// The same real-toolchain round trip for a single-tile F16xF16->F32 matmul (RADV coopmat config 13:
    /// Subgroup scope, 16x16x16) with a `__spirv_CompositeConstruct`-seeded zero accumulator instead of a
    /// loaded C: the IR `poot_codegen::emit`'s SPIR-V WMMA arms emit (`emit_wmma_load_coopmat` /
    /// `emit_wmma_mma_coopmat` / `emit_wmma_store_coopmat`). It hits the same Import-stub gap as the other
    /// three builtins.
    #[test]
    fn real_llc_coopmat_tile_with_zero_accumulator_validates_after_fix() {
        fn have(bin: &str) -> bool {
            std::process::Command::new(bin)
                .arg("--version")
                .output()
                .is_ok()
        }
        if !have("llc") || !have("spirv-val") {
            eprintln!("llc/spirv-val not on PATH; skipping coopmat tile round-trip probe");
            return;
        }

        let dir = std::env::temp_dir().join("poot-codegen-coopmat-tile-probe");
        std::fs::create_dir_all(&dir).unwrap();
        let ll = dir.join("coopmat_tile.ll");
        let spv = dir.join("coopmat_tile.spv");
        std::fs::write(&ll, COOPMAT_TILE_LL).unwrap();

        let c = std::process::Command::new("llc")
            .args([
                "-O0",
                "-filetype=obj",
                "-mtriple=spirv-unknown-vulkan1.3-compute",
                "--spirv-ext=+SPV_KHR_cooperative_matrix",
            ])
            .arg(&ll)
            .arg("-o")
            .arg(&spv)
            .output()
            .unwrap();
        assert!(
            c.status.success(),
            "llc failed:\n{}",
            String::from_utf8_lossy(&c.stderr)
        );

        let before_bytes = std::fs::read(&spv).unwrap();
        let before_val = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&spv)
            .output()
            .unwrap();
        assert!(
            !before_val.status.success(),
            "expected raw llc output to FAIL spirv-val (Linkage capability); it passed instead - \
             re-probe before trusting the fix"
        );
        assert!(
            String::from_utf8_lossy(&before_val.stderr).contains("Linkage"),
            "expected the Linkage-capability error specifically, got:\n{}",
            String::from_utf8_lossy(&before_val.stderr)
        );

        let before_words: Vec<u32> = before_bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let after_words = fix_coopmat_calls(before_words);
        let after_path = dir.join("coopmat_tile.after.spv");
        let mut after_bytes = Vec::with_capacity(after_words.len() * 4);
        for w in after_words {
            after_bytes.extend_from_slice(&w.to_le_bytes());
        }
        std::fs::write(&after_path, after_bytes).unwrap();

        let after_val = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.3"])
            .arg(&after_path)
            .output()
            .unwrap();
        assert!(
            after_val.status.success(),
            "spirv-val failed on the rewritten module:\n{}",
            String::from_utf8_lossy(&after_val.stderr)
        );
    }

    const COOPMAT_TILE_LL: &str = r#"target triple = "spirv-unknown-vulkan1.3-compute"

@a_lds = internal addrspace(3) global [512 x i8] zeroinitializer, align 16
@b_lds = internal addrspace(3) global [512 x i8] zeroinitializer, align 16
@d_lds = internal addrspace(3) global [1024 x i8] zeroinitializer, align 16

define void @main() #0 {
entry:
  %aptr = getelementptr [512 x i8], ptr addrspace(3) @a_lds, i32 0, i32 0
  %bptr = getelementptr [512 x i8], ptr addrspace(3) @b_lds, i32 0, i32 0
  %dptr = getelementptr [1024 x i8], ptr addrspace(3) @d_lds, i32 0, i32 0

  %ma = tail call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3) %aptr, i32 0, i64 16, i32 0)
  %mb = tail call spir_func target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3) %bptr, i32 0, i64 16, i32 0)
  %mc = tail call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z27__spirv_CompositeConstruct(float 0.000000e+00)
  %md = tail call spir_func target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) %ma, target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1) %mb, target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %mc, i32 0)
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3) %dptr, target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) %md, i32 0, i64 16, i32 0)
  ret void
}

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z27__spirv_CompositeConstruct(float)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 0), target("spirv.CooperativeMatrixKHR", half, 3, 16, 16, 1), target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2), i32)
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", float, 3, 16, 16, 2), i32, i64, i32)

attributes #0 = { "hlsl.numthreads"="32,1,1" "hlsl.shader"="compute" }
"#;

    const COOPMAT_VULKAN_ENV_LL: &str = r#"target triple = "spirv-unknown-vulkan1.3-compute"

@a_lds = internal addrspace(3) global [768 x i8] zeroinitializer, align 16
@b_lds = internal addrspace(3) global [768 x i8] zeroinitializer, align 16
@c_lds = internal addrspace(3) global [576 x i8] zeroinitializer, align 16
@d_lds = internal addrspace(3) global [576 x i8] zeroinitializer, align 16

define void @main() #0 {
entry:
  %aptr = getelementptr [768 x i8], ptr addrspace(3) @a_lds, i32 0, i32 0
  %bptr = getelementptr [768 x i8], ptr addrspace(3) @b_lds, i32 0, i32 0
  %cptr = getelementptr [576 x i8], ptr addrspace(3) @c_lds, i32 0, i32 0
  %dptr = getelementptr [576 x i8], ptr addrspace(3) @d_lds, i32 0, i32 0

  %m2 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3) %aptr, i32 0, i64 12, i32 1)
  %m3 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3) %bptr, i32 0, i64 0)
  %m1 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z32__spirv_CooperativeMatrixLoadKHR_3(ptr addrspace(3) %cptr, i32 0, i64 12, i32 0)
  %m5 = tail call spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) %m2, target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) %m3, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m1, i32 12)
  tail call spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3) %dptr, target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) %m5, i32 0, i64 12, i32 1)
  ret void
}

declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0) @_Z32__spirv_CooperativeMatrixLoadKHR_1(ptr addrspace(3), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1) @_Z32__spirv_CooperativeMatrixLoadKHR_2(ptr addrspace(3), i32, i64)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z32__spirv_CooperativeMatrixLoadKHR_3(ptr addrspace(3), i32, i64, i32)
declare dso_local spir_func target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2) @_Z34__spirv_CooperativeMatrixMulAddKHR(target("spirv.CooperativeMatrixKHR", i32, 3, 12, 48, 0), target("spirv.CooperativeMatrixKHR", i32, 3, 48, 12, 1), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32)
declare dso_local spir_func void @_Z33__spirv_CooperativeMatrixStoreKHR(ptr addrspace(3), target("spirv.CooperativeMatrixKHR", i32, 3, 12, 12, 2), i32, i64, i32)

attributes #0 = { "hlsl.numthreads"="64,1,1" "hlsl.shader"="compute" }
"#;
}
