import React, { useState } from 'react';

type StepDef = {
  title: string;
  artifact: string;
  code: string;
  lang: 'rust' | 'json' | 'llvm' | 'ptx' | 'text';
  note: string;
};

const STEPS: StepDef[] = [
  {
    title: '1. Rust source',
    artifact: 'add.rs (a #[kernel] fn, simplified)',
    lang: 'rust',
    code: `#[kernel]
pub fn elementwise_add(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
) {
    let i = thread_index();
    if i < out.len() {
        out[i] = a[i] + b[i];
    }
}`,
    note: 'Full Rust: type-checked, borrow-checked. Testable on CPU under cfg(test).',
  },
  {
    title: '2. Stable MIR import',
    artifact: 'pootc -> kernel IR (in memory)',
    lang: 'text',
    code: `Body "__poot_kernel_elementwise_add" {
  params: a: &[f32], b: &[f32], out: &mut [f32]
  blocks: [
    BB0: ThreadIndexCall { dest: i, dim: X } -> BB1
    BB1: SwitchInt(i < out.len()) -> BB2 | BB3
    BB2: {
      t0 = a[i]
      t1 = b[i]
      out[i] = t0 + t1
    } Goto BB3
    BB3: Return
  ]
}`,
    note: 'pootc reads Stable MIR (rustc_public). Unsupported constructs -> named diagnostic.',
  },
  {
    title: '3. .kir.json asset',
    artifact: '<crate>/assets/<kernel>.kir.json (shape only)',
    lang: 'json',
    code: `{
  "name": "__poot_kernel_elementwise_add",
  "param_count": 3,
  "locals": [ {"ty":"Unit","mutable":true},
              {"ty":{"Ref":{"mutable":false,"pointee":{"Slice":"F32"}}},"mutable":false},
              ... ],
  "blocks": [ ... ]
}`,
    note: 'Committed to the repo. Drift-guarded: test fails if source changes without regen.',
  },
  {
    title: '4. LLVM IR (SPIR-V, NVPTX, or AMDGCN)',
    artifact: 'poot-codegen emits inline',
    lang: 'llvm',
    code: `; SPIR-V target (illustrative):
define spir_kernel void @elementwise_add(
    ptr addrspace(1) %a, i64 %a_len,
    ptr addrspace(1) %b, i64 %b_len,
    ptr addrspace(1) %out, i64 %out_len) {
entry:
  %i = call i64 @llvm.spv.thread.id(i32 0)
  %cmp = icmp ult i64 %i, %out_len
  br i1 %cmp, %live, %dead
live:
  %ap = getelementptr float, ptr as(1) %a, i64 %i
  %a_val = load float, ptr as(1) %ap
  ...
  ret void
}`,
    note: 'Target-specific address spaces and intrinsics. Same kernel IR, one emitter with per-target choices.',
  },
  {
    title: '5. GPU binary',
    artifact: '.spv, .ptx, or .hsaco',
    lang: 'ptx',
    code: `; nvptx64 output (illustrative excerpt):
.visible .entry elementwise_add(
    .param .u64 a, .param .u64 a_len,
    .param .u64 b, .param .u64 b_len,
    .param .u64 out, .param .u64 out_len)
{
    .reg .u64  %rd<8>;
    .reg .f32  %f<4>;
    .reg .pred %p<2>;
    mov.u32    %r0, %tid.x;
    cvt.u64.u32 %rd0, %r0;
    ld.param.u64 %rd1, [out_len];
    setp.lt.u64  %p0, %rd0, %rd1;
    @!%p0 bra  $done;
    ...
$done: ret;
}`,
    note: 'llc produces SPIR-V and PTX; the ROCm-flavored clang links a loadable HSACO. Loaded by wgpu (SPIR-V), cudarc (PTX), or HSA (HSACO), and dispatched per Plan entry.',
  },
];

export default function KernelPipelineDiagram(): React.ReactElement {
  const [step, setStep] = useState(0);
  const s = STEPS[step];

  return (
    <div style={{ margin: '1.5rem 0', border: '1px solid var(--ifm-color-emphasis-200)', borderRadius: 8, overflow: 'hidden' }}>
      {/* Step tabs */}
      <div style={{ display: 'flex', borderBottom: '1px solid var(--ifm-color-emphasis-200)', overflowX: 'auto' }}>
        {STEPS.map((st, i) => (
          <button key={i} onClick={() => setStep(i)}
            style={{
              flex: '0 0 auto', padding: '8px 14px', border: 'none',
              borderBottom: step === i ? '2.5px solid var(--ifm-color-primary)' : '2.5px solid transparent',
              background: step === i ? 'color-mix(in srgb, var(--ifm-color-primary) 8%, transparent)' : 'transparent',
              fontWeight: step === i ? 700 : 400, fontSize: '0.8rem',
              color: step === i ? 'var(--ifm-color-primary)' : 'var(--ifm-color-emphasis-700)',
              cursor: 'pointer', whiteSpace: 'nowrap',
            }}
          >
            {st.title}
          </button>
        ))}
      </div>

      <div style={{ padding: '1rem 1.25rem', background: 'var(--ifm-background-surface-color)' }}>
        <div style={{ fontSize: '0.78rem', color: 'var(--ifm-color-emphasis-600)', marginBottom: '0.5rem', fontFamily: 'monospace' }}>
          {s.artifact}
        </div>
        <pre style={{
          background: 'var(--ifm-color-emphasis-100)', borderRadius: 6, padding: '0.75rem 1rem',
          fontSize: '0.75rem', overflow: 'auto', margin: '0 0 0.75rem 0', lineHeight: 1.5,
          border: '1px solid var(--ifm-color-emphasis-200)', maxHeight: 260,
        }}>
          <code>{s.code}</code>
        </pre>
        <div style={{ fontSize: '0.82rem', color: 'var(--ifm-color-emphasis-700)', background: 'color-mix(in srgb, var(--ifm-color-primary) 6%, transparent)', borderLeft: '3px solid var(--ifm-color-primary)', padding: '0.5rem 0.75rem', borderRadius: '0 4px 4px 0' }}>
          {s.note}
        </div>
      </div>

      {/* Progress dots */}
      <div style={{ display: 'flex', justifyContent: 'center', gap: 6, padding: '0.6rem', borderTop: '1px solid var(--ifm-color-emphasis-100)' }}>
        {STEPS.map((_, i) => (
          <button key={i} onClick={() => setStep(i)}
            style={{
              width: 10, height: 10, borderRadius: '50%', border: 'none', cursor: 'pointer', padding: 0,
              background: i === step ? 'var(--ifm-color-primary)' : 'var(--ifm-color-emphasis-300)',
            }}
          />
        ))}
      </div>
    </div>
  );
}
