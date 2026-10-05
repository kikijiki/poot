import React from 'react';

type Status = 'yes' | 'no' | 'partial' | 'na';
type Row = { feature: string; wgpu: Status; rocm: Status; ptx: Status; note?: string };

// Kernel- and dispatch-level capability, not the serving-level feature matrix (models,
// quantization formats, LoRA, tensor parallelism, ...) - see /reference/feature-matrix for that.
const ROWS: Row[] = [
  { feature: 'Elementwise kernels', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Reduce kernels', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'GEMV (decode matmul)', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Tiled GEMM (prefill matmul)', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Dequant-in-matmul (GPTQ/AWQ/GGUF Q8_0/Q4_0/K-quants)', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Flash attention decode', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Flash attention prefill', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Paged KV cache', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Fixed-capacity KV + masking', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Workgroup-local memory (LDS)', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  { feature: 'Global atomics', wgpu: 'yes', rocm: 'yes', ptx: 'yes' },
  {
    feature: 'Device-native graph capture/replay',
    wgpu: 'no',
    rocm: 'yes',
    ptx: 'yes',
    note: 'ROCm: raw HSA AQL packet replay. PTX: CUDA graphs. wgpu has no equivalent API.',
  },
  {
    feature: 'Cached re-encode (per-token)',
    wgpu: 'yes',
    rocm: 'no',
    ptx: 'no',
    note: "wgpu's own approximation of graph replay",
  },
  {
    feature: 'Cooperative / persistent grid',
    wgpu: 'no',
    rocm: 'yes',
    ptx: 'yes',
    note: 'wgpu has no grid-wide sync primitive',
  },
  {
    feature: 'Megakernel (on-device token loop)',
    wgpu: 'no',
    rocm: 'no',
    ptx: 'no',
    note: 'Removed. Planned as a recorder behind the one compiled program on PTX and ROCm. wgpu has no grid-wide sync primitive.',
  },
  {
    feature: 'bf16 compute',
    wgpu: 'no',
    rocm: 'yes',
    ptx: 'yes',
    note: 'Opt-in. wgpu: VK_KHR_shader_bfloat16 absent on most iGPUs, so compute stays f32.',
  },
  {
    feature: 'Tensor-core matmul (WMMA)',
    wgpu: 'no',
    rocm: 'yes',
    ptx: 'yes',
    note: 'ROCm: RDNA WMMA, verified on gfx1151. wgpu has a narrow experimental SPIR-V coopmat primitive no model uses yet.',
  },
];

const STATUS_LABEL: Record<Status, string> = {
  yes: 'Yes',
  no: 'No',
  partial: 'Partial',
  na: 'N/A',
};

function StatusBadge({ status }: { status: Status }): React.ReactElement {
  return <span className={`matrix-status matrix-status--${status}`}>{STATUS_LABEL[status]}</span>;
}

export default function BackendTable(): React.ReactElement {
  return (
    <table className="matrix-table">
      <thead>
        <tr>
          <th>Capability</th>
          <th>wgpu / SPIR-V</th>
          <th>ROCm / AMD</th>
          <th>PTX / NVIDIA</th>
          <th>Notes</th>
        </tr>
      </thead>
      <tbody>
        {ROWS.map((row) => (
          <tr key={row.feature}>
            <td>{row.feature}</td>
            <td className="matrix-cell-status">
              <StatusBadge status={row.wgpu} />
            </td>
            <td className="matrix-cell-status">
              <StatusBadge status={row.rocm} />
            </td>
            <td className="matrix-cell-status">
              <StatusBadge status={row.ptx} />
            </td>
            <td className="matrix-cell-notes">{row.note ?? ''}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
