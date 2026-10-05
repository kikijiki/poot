// Shared types and framework display config for the benchmark dashboards.

export type Framework = 'poot' | 'candle' | 'transformers' | 'vllm' | 'llamacpp';

export type Pct = {p50: number; p90: number; p99: number};

export type CurvePoint = {
  isl: number;
  osl: number;
  prompt_tokens?: number;
  iters?: number;
  ttft_ms: Pct;
  tpot_ms: Pct;
  itl_ms: Pct;
  decode_tok_s: number;
};

export type ResultRow = {
  framework: Framework;
  model: string;
  precision: string;
  scenario: string;
  status: 'ok' | 'unsupported' | 'error';
  reason?: string;
  caveats?: string[];
  // provenance: which binary ran on which backend and device (build_sha is set by runners built from
  // this repository, so it is absent on other engines' rows and on rows whose runner failed).
  build_sha?: string;
  binary_sha256?: string;
  backend?: string;
  device?: string | null;
  // single-point
  prompt_tokens?: number;
  gen_tokens?: number;
  ttft_ms?: number;
  tpot_ms?: number;
  e2e_ms?: number;
  decode_tok_s?: number;
  iterations?: number;
  ttft_ms_stdev?: number;
  decode_tok_s_stdev?: number;
  // decode-curve
  osl?: number;
  isl?: number;
  curve?: CurvePoint[];
  // memory
  peak_vram_bytes?: number;
  peak_rss_bytes?: number;
  vram_baseline_bytes?: number;
  self_reported_vram_bytes?: number;
  // resource utilization (sampled per cell by the harness)
  gpu_util_mean_pct?: number;
  gpu_util_peak_pct?: number;
  cpu_util_mean_pct?: number;
  cpu_util_peak_pct?: number;
  // GPU power/energy (NVIDIA-only; null on shared-memory iGPU backends).
  peak_power_w?: number;
  energy_j?: number;
  energy_wh?: number;
  // Per-cell time series sampled ~1 Hz; t_s is seconds since cell start. Spans all ISLs of a
  // decode-curve cell with no per-ISL markers.
  util_series?: UtilSample[];
};

// [t_s, gpu_pct, cpu_pct, vram_bytes, rss_bytes, power_w]; all but t_s may be null.
export type UtilSample = [
  number,
  number | null,
  number | null,
  number | null,
  number | null,
  number | null,
];

export type RunEnv = {
  captured_at_utc: string | null;
  gpu_name: string | null;
  gpu_driver: string | null;
  gpu_memory_total: string | null;
  cuda_version: string | null;
  host_kernel: string | null;
  cpu: string | null;
  harness_version: string | null;
};

export type Run = {
  id: string;
  source: 'results' | 'sample' | 'fixture';
  sample: boolean;
  sampleLabel?: string;
  env: RunEnv;
  results: ResultRow[];
};

export type BenchData = {
  generated_at: string;
  has_real_data: boolean;
  real_run_count: number;
  run_count: number;
  runs: Run[];
};

// poot gets the brand green and is drawn thicker and on top.
export const FRAMEWORK_COLORS: Record<Framework, string> = {
  poot: '#25c2a0',
  candle: '#e8833a',
  transformers: '#a855f7',
  vllm: '#3b82f6',
  llamacpp: '#ef4444',
};

export const FRAMEWORK_LABELS: Record<Framework, string> = {
  poot: 'poot',
  candle: 'candle',
  transformers: 'transformers',
  vllm: 'vLLM',
  llamacpp: 'llama.cpp',
};

export const FRAMEWORK_ORDER: Framework[] = [
  'poot',
  'candle',
  'transformers',
  'vllm',
  'llamacpp',
];

export function frameworkColor(f: string): string {
  return FRAMEWORK_COLORS[f as Framework] ?? '#888888';
}

export function frameworkLabel(f: string): string {
  return FRAMEWORK_LABELS[f as Framework] ?? f;
}

export function bytesToGiB(bytes: number | undefined | null): number | null {
  if (bytes == null) return null;
  return bytes / 2 ** 30;
}

export function shortDate(iso: string | null): string {
  if (!iso) return '';
  return iso.slice(0, 10);
}
