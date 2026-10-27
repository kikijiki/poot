import type { PointMetric } from "./types";

export type Unit = { suffix: string; scale: number; digits: number };

export const TOK_S: Unit = { suffix: "tok/s", scale: 1, digits: 1 };
export const MS: Unit = { suffix: "ms", scale: 1, digits: 1 };
export const GIB: Unit = { suffix: "GiB", scale: 1 / 2 ** 30, digits: 2 };
export const JOULE: Unit = { suffix: "J", scale: 1, digits: 0 };

// The charts of the page, in order. A chart is drawn only when every hardware entry has data for its metric (see commonMetrics).
// A `wholeRun` metric (memory peaks, energy) is one value per run, not per input length: it is one bar per series, with no input
// axis. End-to-end time exists only for the 128-token runs (see perf_page.py).
export const CHART_METRICS: { key: PointMetric; title: string; wholeRun?: true }[] = [
  { key: "decode_tok_s", title: "Decode tok/s" },
  { key: "prefill_tok_s", title: "Prefill tok/s" },
  { key: "ttft_ms", title: "Time to first token" },
  { key: "e2e_ms", title: "End-to-end (128 tokens)" },
  { key: "peak_rss_bytes", title: "Host memory at inference", wholeRun: true },
  { key: "peak_vram_bytes", title: "GPU memory at inference", wholeRun: true },
  { key: "energy_j", title: "Energy (whole run)", wholeRun: true },
];

// The whole-run memory metrics: peaks measured after the load and warmup.
export const MEMORY_METRICS: PointMetric[] = ["peak_rss_bytes", "peak_vram_bytes"];

export type MetricInfo = { key: PointMetric; label: string; unit: Unit; higherIsBetter: boolean };

// Every measured quantity, for table columns and tooltips.
export const POINT_METRICS: MetricInfo[] = [
  { key: "decode_tok_s", label: "Decode", unit: TOK_S, higherIsBetter: true },
  { key: "tpot_ms", label: "Time per output token", unit: MS, higherIsBetter: false },
  { key: "prefill_tok_s", label: "Prefill", unit: TOK_S, higherIsBetter: true },
  { key: "ttft_ms", label: "First token", unit: MS, higherIsBetter: false },
  { key: "e2e_ms", label: "End to end", unit: MS, higherIsBetter: false },
  { key: "peak_vram_bytes", label: "GPU memory at inference", unit: GIB, higherIsBetter: false },
  { key: "peak_rss_bytes", label: "Host memory at inference", unit: GIB, higherIsBetter: false },
  { key: "load_vram_bytes", label: "GPU memory while loading", unit: GIB, higherIsBetter: false },
  { key: "load_rss_bytes", label: "Host memory while loading", unit: GIB, higherIsBetter: false },
  { key: "energy_j", label: "Energy, whole run", unit: JOULE, higherIsBetter: false },
];

export function metricInfo(key: PointMetric): MetricInfo {
  return POINT_METRICS.find((m) => m.key === key) as MetricInfo;
}

export function num(value: number, unit: Unit): string {
  const v = value * unit.scale;
  // Large values need no decimals; small ones keep their digits.
  return v >= 1000 ? Math.round(v).toLocaleString("en-US") : v.toFixed(unit.digits);
}

export function range(stat: { min: number; max: number }, unit: Unit): string {
  return `${num(stat.min, unit)} to ${num(stat.max, unit)}`;
}

export function hostCount(hosts: number): string {
  return `${hosts} ${hosts === 1 ? "host" : "hosts"}`;
}

// A series is a framework on one backend. Its colour is fixed by that pair alone, the same on every hardware, model and column,
// never by position or by which series are on screen. Slot n is --series-n of performance.module.css.
export const SERIES_SLOTS: Readonly<Record<string, number>> = {
  "poot ptx": 1,
  "poot rocm": 2,
  "poot vulkan": 3,
  "poot wgpu": 4,
  "llamacpp cuda": 5,
  "llamacpp vulkan": 6,
  "candle cuda": 7,
  "transformers cuda": 8,
};

// The slot of a pair that has no entry above: neutral, so a new pair is visibly unassigned instead of borrowing another's colour.
const UNASSIGNED_SLOT = 9;

export function seriesId(framework: string, backend: string): string {
  return `${framework} ${backend}`;
}

export function seriesSlot(id: string): number {
  return SERIES_SLOTS[id] ?? UNASSIGNED_SLOT;
}

export function seriesColor(id: string): string {
  return `var(--series-${seriesSlot(id)})`;
}

export function shortSha(sha?: string): string {
  return sha ? sha.slice(0, 8) : "";
}
