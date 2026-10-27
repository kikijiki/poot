// The shape of website/src/data/performance.json, which benchmarks/harness/perf_page.py derives.

export type Stat = { median: number; min: number; max: number; n: number };

// A measured quantity of a point. tpot, first-token latency and end-to-end time are derived from the decode and prefill rates and
// appear in tables and tooltips only.
export type PointMetric = "decode_tok_s" | "prefill_tok_s" | "ttft_ms" | "tpot_ms" | "e2e_ms" | "peak_vram_bytes" | "peak_rss_bytes" | "load_vram_bytes" | "load_rss_bytes" | "energy_j";

// A point whose repeats do not verify but is committed on the list of known-unverified points: its numbers are still measured.
export type Unverified = { pair: string; failed_repeats: number; reason: string };

// The lowest and highest of the per-host medians behind a point.
export type Spread = { min: number; max: number };

// One framework on one backend, at one input length (prompt tokens). The value is the median of the per-host medians over the
// verified cells of the hosts that measured it (every cell when none verifies, and then `verified` is false). A superseded set
// is a point of its own with one host.
export type Point = {
  // The run sets behind the point, comma separated.
  set: string;
  date: string;
  model: string;
  framework: string;
  backend: string;
  series: string;
  precision: string | null;
  // Input tokens: the prompt length of a single-point run, or a context length of a sweep.
  isl: number;
  repeats: number;
  // The hosts whose cells the value is taken over, and how many cells of the class were verified or not.
  hosts: number;
  // The labels of those hosts.
  host_labels: string[];
  cells: { verified: number; unverified: number };
  spread: Partial<Record<PointMetric, Spread>>;
  latest: boolean;
  verified: boolean;
  // What the memory metrics measure: "inference" is the peak after the model is loaded and warmed up (the load peak is stated
  // beside it); "includes_load" is the lifetime peak of a run whose load boundary could not be found.
  memory_scope: "inference" | "includes_load";
  // The noise band of the headline metric as a fraction of its median.
  band: number;
  metrics: Partial<Record<PointMetric, Stat>>;
  identity: Record<string, string>;
  // Memory and energy of a sweep are peaks over every context length, shown on its longest.
  over_sweep?: boolean;
  unverified?: Unverified;
};

// One GPU model and its points.
export type Hardware = {
  id: string;
  label: string;
  points: Point[];
};

export type PerformanceData = {
  schema: number;
  policy: { min_repeats: number; max_uncertainty: number };
  hardware: Hardware[];
};
