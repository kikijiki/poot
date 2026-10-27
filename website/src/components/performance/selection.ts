// What the Performance page draws: a selection of hardware and one model (every metric of it is charted), resolved
// against the data (a pick the data cannot satisfy falls back), offered only the options that exist, and kept in the URL
// query. The data is one workload family: rates over input tokens. Pure functions of the data, so the filtering and the
// aggregation maths are tested without a browser.
import { CHART_METRICS, MEMORY_METRICS, num, seriesId, seriesSlot, type Unit } from "./format";
import type { Hardware, PerformanceData, Point, PointMetric, Stat } from "./types";

// One column of the page: a hardware and a model.
export type Side = { hw: string; model: string };

// Column A, and column B when comparison mode is on.
export type Selection = Side & { b?: Side };

// What a URL or a control asks for: any part may be missing or unknown.
export type RawSelection = Partial<Side> & { b?: Partial<Side> };

export const pointSeries = (p: Point) => seriesId(p.framework, p.backend);
const bySeries = (a: string, b: string) => seriesSlot(a) - seriesSlot(b) || a.localeCompare(b);

export function findHardware(report: PerformanceData, id: string): Hardware | undefined {
  return report.hardware.find((h) => h.id === id);
}

// The page opens on the AMD Radeon 8060S (Strix Halo), the box poot is developed on; any other first entry if it is absent.
const DEFAULT_HARDWARE = "amd-radeon-8060s-graphics";

// The size of a model in billions of parameters, read from the `<n>b` token of its id ("qwen3-0.6b" is 0.6, "qwen3.8-27b-q4" is 27); the data
// carries no parameter count. A model whose id has no such token has an unknown size and counts as the largest.
export function modelSize(model: string): number {
  const size = /(?:^|[-_])(\d+(?:\.\d+)?)b(?=$|[-_])/.exec(model);
  return size ? Number(size[1]) : Number.POSITIVE_INFINITY;
}

// The model the page opens on: the smallest the hardware has (ties and unknown sizes by name), so the first view loads and compares the
// cheapest rows.
export function defaultModel(models: string[]): string {
  return [...models].sort((a, b) => (modelSize(a) === modelSize(b) ? a.localeCompare(b) : modelSize(a) < modelSize(b) ? -1 : 1))[0] ?? "";
}

export function defaultHardware(report: PerformanceData): Hardware {
  return findHardware(report, DEFAULT_HARDWARE) ?? report.hardware[0];
}

// The points of the selected model. Charts draw the newest set of each series and input length only.
export function selectPoints(hw: Hardware, sel: Side): Point[] {
  return hw.points.filter((p) => p.latest && p.model === sel.model);
}

export type Options = {
  hardware: { id: string; label: string }[];
  models: string[];
};

export function options(report: PerformanceData, sel: Pick<Side, "hw">): Options {
  const hw = findHardware(report, sel.hw) as Hardware;
  return {
    hardware: report.hardware.map((h) => ({ id: h.id, label: h.label })),
    models: [...new Set(hw.points.filter((p) => p.latest).map((p) => p.model))].sort(),
  };
}

// The metrics the page may chart: those some newest point measured on every hardware entry. A metric one hardware lacks is drawn on
// none, so the columns of a comparison always hold the same charts; it appears by itself once that hardware has the data.
export function commonMetrics(report: PerformanceData): PointMetric[] {
  return CHART_METRICS.map((m) => m.key).filter((key) => report.hardware.every((h) => h.points.some((p) => p.latest && p.metrics[key])));
}

// The charts of the selected model, in the page's order: every common metric some point of it measured (no empty card).
export function chartMetrics(report: PerformanceData, points: Point[]): PointMetric[] {
  const common = commonMetrics(report);
  return CHART_METRICS.map((m) => m.key).filter((key) => common.includes(key) && points.some((p) => p.metrics[key]));
}

function resolveSide(report: PerformanceData, raw: Partial<Side>, fallbackHw: Hardware): Side {
  const hw = findHardware(report, raw.hw ?? "") ?? fallbackHw;
  const models = options(report, { hw: hw.id }).models;
  return { hw: hw.id, model: models.find((m) => m === raw.model) ?? defaultModel(models) };
}

// The second column when comparison is switched on: the same model on another hardware if one has it, else the smallest other model of
// the same hardware.
export function defaultSecond(report: PerformanceData, a: Side): Side {
  const other = report.hardware.find((h) => h.id !== a.hw && options(report, { hw: h.id }).models.includes(a.model));
  if (other) return { hw: other.id, model: a.model };
  const another = defaultModel(options(report, { hw: a.hw }).models.filter((m) => m !== a.model));
  return { hw: a.hw, model: another || a.model };
}

// A pick the data cannot satisfy is dropped: an unknown hardware is replaced by the default, an unknown model by the
// smallest model of the hardware. A model stays when the new hardware has it.
export function resolve(report: PerformanceData, raw: RawSelection): Selection {
  const a = resolveSide(report, raw, defaultHardware(report));
  if (!raw.b) return a;
  return { ...a, b: resolveSide(report, raw.b, findHardware(report, defaultSecond(report, a).hw) as Hardware) };
}

export function toQuery(sel: Selection): string {
  const q = new URLSearchParams({ hw: sel.hw, model: sel.model });
  if (sel.b) {
    q.set("compare", "1");
    q.set("hw2", sel.b.hw);
    q.set("model2", sel.b.model);
  }
  return q.toString();
}

export function fromQuery(query: string): RawSelection {
  const q = new URLSearchParams(query);
  return {
    hw: q.get("hw") ?? undefined,
    model: q.get("model") ?? undefined,
    ...(q.get("compare") === "1" ? { b: { hw: q.get("hw2") ?? undefined, model: q.get("model2") ?? undefined } } : {}),
  };
}

export type Line = { id: string; label: string; verified: boolean; points: { isl: number; stat: Stat; point: Point }[] };

// One line per framework and backend over the input tokens, for one point metric. A line is verified when all its points are.
export function seriesLines(points: Point[], metric: PointMetric): Line[] {
  const lines = new Map<string, Line>();
  for (const p of [...points].sort((a, b) => a.isl - b.isl)) {
    const stat = p.metrics[metric];
    if (!stat) continue;
    const id = pointSeries(p);
    const line = lines.get(id) ?? { id, label: p.series, verified: true, points: [] };
    line.points.push({ isl: p.isl, stat, point: p });
    line.verified &&= p.verified;
    lines.set(id, line);
  }
  return [...lines.values()].sort((a, b) => bySeries(a.id, b.id));
}

// One bar per framework and backend for a whole-run metric: the value of the point on its longest input length that carries it
// (the harness puts a run's peaks and energy there).
export function runBars(points: Point[], metric: PointMetric): Bar[] {
  const bars = new Map<string, Bar>();
  for (const p of [...points].sort((a, b) => a.isl - b.isl)) {
    const stat = p.metrics[metric];
    if (stat) bars.set(pointSeries(p), { id: pointSeries(p), label: p.series, value: stat.median, verified: p.verified, note: memoryNote(p, metric), point: p });
  }
  return [...bars.values()].sort((a, b) => bySeries(a.id, b.id));
}

// The series of the selected model, one entry per framework and backend in the order the charts draw them: the page's one legend.
export function legendSeries(points: Point[]): { id: string; label: string }[] {
  const series = new Map(points.map((p) => [pointSeries(p), p.series]));
  return [...series].sort(([a], [b]) => bySeries(a, b)).map(([id, label]) => ({ id, label }));
}

export type Bar = { id: string; label: string; value: number; verified: boolean; note: string | null; point: Point };

// What a memory bar measures beyond its title: a run with no usable load boundary says its peak includes the load. Null otherwise
// and for a metric that is not memory.
export function memoryNote(p: Point, metric: PointMetric): string | null {
  return MEMORY_METRICS.includes(metric) && p.memory_scope === "includes_load" ? "includes load" : null;
}

function niceMax(value: number): number {
  const exp = 10 ** Math.floor(Math.log10(value));
  for (const step of [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]) if (step * exp >= value) return step * exp;
  return 10 * exp;
}

// The top of the y axis (in the unit's scale) that fits every line of every set. Charts of one metric in two columns pass both sets
// so the same value is the same height in both.
export function yTop(lineSets: Line[][], scale: number): number {
  return niceMax(Math.max(...lineSets.flatMap((lines) => lines.flatMap((l) => l.points.map((p) => p.stat.max)))) * scale);
}

// One row of a chart tooltip: a series and its value, formatted with the chart's unit. A series with no point at the input length
// reads "no data".
export type TipRow = { id: string; label: string; value: string };

export const NO_DATA = "no data";

// Every series of the chart at one input length, in the order of the legend (the order of `lines`).
export function rowsAt(lines: Line[], isl: number, unit: Unit): TipRow[] {
  return lines.map((l) => {
    const p = l.points.find((q) => q.isl === isl);
    return { id: l.id, label: l.label, value: p ? `${num(p.stat.median, unit)} ${unit.suffix}` : NO_DATA };
  });
}
