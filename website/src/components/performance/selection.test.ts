import { describe, expect, test } from "bun:test";
import data from "../../data/performance.json";
import { SERIES_SLOTS, seriesColor } from "./format";
import { CHART_METRICS, MS, TOK_S } from "./format";
import { legendSeries, NO_DATA, rowsAt, chartMetrics, commonMetrics, runBars, defaultSecond, yTop, defaultHardware, defaultModel, modelSize, fromQuery, options, pointSeries, resolve, selectPoints, seriesLines, toQuery } from "./selection";
import type { Hardware, PerformanceData, Point, Stat } from "./types";

const stat = (median: number): Stat => ({ median, min: median * 0.99, max: median * 1.01, n: 5 });

type Spec = { framework: string; backend: string; model: string; isl: number; decode?: number; prefill?: number; extra?: Partial<Point> };

function point(s: Spec): Point {
  return {
    set: "set",
    date: "2026-10-23",
    model: s.model,
    framework: s.framework,
    backend: s.backend,
    series: `${s.framework} (${s.backend})`,
    precision: "bf16",
    isl: s.isl,
    repeats: 5,
    hosts: 1,
    host_labels: ["Host A"],
    cells: { verified: 1, unverified: 0 },
    spread: {},
    latest: true,
    verified: true,
    memory_scope: "inference",
    band: 0.01,
    metrics: { ...(s.decode ? { decode_tok_s: stat(s.decode) } : {}), ...(s.prefill ? { prefill_tok_s: stat(s.prefill) } : {}) },
    identity: {},
    ...s.extra,
  };
}

const poot = (model: string, isl: number, decode: number, prefill?: number, extra?: Partial<Point>, backend = "ptx") =>
  point({ framework: "poot", backend, model, isl, decode, prefill, extra });
const llama = (model: string, isl: number, decode: number, prefill?: number, extra?: Partial<Point>) =>
  point({ framework: "llamacpp", backend: "cuda", model, isl, decode, prefill, extra });

const hw = (id: string, points: Point[]): Hardware => ({ id, label: id, points });

const strix = hw("amd-radeon-8060s-graphics", [
  poot("m1", 128, 100, 900),
  poot("m1", 128, 80, 700, undefined, "vulkan"),
  llama("m1", 128, 150, 3000),
  poot("m1", 8192, 50, 400),
  llama("m1", 8192, 100, 2000),
  poot("m2", 128, 20, 90),
  poot("m2", 6, 22),
]);
const a100 = hw("a100", [poot("n1", 128, 50), llama("n1", 128, 200), poot("n2", 128, 20)]);
const report: PerformanceData = { schema: 3, policy: { min_repeats: 5, max_uncertainty: 0.1 }, hardware: [a100, strix] };
const sel = (raw: Parameters<typeof resolve>[1]) => resolve(report, raw);

describe("defaults", () => {
  test("the page opens on the Strix Halo entry, whatever the other hardware has", () => {
    expect(defaultHardware(report).id).toBe("amd-radeon-8060s-graphics");
  });

  test("without a Strix Halo entry the first hardware is the default", () => {
    expect(defaultHardware({ ...report, hardware: [a100] }).id).toBe("a100");
  });

  test("the selection has no host: one hardware choice per GPU model, hosts only inform the point", () => {
    expect(Object.keys(sel({})).sort()).toEqual(["hw", "model"]);
    expect(toQuery(sel({}))).toBe("hw=amd-radeon-8060s-graphics&model=m1");
    expect(fromQuery("hw=a100&host=h2&model=n1")).not.toHaveProperty("host");
  });

  test("the default selection is Strix Halo and its first model", () => {
    expect(sel({})).toEqual({ hw: "amd-radeon-8060s-graphics", model: "m1" });
  });
});

describe("one model at a time", () => {
  test("only the points of the selected model are selected", () => {
    expect(new Set(selectPoints(strix, sel({ model: "m2" })).map((p) => p.model))).toEqual(new Set(["m2"]));
  });

  test("an unknown model falls back to the first model of the hardware", () => {
    expect(sel({ model: "nope" }).model).toBe("m1");
  });

  test("a superseded set is left out of the charts", () => {
    const old = poot("m1", 128, 10, undefined, { latest: false });
    const withOld = hw("amd-radeon-8060s-graphics", [...strix.points, old]);
    const s = resolve({ ...report, hardware: [withOld] }, {});
    expect(selectPoints(withOld, s)).toHaveLength(selectPoints(strix, s).length);
  });

  test("an unknown hardware falls back to the default", () => {
    expect(sel({ hw: "nope" }).hw).toBe("amd-radeon-8060s-graphics");
  });
});

describe("charts", () => {
  const withMetrics = (model: string, metrics: Point["metrics"], isl = 128, backend = "ptx") => poot(model, isl, 1, undefined, { metrics }, backend);
  const reportOf = (...hardware: Hardware[]): PerformanceData => ({ ...report, hardware });
  const all: Point["metrics"] = { peak_rss_bytes: stat(1), ttft_ms: stat(2), decode_tok_s: stat(3), e2e_ms: stat(4), energy_j: stat(5), peak_vram_bytes: stat(6), prefill_tok_s: stat(7) };
  const full = hw("full", [withMetrics("m", all)]);

  test("every metric all hardware entries have is charted, in the page's order", () => {
    const r = reportOf(full, hw("full2", [withMetrics("m", all)]));
    expect(chartMetrics(r, selectPoints(full, { hw: "full", model: "m" }))).toEqual(["decode_tok_s", "prefill_tok_s", "ttft_ms", "e2e_ms", "peak_rss_bytes", "peak_vram_bytes", "energy_j"]);
  });

  test("a metric one hardware entry lacks is charted on none, and appears once that hardware has it", () => {
    const lacking = hw("lacking", [withMetrics("m", { decode_tok_s: stat(3), ttft_ms: stat(1) })]);
    const r = reportOf(full, lacking);
    expect(commonMetrics(r)).toEqual(["decode_tok_s", "ttft_ms"]);
    expect(chartMetrics(r, selectPoints(full, { hw: "full", model: "m" }))).toEqual(["decode_tok_s", "ttft_ms"]);
    const sampled = hw("lacking", [withMetrics("m", { decode_tok_s: stat(3), ttft_ms: stat(1), peak_vram_bytes: stat(2), energy_j: stat(5) })]);
    expect(commonMetrics(reportOf(full, sampled))).toEqual(["decode_tok_s", "ttft_ms", "peak_vram_bytes", "energy_j"]);
  });

  test("a superseded point does not make a metric common, and a model without the metric has no chart for it", () => {
    const old = hw("old", [poot("m", 128, 1, undefined, { latest: false, metrics: { energy_j: stat(1), decode_tok_s: stat(1) } })]);
    expect(commonMetrics(reportOf(full, old))).not.toContain("energy_j");
    const other = hw("other", [withMetrics("m", all), withMetrics("n", { decode_tok_s: stat(3) })]);
    expect(chartMetrics(reportOf(other, full), selectPoints(other, { hw: "other", model: "n" }))).toEqual(["decode_tok_s"]);
    expect(chartMetrics(reportOf(other, full), [])).toEqual([]);
  });

  test("a whole-run metric is one bar per framework and backend, on the longest input length that carries it", () => {
    const points = [
      withMetrics("m", { peak_rss_bytes: stat(10) }, 128),
      withMetrics("m", { peak_rss_bytes: stat(30) }, 8192),
      withMetrics("m", { peak_rss_bytes: stat(20) }, 128, "rocm"),
      withMetrics("m", { decode_tok_s: stat(1) }, 128, "vulkan"),
    ];
    expect(runBars(points, "peak_rss_bytes").map((b) => [b.label, b.value])).toEqual([["poot (ptx)", 30], ["poot (rocm)", 20]]);
    expect(CHART_METRICS.filter((m) => m.wholeRun).map((m) => m.key)).toEqual(["peak_rss_bytes", "peak_vram_bytes", "energy_j"]);
  });

  test("a memory bar states the load peak beside the inference peak, or says it includes the load", () => {
    const GiB = 2 ** 30;
    const split = withMetrics("m", { peak_rss_bytes: stat(6 * GiB), load_rss_bytes: stat(26 * GiB) });
    const lifetime = withMetrics("m", { peak_rss_bytes: stat(53 * GiB) }, 128, "rocm");
    lifetime.memory_scope = "includes_load";
    expect(runBars([split, lifetime], "peak_rss_bytes").map((b) => [b.label, b.note])).toEqual([["poot (ptx)", "26.00 GiB while loading"], ["poot (rocm)", "includes model load"]]);
    expect(runBars([withMetrics("m", { energy_j: stat(1) })], "energy_j").map((b) => b.note)).toEqual([null]);
  });
});

describe("URL round trip", () => {
  test("a selection survives the query string", () => {
    const s = sel({ hw: "a100", model: "n2" });
    expect(resolve(report, fromQuery(toQuery(s)))).toEqual(s);
  });

  test("the URL has hardware and model only: no metrics, no host, no scenario, no view", () => {
    expect([...new URLSearchParams(toQuery(sel({}))).keys()]).toEqual(["hw", "model"]);
  });

  test("a query with unknown values or the dropped parameters resolves to the defaults", () => {
    expect(resolve(report, fromQuery("hw=zzz&metrics=bogus&host=h&view=table&model=x"))).toEqual(sel({}));
  });
});

describe("series over the input tokens", () => {
  test("one line per framework and backend, ordered by input tokens, only points that carry the metric", () => {
    const lines = seriesLines(selectPoints(strix, sel({})), "decode_tok_s");
    expect(lines.map((l) => l.label).sort()).toEqual(["llamacpp (cuda)", "poot (ptx)", "poot (vulkan)"]);
    const llamaLine = lines.find((l) => l.label === "llamacpp (cuda)")!;
    expect(llamaLine.points.map((p) => p.isl)).toEqual([128, 8192]);
    const prefill = seriesLines(selectPoints(strix, sel({})), "prefill_tok_s");
    expect(prefill.find((l) => l.label === "llamacpp (cuda)")!.points).toHaveLength(2);
  });

  test("a line with an unverified point is not verified", () => {
    const pts = [poot("m", 128, 10), poot("m", 512, 9, undefined, { verified: false })];
    expect(seriesLines(pts, "decode_tok_s")[0].verified).toBe(false);
  });

  test("single points and sweep points of one series share the axis", () => {
    const lines = seriesLines(selectPoints(strix, sel({ model: "m2" })), "decode_tok_s");
    expect(lines[0].points.map((p) => p.isl)).toEqual([6, 128]);
  });
});

describe("points aggregated over hosts", () => {
  const merged = hw("g", [
    poot("m", 128, 120, undefined, { hosts: 3, cells: { verified: 3, unverified: 0 }, spread: { decode_tok_s: { min: 100, max: 400 } } }),
    llama("m", 128, 60),
  ]);

  test("a multi-host point is one point of one series, never a split line", () => {
    const lines = seriesLines(selectPoints(merged, resolve({ ...report, hardware: [merged] }, {})), "decode_tok_s");
    expect(lines.map((l) => l.points.length)).toEqual([1, 1]);
  });

});

describe("default model", () => {
  test("a model's size is read from its id; an id without one is the largest", () => {
    expect([modelSize("qwen3-0.6b"), modelSize("qwen3-4b-q4km"), modelSize("qwen3.8-27b-q4")]).toEqual([0.6, 4, 27]);
    expect(modelSize("qwen3.8-flash-next-q2")).toBe(Number.POSITIVE_INFINITY);
  });

  test("the smallest model the hardware has is the default, whatever the data order", () => {
    expect(defaultModel(["qwen3.8-27b-q4", "qwen3-4b", "qwen3-0.6b", "qwen3.8-flash-next-q2"])).toBe("qwen3-0.6b");
    expect(defaultModel(["qwen3.8-flash-next-q2", "qwen3.8-27b-q4", "qwen3-4b-q4km"])).toBe("qwen3-4b-q4km");
    expect(defaultModel(["qwen3-4b-q4km", "qwen3-4b"])).toBe("qwen3-4b");
  });

  test("sizes are compared as numbers, not text", () => {
    expect(defaultModel(["m-10b", "m-9b", "m-2.5b"])).toBe("m-2.5b");
  });

  test("a hardware with no sized model opens on its first model by name", () => {
    expect(defaultModel(["other", "another"])).toBe("another");
    expect(defaultModel([])).toBe("");
  });

  test("an unknown model in the URL resolves to the smallest, and a known one wins", () => {
    const g = hw("g", [poot("big-8b", 128, 1), poot("qwen3-0.6b", 128, 1)]);
    expect(resolve({ ...report, hardware: [g] }, { model: "nope" }).model).toBe("qwen3-0.6b");
    expect(resolve({ ...report, hardware: [g] }, { model: "big-8b" }).model).toBe("big-8b");
  });

  test("the page opens on the smallest model of the committed data", () => {
    const committed = data as unknown as PerformanceData;
    expect(resolve(committed, {}).model).toBe("qwen3-0.6b");
  });

  test("the second column opens on the same model elsewhere, else the smallest other model", () => {
    const c: PerformanceData = { ...report, hardware: [hw("x", [poot("m-8b", 128, 1), poot("m-1b", 128, 1), poot("m-3b", 128, 1)])] };
    expect(defaultSecond(c, { hw: "x", model: "m-1b" }).model).toBe("m-3b");
  });
});

describe("comparison mode", () => {
  const two = (id: string, models: string[]) => hw(id, models.map((m) => poot(m, 128, 10)));
  const cmp: PerformanceData = { ...report, hardware: [two("g1", ["a", "b"]), two("g2", ["a"]), two("g3", ["c"])] };

  test("switched on, column B is the same model on another hardware that has it", () => {
    expect(defaultSecond(cmp, { hw: "g1", model: "a" })).toEqual({ hw: "g2", model: "a" });
  });

  test("with no other hardware having the model, column B is another model of the same hardware", () => {
    expect(defaultSecond(cmp, { hw: "g1", model: "b" })).toEqual({ hw: "g1", model: "a" });
    expect(defaultSecond(cmp, { hw: "g3", model: "c" })).toEqual({ hw: "g3", model: "c" });
  });

  test("the second column resolves on its own and a model stays when its new hardware has it", () => {
    const s = resolve(cmp, { hw: "g1", model: "a", b: { hw: "g2", model: "b" } });
    expect(s).toEqual({ hw: "g1", model: "a", b: { hw: "g2", model: "a" } });
    expect(resolve(cmp, { hw: "g2", model: "a" }).b).toBeUndefined();
  });

  test("the URL carries compare=1 with hw2 and model2, and a shared link restores both columns", () => {
    const s = resolve(cmp, { hw: "g1", model: "b", b: { hw: "g2", model: "a" } });
    const q = new URLSearchParams(toQuery(s));
    expect([q.get("compare"), q.get("hw2"), q.get("model2")]).toEqual(["1", "g2", "a"]);
    expect(resolve(cmp, fromQuery(toQuery(s)))).toEqual(s);
    expect(new URLSearchParams(toQuery(resolve(cmp, { hw: "g1", model: "a" }))).has("compare")).toBe(false);
  });

  test("a compare link with unknown column B values resolves to a valid column B", () => {
    expect(resolve(cmp, fromQuery("hw=g1&model=a&compare=1&hw2=zzz&model2=zzz")).b).toEqual({ hw: "g2", model: "a" });
  });

  test("charts of one metric in two columns share one y range, so equal values are equal heights", () => {
    const small = seriesLines([poot("m", 128, 40)], "decode_tok_s");
    const big = seriesLines([poot("m", 128, 900)], "decode_tok_s");
    expect(yTop([small], 1)).toBe(50);
    expect(yTop([small, big], 1)).toBe(yTop([big, small], 1));
    expect(yTop([small, big], 1)).toBeGreaterThanOrEqual(900);
  });

  test("a metric missing in one column adds nothing to the shared range", () => {
    const withTtft = poot("m", 128, 10, undefined, { metrics: { decode_tok_s: stat(10), ttft_ms: stat(500) } });
    expect(yTop([seriesLines([withTtft], "ttft_ms"), seriesLines([poot("m", 128, 10)], "ttft_ms")], 1)).toBe(yTop([seriesLines([withTtft], "ttft_ms")], 1));
  });
});

describe("series colours", () => {
  const pairs = [...new Set((data as unknown as PerformanceData).hardware.flatMap((h) => h.points.map(pointSeries)))];

  test("every framework and backend pair of the committed data has its own slot", () => {
    expect(pairs.filter((id) => !(id in SERIES_SLOTS))).toEqual([]);
    expect(new Set(Object.values(SERIES_SLOTS)).size).toBe(Object.keys(SERIES_SLOTS).length);
  });

  test("a colour depends on the pair alone", () => {
    expect(seriesColor("poot ptx")).toBe("var(--series-1)");
    expect(seriesColor("poot rocm")).not.toBe(seriesColor("poot ptx"));
    expect(seriesColor("llamacpp cuda")).not.toBe(seriesColor("llamacpp vulkan"));
  });
});

describe("tooltip rows", () => {
  const lines = seriesLines(selectPoints(strix, { hw: strix.id, model: "m1" }), "decode_tok_s");

  test("every series is listed in legend order with the chart's metric and unit", () => {
    const rows = Object.fromEntries(rowsAt(lines, 128, TOK_S).map((r) => [r.id, r.value]));
    expect(rows).toEqual({ "poot ptx": "100.0 tok/s", "poot vulkan": "80.0 tok/s", "llamacpp cuda": "150.0 tok/s" });
    expect(rowsAt(lines, 128, TOK_S).map((r) => r.id)).toEqual(lines.map((l) => l.id));
  });

  test("a series with no point at the x reads no data", () => {
    const at8192 = rowsAt(lines, 8192, TOK_S);
    expect(at8192.map((r) => r.id)).toEqual(lines.map((l) => l.id));
    expect(at8192.find((r) => r.id === "poot vulkan")?.value).toBe(NO_DATA);
    expect(at8192.find((r) => r.id === "poot ptx")?.value).toBe("50.0 tok/s");
  });

  test("the value is formatted with the unit it is given", () => {
    const ms = seriesLines([poot("m1", 128, 100, undefined, { metrics: { ttft_ms: stat(12.34) } })], "ttft_ms");
    expect(rowsAt(ms, 128, MS)[0].value).toBe("12.3 ms");
  });
});

describe("legend", () => {
  test("one entry per series of the model, in chart order, whatever the metric or input length", () => {
    const points = selectPoints(strix, { hw: strix.id, model: "m1" });
    expect(legendSeries(points).map((s) => s.id)).toEqual(["poot ptx", "poot vulkan", "llamacpp cuda"]);
    expect(legendSeries(points)[2].label).toBe("llamacpp (cuda)");
    expect(legendSeries(points).map((s) => s.id)).toEqual(seriesLines(points, "decode_tok_s").map((l) => l.id));
  });

  test("a model another hardware has does not add its series", () => {
    expect(legendSeries(selectPoints(a100, { hw: a100.id, model: "n1" })).map((s) => s.id)).toEqual(["poot ptx", "llamacpp cuda"]);
    expect(legendSeries([])).toEqual([]);
  });
});
