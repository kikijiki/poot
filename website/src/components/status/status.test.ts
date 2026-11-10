import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import data from "../../data/status.json";
import performance from "../../data/performance.json";
import { BACKEND_PAIRS } from "../performance/format";
import { iconNames } from "./Icon";
import { badgeLabel, type StatusData } from "./types";

const board = data as StatusData;
const docs = join(import.meta.dir, "../../../docs");

// A site path `/docs/a/b` is served by docs/a/b.{md,mdx} or docs/a/b/index.{md,mdx}.
function resolves(link: string): boolean {
  const rel = link.replace(/^\/docs\/?/, "");
  return ["md", "mdx"].some((ext) => existsSync(join(docs, `${rel}.${ext}`)) || existsSync(join(docs, rel, `index.${ext}`)));
}

describe("status board data", () => {
  test("every card sits in a declared column and every column has cards", () => {
    const ids = board.columns.map((c) => c.id);
    for (const card of board.cards) expect(ids).toContain(card.column);
    for (const id of ids) expect(board.cards.some((c) => c.column === id)).toBe(true);
  });

  test("badges and icons are known", () => {
    for (const card of board.cards) {
      expect(Object.keys(badgeLabel)).toContain(card.badge);
      expect(iconNames).toContain(card.icon);
    }
  });

  test("card ids are unique", () => {
    expect(new Set(board.cards.map((c) => c.id)).size).toBe(board.cards.length);
  });

  test("every link resolves to a docs page", () => {
    const missing = board.cards.filter((c) => !resolves(c.link)).map((c) => `${c.id} -> ${c.link}`);
    expect(missing).toEqual([]);
  });

  test("no card numbers in the data", () => {
    expect(JSON.stringify(board)).not.toMatch(/\b(POOT|card)[- ]?#?\d+/i);
  });

  test("no text is a long paragraph", () => {
    for (const text of [board.later, ...board.limitations, ...board.cards.map((c) => c.line)]) expect(text.length).toBeLessThan(160);
  });

  // Every named family and format is a row of the generated supported-models page that plans on at least one backend,
  // so the Status page cannot name something the registry no longer has.
  test("every family and format named on a card is a planning row of the supported-models page", () => {
    const page = readFileSync(join(docs, "develop/supported-models.md"), "utf8");
    const rows = page.split("\n").filter((line) => line.startsWith("| "));
    const planning = (ref: string) => rows.some((row) => (row.startsWith(`| \`${ref}\` |`) || row.startsWith(`| ${ref} |`)) && row.includes(">Plans<"));
    const refs = board.cards.flatMap((c) => (c.groups ?? []).flatMap((g) => g.items.flatMap((i) => (i.ref ? [i.ref] : []))));
    expect(refs.length).toBeGreaterThan(10);
    expect(refs.filter((ref) => !planning(ref))).toEqual([]);
  });

  test("a card names at most 16 things", () => {
    for (const card of board.cards) expect((card.groups ?? []).flatMap((g) => g.items).length).toBeLessThanOrEqual(16);
  });

  // poot is compared with llama.cpp only on the same backend API (BACKEND_PAIRS): poot Vulkan with llama.cpp Vulkan, ROCm with ROCm,
  // PTX with CUDA. A pair of one cell is the two series' medians at one model and input length, both verified and latest.
  type PairPoint = { framework: string; backend: string; isl: number; model: string; verified: boolean; latest: boolean; over_sweep?: boolean; metrics: Record<string, { median: number }> };
  type Pair = { hw: string; backend: string; key: string; poot: number; llamacpp: number };
  const pairedCells = (metric: string, wholeRun = false): Pair[] => {
    const cells: Pair[] = [];
    for (const hw of performance.hardware as unknown as { id: string; points: PairPoint[] }[]) {
      const at = new Map<string, number>();
      for (const p of hw.points) if (p.verified && p.latest && p.metrics[metric] && (!wholeRun || p.over_sweep)) at.set(`${p.framework} ${p.backend} ${p.model}@${p.isl}`, p.metrics[metric].median);
      for (const pair of BACKEND_PAIRS) {
        for (const [key, poot] of at) {
          const [framework, backend, cell] = key.split(" ");
          if (`${framework} ${backend}` !== pair.poot) continue;
          const llamacpp = at.get(`${pair.llamacpp} ${cell}`);
          if (llamacpp !== undefined) cells.push({ hw: hw.id, backend: pair.key, key: cell, poot, llamacpp });
        }
      }
    }
    return cells;
  };
  const limitations = board.limitations.join(" ");

  // The page says poot is behind llama.cpp on the same backend on decode in every pair, by the stated range, and on first-token time in
  // the stated share of pairs. That is a claim about the data: when poot reaches llama.cpp on decode anywhere, or the counts and ranges
  // drift, this test says which. The first-token exceptions are the small qwen3 rows (Strix Halo ROCm, RTX PRO 6000 PTX) the page names.
  test("poot is behind llama.cpp on the same backend on decode in every pair, and on first-token time as the page states", () => {
    const decode = pairedCells("decode_tok_s");
    expect(decode.filter((c) => c.poot >= c.llamacpp).map((c) => `${c.hw} ${c.backend} ${c.key}`)).toEqual([]);
    const slower = decode.map((c) => c.llamacpp / c.poot);
    expect(limitations).toContain(`from ${Math.min(...slower).toFixed(2)}x to ${Math.max(...slower).toFixed(1)}x slower`);

    const first = pairedCells("ttft_ms");
    const behind = first.filter((c) => c.poot > c.llamacpp);
    expect(limitations).toContain(`Behind on first token in ${behind.length} of ${first.length} cells, up to ${Math.max(...behind.map((c) => c.poot / c.llamacpp)).toFixed(1)}x slower`);
    const ahead = first.filter((c) => c.poot <= c.llamacpp);
    const smallQwen = /^qwen3-(0\.6b|4b|4b-q4km)@/;
    expect(ahead.filter((c) => !smallQwen.test(c.key) || c.backend === "vulkan").map((c) => `${c.hw} ${c.backend} ${c.key}`)).toEqual([]);
    expect(ahead.length).toBeGreaterThan(0);
  });

  // Memory is the peak at inference (after load and warmup), which the harness derives from the stored traces. The page says poot holds
  // more GPU memory than llama.cpp on the same backend in every pair, by up to the stated factor, and less host memory in every pair;
  // both are claims about the data, so a pair that contradicts one fails here.
  test("poot holds at least as much GPU memory as llama.cpp on the same backend in every pair, by up to the stated factor", () => {
    const cells = pairedCells("peak_vram_bytes", true);
    expect(cells.length).toBeGreaterThan(10);
    expect(cells.filter((c) => c.poot < c.llamacpp).map((c) => `${c.hw} ${c.backend} ${c.key}`)).toEqual([]);
    expect(limitations).toContain(`Holds up to ${Math.max(...cells.map((c) => c.poot / c.llamacpp)).toFixed(1)}x the GPU memory of llama.cpp`);
  });

  test("poot holds less host memory than llama.cpp on the same backend in every pair", () => {
    const cells = pairedCells("peak_rss_bytes", true);
    expect(cells.length).toBeGreaterThan(10);
    expect(cells.filter((c) => c.poot >= c.llamacpp).map((c) => `${c.hw} ${c.backend} ${c.key}`)).toEqual([]);
  });
});
