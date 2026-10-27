import { describe, expect, test } from "bun:test";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import data from "../../data/status.json";
import performance from "../../data/performance.json";
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

  // The page says poot is behind llama.cpp on decode and first-token time in every measured cell. That is a claim about
  // the data: when poot reaches a baseline on either metric in any verified, latest cell the claim is false and this
  // test says which one. Memory has its own tests below.
  test("poot is behind the best baseline on decode and first-token time in every compared verified cell of every hardware", () => {
    type Point = { framework: string; isl: number; model: string; verified: boolean; latest: boolean; metrics: Record<string, { median: number }> };
    const better: Record<string, (values: number[]) => number> = { decode_tok_s: (v) => Math.max(...v), ttft_ms: (v) => Math.min(...v) };
    const ahead: string[] = [];
    for (const hw of performance.hardware as unknown as { id: string; points: Point[] }[]) {
      for (const metric of Object.keys(better)) {
        const cells = new Map<string, { poot: number[]; base: number[] }>();
        for (const p of hw.points) {
          const m = p.metrics[metric];
          if (!p.verified || !p.latest || !m) continue;
          const cell = cells.get(`${p.model}@${p.isl}`) ?? { poot: [], base: [] };
          cell[p.framework === "poot" ? "poot" : "base"].push(m.median);
          cells.set(`${p.model}@${p.isl}`, cell);
        }
        for (const [key, cell] of cells) {
          if (!cell.poot.length || !cell.base.length) continue;
          const [poot, base] = [better[metric](cell.poot), better[metric](cell.base)];
          const behind = metric === "decode_tok_s" ? poot < base : poot > base;
          if (!behind) ahead.push(`${hw.id} ${metric} ${key}`);
        }
      }
    }
    expect(ahead).toEqual([]);
  });

  // Memory is the peak at inference (after load and warmup), which the harness derives from the stored traces. The page says poot holds
  // more GPU memory than the best baseline in every compared cell and less host memory on most Strix Halo cells; both are claims about
  // the data, so a cell that contradicts one fails here.
  type MemPoint = { framework: string; isl: number; model: string; verified: boolean; latest: boolean; over_sweep?: boolean; metrics: Record<string, { median: number }> };
  const memoryCells = (hw: { points: MemPoint[] }, metric: string) => {
    const cells = new Map<string, { poot: number[]; base: number[] }>();
    for (const p of hw.points) {
      const m = p.metrics[metric];
      if (!p.verified || !p.latest || !p.over_sweep || !m) continue;
      const cell = cells.get(p.model) ?? { poot: [], base: [] };
      cell[p.framework === "poot" ? "poot" : "base"].push(m.median);
      cells.set(p.model, cell);
    }
    return [...cells].filter(([, c]) => c.poot.length && c.base.length).map(([model, c]) => ({ model, poot: Math.min(...c.poot), base: Math.min(...c.base) }));
  };
  const hardware = performance.hardware as unknown as { id: string; points: MemPoint[] }[];

  test("poot holds at least as much GPU memory as the best baseline in every compared verified cell", () => {
    const lower: string[] = [];
    for (const hw of hardware) for (const c of memoryCells(hw, "peak_vram_bytes")) if (c.poot < c.base) lower.push(`${hw.id} ${c.model}`);
    expect(lower).toEqual([]);
  });

  test("poot holds less host memory than the best baseline on most Strix Halo cells", () => {
    const strix = hardware.find((h) => h.id.includes("8060s"));
    const cells = memoryCells(strix as { points: MemPoint[] }, "peak_rss_bytes");
    expect(cells.length).toBeGreaterThan(2);
    expect(cells.filter((c) => c.poot < c.base).length).toBeGreaterThan(cells.length / 2);
  });
});
