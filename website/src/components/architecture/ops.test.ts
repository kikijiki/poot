import { describe, expect, test } from "bun:test";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { CATEGORIES } from "./ops";

const page = readFileSync(join(import.meta.dir, "../../../docs/architecture/graph-ir.mdx"), "utf8");

const EXPECTED_OPS: Record<string, string[]> = {
  elementwise: ["Unary(op)", "Binary(op)", "Select", "Cast(to)"],
  shape: ["Broadcast", "Reshape", "Transpose", "Slice", "Concat"],
  data: ["Gather", "ScatterUpdate", "DynamicUpdateSlice", "ArgTopK(k)", "ExpertGroup(experts)"],
  reduce: ["Reduce(op, axis, keepdim)"],
  compute: [
    "MatMul", "IndexedMatMul", "PackedDequant", "PackedContraction / PackedRowGather",
    "DenseContraction / DenseRowGather", "GroupedDenseContraction", "Rope(rot)", "Attention",
    "Fused / FusedRow", "ContractionEpilogue", "AllReduce / AllGather",
  ],
};

describe("graph IR op reference", () => {
  test("every op appears exactly once, in its category", () => {
    const actual = Object.fromEntries(CATEGORIES.map((c) => [c.id, c.ops.map((o) => o.name)]));
    expect(actual).toEqual(EXPECTED_OPS);
  });

  test("every op has annotated inputs, an annotated output and notes", () => {
    for (const op of CATEGORIES.flatMap((c) => c.ops)) {
      expect(op.inputs.length).toBeGreaterThan(0);
      for (const field of [...op.inputs, op.output]) {
        expect(field.name.trim().length).toBeGreaterThan(0);
        expect(field.shape ?? "x").not.toBe("");
        expect(field.meaning.trim().length).toBeGreaterThan(0);
      }
      expect(op.notes.length).toBeGreaterThan(0);
    }
  });

  test("an op's input names are distinct", () => {
    for (const op of CATEGORIES.flatMap((c) => c.ops)) {
      const names = op.inputs.map((i) => i.name);
      expect(new Set(names).size).toBe(names.length);
    }
  });

  test("each category is a linkable subsection with a description and its list", () => {
    for (const c of CATEGORIES) {
      const title = c.id[0].toUpperCase() + c.id.slice(1);
      expect(page).toMatch(new RegExp(`^### ${title}\\n\\n[A-Z].+\\n\\n<OpList category="${c.id}" />$`, "m"));
    }
  });
});
