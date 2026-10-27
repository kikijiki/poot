import { describe, expect, test } from "bun:test";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { problems, sections } from "./navbar.mjs";

// A docs tree: a home page, a section folder (a sidebar), a folder holding one page, and a top-level page.
function docs() {
  const dir = mkdtempSync(join(tmpdir(), "navbar-"));
  writeFileSync(join(dir, "index.md"), "---\nsidebar_position: 1\n---\n");
  mkdirSync(join(dir, "serve"));
  writeFileSync(join(dir, "serve", "index.md"), "---\ntitle: Serving\n---\n");
  writeFileSync(join(dir, "serve", "_category_.json"), JSON.stringify({ label: "Serving", position: 2, link: { type: "doc", id: "serve/index" } }));
  mkdirSync(join(dir, "performance"));
  writeFileSync(join(dir, "performance", "index.mdx"), "---\ntitle: Performance\nsidebar_position: 3\n---\n");
  writeFileSync(join(dir, "status.mdx"), "---\nid: status\ntitle: Status\nsidebar_position: 4\n---\n");
  return dir;
}

const config = () => ({
  navbar: {
    items: [
      { type: "docSidebar", sidebarId: "serve", position: "left", label: "Serving" },
      { type: "doc", docId: "performance/index", position: "left", label: "Performance" },
      { type: "doc", docId: "status", position: "left", label: "Status" },
      { href: "https://example.com", label: "GitHub", position: "right" },
    ],
  },
});
const sidebarIds = ["serve"];

describe("navbar", () => {
  test("sections are the sidebar folders and the single pages, in position order", () => {
    expect(sections(docs()).map((s) => s.key)).toEqual(["serve", "performance/index", "status"]);
  });

  test("a navbar that lists every section passes", () => {
    expect(problems(docs(), config(), sidebarIds)).toEqual([]);
  });

  test("a section without a navbar entry fails", () => {
    const c = config();
    c.navbar.items.splice(2, 1);
    expect(problems(docs(), c, sidebarIds)).toContain('the section "Status" (status) has no navbar entry');
  });

  test("an item that points to a missing doc fails", () => {
    const c = config();
    c.navbar.items.push({ type: "doc", docId: "gone/index", position: "left", label: "Gone" });
    expect(problems(docs(), c, sidebarIds)).toContain('navbar item "Gone" points to the missing doc gone/index');
  });

  test("a section folder that is a doc link instead of its sidebar fails", () => {
    const c = config();
    c.navbar.items[0] = { type: "doc", docId: "serve/index", position: "left", label: "Serving" };
    expect(problems(docs(), c, sidebarIds).some((e) => e.includes("has no navbar entry"))).toBe(true);
  });

  test("a navbar out of position order fails", () => {
    const c = config();
    c.navbar.items.reverse();
    expect(problems(docs(), c, sidebarIds).some((e) => e.includes("differs from the section order"))).toBe(true);
  });

  test("a page in a folder with no category file is in no sidebar and fails", () => {
    const dir = docs();
    writeFileSync(join(dir, "performance", "extra.md"), "---\ntitle: Extra\n---\n");
    expect(problems(dir, config(), sidebarIds).some((e) => e.includes("are in no sidebar"))).toBe(true);
  });

  test("a sidebar that is no section folder fails", () => {
    expect(problems(docs(), config(), ["serve", "ghost"])).toContain("the sidebar ghost is no section folder");
  });
});
