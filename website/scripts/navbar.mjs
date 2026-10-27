// The navbar is the only place for the top-level sections. A section is either
//   - a folder of docs/ with a _category_.json: its pages are one sidebar (the sidebar id is the folder name,
//     sidebars.ts) and its navbar entry is a `docSidebar` item; or
//   - a single page (a top-level page other than the docs home, or a folder holding only an index page): it has no
//     sidebar and its navbar entry is a `doc` item.
// This check fails when a section has no navbar entry, an entry points at nothing, the order differs from the
// sections' positions, or a page is in no sidebar and no navbar entry (unreachable).
import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";

function frontmatter(path) {
  const text = readFileSync(path, "utf8");
  const match = text.match(/^---\n([\s\S]*?)\n---/);
  const field = (name) => match?.[1].match(new RegExp(`^${name}:\\s*(.+)$`, "m"))?.[1].trim().replace(/^["']|["']$/g, "");
  return { id: field("id"), title: field("title"), position: Number(field("sidebar_position") ?? Infinity) };
}

const isPage = (name) => /\.mdx?$/.test(name);
const isIndex = (name) => /^index\.mdx?$/.test(name);

export function sections(docsDir) {
  const found = [];
  for (const entry of readdirSync(docsDir, { withFileTypes: true })) {
    const path = join(docsDir, entry.name);
    if (entry.isDirectory() && existsSync(join(path, "_category_.json"))) {
      const category = JSON.parse(readFileSync(join(path, "_category_.json"), "utf8"));
      found.push({ kind: "sidebar", key: entry.name, label: category.label, position: category.position });
    } else if (entry.isDirectory()) {
      const index = readdirSync(path).find(isIndex);
      if (!index) continue;
      const meta = frontmatter(join(path, index));
      found.push({ kind: "doc", key: `${entry.name}/index`, label: meta.title, position: meta.position });
    } else if (entry.isFile() && isPage(entry.name) && !isIndex(entry.name)) {
      const meta = frontmatter(path);
      found.push({ kind: "doc", key: meta.id ?? entry.name.replace(/\.mdx?$/, ""), label: meta.title, position: meta.position });
    }
  }
  return found.sort((a, b) => a.position - b.position);
}

function docExists(docsDir, docId) {
  return [docId, `${docId}/index`].some((id) => [".md", ".mdx"].some((ext) => existsSync(join(docsDir, id + ext))));
}

const keyOf = (item) => (item.type === "docSidebar" ? item.sidebarId : item.docId);

// The problems of a docusaurus `themeConfig` and the ids of the sidebars; empty when the navbar matches the sections.
export function problems(docsDir, themeConfig, sidebarIds) {
  const errors = [];
  const expected = sections(docsDir);
  const items = themeConfig.navbar.items.filter((item) => item.position === "left");
  for (const item of items) {
    if (item.type === "docSidebar") {
      if (!sidebarIds.includes(item.sidebarId)) errors.push(`navbar item "${item.label}" points to the missing sidebar ${item.sidebarId}`);
    } else if (item.type === "doc") {
      if (!docExists(docsDir, item.docId)) errors.push(`navbar item "${item.label}" points to the missing doc ${item.docId}`);
    } else {
      errors.push(`navbar item "${item.label}" is neither a sidebar nor a doc link`);
    }
  }
  for (const section of expected) {
    const item = items.find((i) => keyOf(i) === section.key);
    if (!item) {
      errors.push(`the section "${section.label}" (${section.key}) has no navbar entry`);
      continue;
    }
    const type = section.kind === "sidebar" ? "docSidebar" : "doc";
    if (item.type !== type) errors.push(`the navbar entry "${item.label}" must be a ${type} item`);
    if (item.label !== section.label) errors.push(`navbar label "${item.label}" differs from the section label "${section.label}"`);
  }
  for (const id of sidebarIds) {
    if (!expected.some((s) => s.kind === "sidebar" && s.key === id)) errors.push(`the sidebar ${id} is no section folder`);
  }
  const order = items.map(keyOf).filter((key) => expected.some((s) => s.key === key));
  if (order.join() !== expected.map((s) => s.key).filter((key) => order.includes(key)).join()) errors.push(`navbar order ${order.join(", ")} differs from the section order`);
  for (const entry of readdirSync(docsDir, { withFileTypes: true })) {
    if (!entry.isDirectory() || existsSync(join(docsDir, entry.name, "_category_.json"))) continue;
    const pages = readdirSync(join(docsDir, entry.name)).filter(isPage);
    if (pages.filter((p) => !isIndex(p)).length > 0) errors.push(`${entry.name}/ has pages besides its index and no _category_.json: they are in no sidebar`);
  }
  return errors;
}
