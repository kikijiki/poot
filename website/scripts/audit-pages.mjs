// Audits every page of the built site (website/build/sitemap.xml) in headless Chromium driven over the DevTools protocol, at 1440 and 500 px,
// and fails with page + selector for the layout problems a reader sees: overflow, scrolling or unhighlighted code, scrolling tables, walls of text,
// heading skips, unlabelled images and diagrams, console errors. Needs `bun run build` first (CHROMIUM=/path/to/chromium; default `chromium`).
// Usage: node scripts/audit-pages.mjs [--lanes] [--only <path-substring>]   audit the built site
//        node scripts/audit-pages.mjs --self-test                            each rule must fail on its fixture in scripts/audit-fixtures
import { spawn } from "node:child_process";
import { readFileSync, readdirSync, mkdtempSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const args = process.argv.slice(2);
const selfTest = args.includes("--self-test");
const byLane = args.includes("--lanes");
const only = args.includes("--only") ? args[args.indexOf("--only") + 1] : null;
const chromium = process.env.CHROMIUM ?? "chromium";
const sitePort = 3996;
const debugPort = 9337;
const viewports = [[1440, 1440], [500, 500]];
const MAX_PARAGRAPH_LINES = 9;
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

// Which lane owns a page, for the failure list. First match wins.
const lanes = [
  ["c1364 (supported-models, build, test)", /\/develop\/(supported-models|build-and-test)$/],
  ["c1365 (architecture A)", /\/architecture\/(attention|npu|execution|tracing|fusion)$/],
  ["c1366 (architecture B, home)", /^(\/|\/docs|\/docs\/architecture(\/.*)?)$/],
  ["c1367 (serve, develop, performance)", /^\/docs\/(serve|develop|performance)(\/.*)?$/],
];
const laneOf = (path) => lanes.find(([, re]) => re.test(path))?.[0] ?? "unowned";

// The rules run inside the page. They return [{ rule, selector, detail }].
const pageChecks = String.raw`(() => {
  const found = [];
  const fail = (rule, el, detail) => {
    const id = el.id ? "#" + el.id : "";
    const cls = typeof el.className === "string" && el.className ? "." + el.className.trim().split(/\s+/).slice(0, 2).join(".") : "";
    const text = (el.textContent || "").trim().replace(/\s+/g, " ").slice(0, 40);
    found.push({ rule, selector: el.tagName.toLowerCase() + id + cls + (text ? ' "' + text + '"' : ""), detail });
  };
  const root = document.querySelector("article .markdown, .markdown") || document.querySelector("main") || document.body;
  const column = root.getBoundingClientRect();
  const sidebar = document.querySelector(".theme-doc-sidebar-container");
  const toc = document.querySelector(".theme-doc-toc-desktop");
  const w = document.documentElement.clientWidth;

  // (a) horizontal overflow
  if (document.documentElement.scrollWidth > w) fail("overflow", document.documentElement, "document scrollWidth " + document.documentElement.scrollWidth + " > " + w);
  for (const el of root.querySelectorAll("pre, table, svg, img, canvas, [class*=Diagram], [class*=timeline]")) {
    // A table that scrolls inside its frame (.table-scroll) is measured by the frame.
    const r = (el.tagName === "TABLE" ? el.closest(".table-scroll") || el : el).getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (r.right > column.right + 1 || r.left < column.left - 1) fail("overflow", el, "box " + Math.round(r.left) + ".." + Math.round(r.right) + " outside the content column " + Math.round(column.left) + ".." + Math.round(column.right));
    for (const side of [sidebar, toc]) {
      const s = side?.getBoundingClientRect();
      if (s && s.width > 0 && r.left < s.right - 1 && r.right > s.left + 1) fail("overflow", el, "overlaps the sidebar or outline");
    }
  }

  // (b) code blocks: scrolling and highlighting
  for (const pre of root.querySelectorAll("pre")) {
    if (pre.scrollWidth > pre.clientWidth + 1) fail("code-scrolls", pre, "scrollWidth " + pre.scrollWidth + " > " + pre.clientWidth);
    const code = pre.querySelector("code") || pre;
    const lang = ((pre.className + " " + (pre.parentElement?.className || "") + " " + code.className).match(/language-([\w-]+)/) || [])[1];
    if (!lang) fail("code-unhighlighted", pre, "no language");
    else if (!["text", "plaintext", "txt"].includes(lang) && !pre.querySelector(".token")) fail("code-unhighlighted", pre, "language " + lang + " has no .token spans");
  }

  // (c) tables that scroll: inside a frame with the shadow cue is fine; a table that scrolls by itself, or a frame
  // that scrolls with no cue, is not
  for (const t of root.querySelectorAll("table")) {
    const frame = t.closest(".table-scroll");
    if (!frame) {
      if (t.scrollWidth > t.clientWidth + 1) fail("table-scrolls", t, "scrollWidth " + t.scrollWidth + " > " + t.clientWidth);
    } else if (frame.scrollWidth > frame.clientWidth + 1 && !getComputedStyle(frame).backgroundImage.includes("gradient")) {
      fail("table-no-cue", frame, "scrolls (" + frame.scrollWidth + " > " + frame.clientWidth + ") with no scroll shadow");
    }
  }

  // (d) walls of text
  for (const p of root.querySelectorAll("p")) {
    const cs = getComputedStyle(p);
    const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.5;
    const lines = p.getBoundingClientRect().height / lh;
    if (lines > ${MAX_PARAGRAPH_LINES} + 0.5) fail("wall-of-text", p, Math.round(lines) + " lines");
  }

  // (e) heading levels
  let prev = 0;
  for (const h of root.querySelectorAll("h1, h2, h3, h4, h5, h6")) {
    const level = Number(h.tagName[1]);
    if (prev && level > prev + 1) fail("heading-skip", h, "h" + prev + " -> h" + level);
    prev = level;
  }

  // (f) images and diagrams without a label
  for (const img of root.querySelectorAll("img")) {
    if (!(img.getAttribute("alt") || "").trim()) fail("unlabelled", img, "img without alt");
  }
  for (const svg of root.querySelectorAll("svg")) {
    const r = svg.getBoundingClientRect();
    if (r.width < 100 || svg.closest("button, a, summary")) continue;
    const labelled = svg.getAttribute("aria-label") || svg.getAttribute("aria-labelledby") || svg.querySelector("title") || svg.getAttribute("aria-hidden") === "true" || svg.closest("[aria-label], [role=img]");
    if (!labelled) fail("unlabelled", svg, "svg without aria-label or title");
  }
  return found;
})()`;

async function connect(browserPort) {
  const target = await (await fetch(`http://127.0.0.1:${browserPort}/json/new?about:blank`, { method: "PUT" })).json();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve) => ws.addEventListener("open", resolve, { once: true }));
  let id = 0;
  const pending = new Map();
  const consoleErrors = [];
  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (message.id) pending.get(message.id)?.(message.result);
    else if (message.method === "Runtime.exceptionThrown") consoleErrors.push(message.params.exceptionDetails.exception?.description ?? message.params.exceptionDetails.text);
    else if (message.method === "Runtime.consoleAPICalled" && message.params.type === "error") consoleErrors.push(message.params.args.map((a) => a.value ?? a.description).join(" "));
    else if (message.method === "Log.entryAdded" && message.params.entry.level === "error") consoleErrors.push(`${message.params.entry.text} ${message.params.entry.url ?? ""}`);
  });
  const send = (method, params = {}) => new Promise((resolve) => (pending.set(++id, resolve), ws.send(JSON.stringify({ id, method, params }))));
  await send("Runtime.enable");
  await send("Log.enable");
  return { send, consoleErrors, close: async () => { ws.close(); await fetch(`http://127.0.0.1:${browserPort}/json/close/${target.id}`); } };
}

async function auditPage(base, path) {
  const failures = [];
  for (const [width, label] of viewports) {
    const tab = await connect(debugPort);
    await tab.send("Emulation.setDeviceMetricsOverride", { width, height: 900, deviceScaleFactor: 1, mobile: false });
    await tab.send("Page.navigate", { url: `${base}${path}?docusaurus-theme=dark` });
    await sleep(3000);
    const result = await tab.send("Runtime.evaluate", { expression: pageChecks, returnByValue: true });
    for (const f of result?.result?.value ?? []) {
      // Code and tables are checked for sideways scrolling at the desktop width only: at 500 px an 86-column line cannot fit.
      // A real code block may scroll at 500 px: it is listed for review, and does not fail.
      if (width < 1000 && f.rule === "code-scrolls") {
        failures.push({ width: label, ...f, info: true });
        continue;
      }
      if (width < 1000 && (f.rule === "table-scrolls" || f.rule === "wall-of-text")) continue;
      failures.push({ width: label, ...f });
    }
    if (width === 1440) for (const message of tab.consoleErrors) failures.push({ width: label, rule: "console-error", selector: "console", detail: String(message).slice(0, 160) });
    await tab.close();
  }
  return failures;
}

async function withBrowser(run) {
  const profile = mkdtempSync(join(tmpdir(), "poot-site-audit-"));
  const browser = spawn(chromium, ["--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", `--user-data-dir=${profile}`, `--remote-debugging-port=${debugPort}`, "about:blank"], { stdio: "ignore" });
  try {
    await sleep(4000);
    return await run();
  } finally {
    browser.kill();
  }
}

function report(results) {
  const rules = ["overflow", "code-scrolls", "code-unhighlighted", "table-scrolls", "table-no-cue", "wall-of-text", "heading-skip", "unlabelled", "console-error"];
  console.log(["page".padEnd(46), ...rules.map((r) => r.slice(0, 9).padEnd(9))].join(" "));
  for (const [path, failures] of results) {
    const counts = rules.map((r) => String(failures.filter((f) => f.rule === r).length || "-").padEnd(9));
    console.log([path.padEnd(46), ...counts].join(" "));
  }
  const review = results.flatMap(([path, f]) => f.filter((x) => x.info).map((x) => [path, x]));
  if (review.length) {
    console.log("\nCode blocks that scroll at 500 px (real code; for review, not failures):");
    for (const [path, f] of review) console.log(`  ${path}: ${f.selector} (${f.detail})`);
  }
  for (const [, f] of results) f.splice(0, f.length, ...f.filter((x) => !x.info));
  const failing = results.filter(([, f]) => f.length);
  const order = byLane ? [...new Set(failing.map(([p]) => laneOf(p)))].sort() : [null];
  for (const lane of order) {
    if (lane) console.log(`\n== ${lane}`);
    for (const [path, failures] of failing.filter(([p]) => !lane || laneOf(p) === lane)) {
      console.log(`\n${path}`);
      for (const f of failures) console.log(`  [${f.width}] ${f.rule}: ${f.selector} (${f.detail})`);
    }
  }
  console.log(`\n${failing.length} of ${results.length} pages fail`);
  return failing.length === 0;
}

if (selfTest) {
  // Each fixture names the one rule it must trip; ok.html must be clean.
  const dir = join(here, "audit-fixtures");
  const server = createServer((req, res) => {
    if (req.url === "/favicon.ico") {
      res.statusCode = 204;
      res.end();
      return;
    }
    try {
      res.setHeader("content-type", "text/html");
      res.end(readFileSync(join(dir, req.url.split("?")[0].replace(/^\//, ""))));
    } catch {
      res.statusCode = 404;
      res.end();
    }
  }).listen(sitePort);
  let ok = true;
  await withBrowser(async () => {
    for (const file of readdirSync(dir).sort()) {
      const expected = file.replace(/\.html$/, "");
      const failures = await auditPage(`http://127.0.0.1:${sitePort}`, `/${file}`);
      const rules = new Set(failures.map((f) => f.rule));
      const pass = expected === "ok" ? rules.size === 0 : rules.has(expected);
      console.log(`${pass ? "ok  " : "FAIL"} ${file}: ${[...rules].join(", ") || "clean"}`);
      if (!pass) for (const f of failures) console.log(`       [${f.width}] ${f.rule}: ${f.selector} (${f.detail})`);
      ok &&= pass;
    }
  });
  server.close();
  process.exit(ok ? 0 : 1);
}

const sitemap = readFileSync(join(here, "..", "build", "sitemap.xml"), "utf8");
const paths = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => new URL(m[1]).pathname.replace(/(.)\/$/, "$1")).filter((p) => p !== "/search" && (!only || p.includes(only)));
const server = spawn("bun", ["run", "serve", "--", "--port", String(sitePort), "--no-open"], { stdio: "ignore", cwd: join(here, "..") });
let ok = false;
try {
  await sleep(3000);
  const results = await withBrowser(async () => {
    const out = [];
    for (const path of paths) out.push([path, await auditPage(`http://127.0.0.1:${sitePort}`, path)]);
    return out;
  });
  ok = report(results);
} finally {
  server.kill();
}
process.exit(ok ? 0 : 1);
