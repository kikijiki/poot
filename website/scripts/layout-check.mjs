// Layout check of a few docs pages with headless Chromium over the DevTools protocol, against the production build
// (`bun run build`). For each page and width it scrolls to the bottom and asserts: the page does not overflow
// horizontally; a page with a sidebar still shows sidebar items at the bottom, and a single page has none; every
// markdown table fills its frame and stays inside the content column. It saves a screenshot at the top and one
// scrolled to the bottom. Usage: CHROMIUM=/path/to/chromium bun scripts/layout-check.mjs <outdir>
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const out = process.argv[2] ?? "screenshots/layout";
const chromium = process.env.CHROMIUM ?? "chromium";
const sitePort = 3996;
const debugPort = 9333;
// [path, has a sidebar]
const pages = [
  ["performance", false],
  ["status", false],
  ["develop/supported-models", true],
  ["serve/api", true],
  ["architecture/backends", true],
];
const widths = [1440, 1320, 1100, 500];
const height = 900;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const server = spawn("bun", ["run", "serve", "--", "--port", String(sitePort), "--no-open"], { stdio: "ignore" });
const browser = spawn(chromium, ["--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", `--remote-debugging-port=${debugPort}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), "poot-layout-"))}`, "about:blank"], { stdio: "ignore" });
mkdirSync(out, { recursive: true });

async function connect() {
  for (let attempt = 0; attempt < 60; attempt++) {
    try {
      const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json`)).json();
      const target = targets.find((t) => t.type === "page");
      if (target) return new WebSocket(target.webSocketDebuggerUrl);
    } catch {}
    await sleep(500);
  }
  throw new Error("chromium did not start");
}

const socket = await connect();
await new Promise((resolve) => socket.addEventListener("open", resolve));
let nextId = 0;
const pending = new Map();
socket.addEventListener("message", (event) => {
  const message = JSON.parse(event.data);
  pending.get(message.id)?.(message);
});
const call = (method, params = {}) =>
  new Promise((resolve) => {
    const id = ++nextId;
    pending.set(id, resolve);
    socket.send(JSON.stringify({ id, method, params }));
  });
const evaluate = async (expression) => (await call("Runtime.evaluate", { expression, returnByValue: true })).result.result.value;

const measure = `(() => {
  document.querySelectorAll("details").forEach((d) => (d.open = true));
  window.scrollTo(0, document.documentElement.scrollHeight);
  const sidebar = document.querySelector(".theme-doc-sidebar-container");
  const items = sidebar ? [...sidebar.querySelectorAll(".menu__link")].filter((a) => { const r = a.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight; }) : [];
  const column = document.querySelector("main").getBoundingClientRect();
  const tables = [...document.querySelectorAll(".table-scroll")].map((frame) => {
    const table = frame.querySelector("table").getBoundingClientRect();
    const box = frame.getBoundingClientRect();
    return { frame: Math.round(box.width), table: Math.round(table.width), fills: table.width >= frame.clientWidth - 1, inside: box.right <= column.right + 1 };
  });
  return { scrollWidth: document.documentElement.scrollWidth, clientWidth: document.documentElement.clientWidth, hasSidebar: !!sidebar, sidebarItemsInView: items.length, tables };
})()`;

let failed = false;
try {
  await sleep(3000);
  for (const [path, withSidebar] of pages) {
    for (const width of widths) {
      await call("Emulation.setDeviceMetricsOverride", { width, height, deviceScaleFactor: 1, mobile: false });
      await call("Page.navigate", { url: `http://127.0.0.1:${sitePort}/docs/${path}?docusaurus-theme=light` });
      await sleep(4000);
      const name = `${path.replaceAll("/", "-")}.${width}`;
      const top = await call("Page.captureScreenshot", { format: "png" });
      writeFileSync(join(out, `${name}.top.png`), Buffer.from(top.result.data, "base64"));
      const result = await evaluate(measure);
      await sleep(500);
      const bottom = await call("Page.captureScreenshot", { format: "png" });
      writeFileSync(join(out, `${name}.bottom.png`), Buffer.from(bottom.result.data, "base64"));
      const wide = width >= 997; // below this Docusaurus hides the sidebar behind the menu button
      const problems = [];
      if (result.scrollWidth > result.clientWidth) problems.push(`the page overflows horizontally (${result.scrollWidth} > ${result.clientWidth})`);
      if (withSidebar && wide && result.sidebarItemsInView === 0) problems.push("the sidebar has no visible items at the bottom");
      if (!withSidebar && result.hasSidebar) problems.push("a single page has a sidebar");
      result.tables.forEach((t, i) => {
        if (!t.fills) problems.push(`table ${i} (${t.table}px) does not fill its frame (${t.frame}px)`);
        if (!t.inside) problems.push(`table ${i} frame leaves the content column`);
      });
      console.log(`${problems.length ? "FAIL" : "ok  "} ${path} ${width}px tables=${result.tables.length}${problems.length ? ": " + problems.join("; ") : ""}`);
      if (problems.length) failed = true;
    }
  }
} finally {
  browser.kill();
  server.kill();
}
process.exit(failed ? 1 : 0);
