// Full-page screenshots of the Performance page with headless Chromium driven over the DevTools protocol
// (CHROMIUM=/path/to/chromium; default `chromium`), against the production build (`bun run build`) served here.
// Also opens every docs page that embeds an architecture diagram at 1440, 1100 and 500 px and fails (exit 1) when the
// page scrolls sideways or an element of the article is wider than the article column (a clipped diagram).
// Usage: node scripts/screenshots.mjs <outdir>
import { spawn } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const out = process.argv[2] ?? "screenshots";
const chromium = process.env.CHROMIUM ?? "chromium";
const sitePort = 3998;
const debugPort = 9333;
const page = `http://127.0.0.1:${sitePort}/docs/performance`;


// [name, query, script run once the page has rendered, variants]
const shots = [
  ["default", "", null, ["dark", "light", "narrow", "medium"]],
  ["another-model", "?model=qwen3-4b", null, ["dark", "narrow"]],
  ["rtx-pro-6000", "?hw=nvidia-rtx-pro-6000-blackwell", null, ["dark", "narrow"]],
  ["compare", "?compare=1&hw=amd-radeon-8060s-graphics&model=qwen3-4b&hw2=nvidia-rtx-pro-6000-blackwell&model2=qwen3.8-flash-next-q2", null, ["dark", "narrow"]],
];
const variants = { dark: [1440, "dark"], light: [1440, "light"], medium: [1100, "dark"], narrow: [500, "dark"] };

// Pages with an architecture diagram component, and the widths each is checked at.
const diagramPages = ["architecture", "architecture/graph-ir", "architecture/kernel-import", "architecture/fusion", "architecture/attention", "architecture/serving-design", "architecture/glossary", "architecture/execution", "architecture/driver", "architecture/registry", "architecture/backends", "architecture/implementation", "architecture/recurrence-scan"];
const diagramWidths = [1440, 1100, 500];
// Elements of the article (excluding ones inside their own scrolling box) whose box leaves the article column.
const overflowProbe = `(() => {
  const root = document.scrollingElement;
  const article = document.querySelector("article .markdown");
  const column = article.getBoundingClientRect();
  const scrolls = (el) => { for (let p = el.parentElement; p && p !== article; p = p.parentElement) { const o = getComputedStyle(p).overflowX; if (o === "auto" || o === "scroll" || o === "hidden") return true; } return false; };
  const wide = [...article.querySelectorAll("*")].filter((el) => { const r = el.getBoundingClientRect(); return r.width > 0 && (r.left < column.left - 1 || r.right > column.right + 1) && !scrolls(el); }).slice(0, 5).map((el) => el.tagName + "." + String(el.className).slice(0, 40));
  return { page: root.scrollWidth - root.clientWidth, wide };
})()`;

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

async function connect(url) {
  const target = await (await fetch(`http://127.0.0.1:${debugPort}/json/new?about:blank`, { method: "PUT" })).json();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((resolve) => ws.addEventListener("open", resolve, { once: true }));
  let id = 0;
  const pending = new Map();
  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    pending.get(message.id)?.(message.result);
  });
  const send = (method, params = {}) => new Promise((resolve) => ((pending.set(++id, resolve)), ws.send(JSON.stringify({ id, method, params }))));
  return { send, close: () => ws.close(), id: target.id, url };
}

const server = spawn("bun", ["run", "serve", "--", "--port", String(sitePort), "--no-open"], { stdio: "ignore" });
const profile = mkdtempSync(join(tmpdir(), "poot-site-shots-"));
const browser = spawn(chromium, ["--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", `--user-data-dir=${profile}`, `--remote-debugging-port=${debugPort}`, "about:blank"], { stdio: "ignore" });
mkdirSync(out, { recursive: true });
try {
  await sleep(5000);
  for (const [name, query, script, which] of shots) {
    for (const variant of which) {
      const [width, theme] = variants[variant];
      const url = `${page}${query}${query ? "&" : "?"}docusaurus-theme=${theme}`;
      const tab = await connect(url);
      await tab.send("Emulation.setDeviceMetricsOverride", { width, height: 900, deviceScaleFactor: 1, mobile: false });
      await tab.send("Page.navigate", { url });
      await sleep(4000);
      if (script) {
        await tab.send("Runtime.evaluate", { expression: script });
        await sleep(500);
      }
      const metrics = await tab.send("Page.getLayoutMetrics");
      const height = Math.ceil(metrics.cssContentSize.height);
      const shot = await tab.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: 0, y: 0, width, height, scale: 1 } });
      const file = join(out, `${name}.${variant}.png`);
      writeFileSync(file, Buffer.from(shot.data, "base64"));
      console.log(`ok   ${file}`);
      tab.close();
      await fetch(`http://127.0.0.1:${debugPort}/json/close/${tab.id}`);
    }
  }
  let overflowing = 0;
  for (const doc of diagramPages) {
    for (const width of diagramWidths) {
      const tab = await connect(`http://127.0.0.1:${sitePort}/docs/${doc}`);
      await tab.send("Emulation.setDeviceMetricsOverride", { width, height: 900, deviceScaleFactor: 1, mobile: false });
      await tab.send("Page.navigate", { url: `http://127.0.0.1:${sitePort}/docs/${doc}` });
      await sleep(3000);
      const probe = await tab.send("Runtime.evaluate", { expression: overflowProbe, returnByValue: true });
      const { page: sideways, wide } = probe.result?.value ?? { page: 0, wide: [`probe failed: ${JSON.stringify(probe.exceptionDetails?.exception?.description)}`] };
      const bad = sideways > 0 || wide.length > 0;
      if (bad) overflowing++;
      console.log(`${bad ? "FAIL" : "ok  "} ${doc} @${width}px${bad ? ` page scrolls ${sideways}px sideways; wider than the article: ${wide.join(", ") || "none"}` : ""}`);
      if (process.env.DIAGRAM_SHOTS) {
        const metrics = await tab.send("Page.getLayoutMetrics");
        const shot = await tab.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: true, clip: { x: 0, y: 0, width, height: Math.ceil(metrics.cssContentSize.height), scale: 1 } });
        writeFileSync(join(out, `${doc.replaceAll("/", "-")}.${width}.png`), Buffer.from(shot.data, "base64"));
      }
      tab.close();
      await fetch(`http://127.0.0.1:${debugPort}/json/close/${tab.id}`);
    }
  }
  if (overflowing > 0) process.exitCode = 1;
} finally {
  browser.kill();
  server.kill();
}
