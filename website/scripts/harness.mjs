// What the headless-browser scripts share: the built site served on an ephemeral port, and a Chromium on an ephemeral
// DevTools port. Neither port is fixed, so two audits (or an audit and a layout check) can run on one box at once.
// Everything started here is stopped when the process exits, including on SIGINT, SIGTERM and SIGHUP.
import { spawn } from "node:child_process";
import { existsSync, mkdtempSync, readFileSync, rmSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { extname, join, normalize, sep } from "node:path";

const cleanups = new Set();
let hooked = false;

// Run `stop` when the process ends, however it ends.
function onExit(stop) {
  if (!hooked) {
    hooked = true;
    process.on("exit", () => cleanups.forEach((run) => run()));
    for (const signal of ["SIGINT", "SIGTERM", "SIGHUP"]) process.on(signal, () => process.exit(128));
  }
  cleanups.add(stop);
  return () => {
    cleanups.delete(stop);
    stop();
  };
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const types = {
  ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml",
  ".png": "image/png", ".jpg": "image/jpeg", ".ico": "image/x-icon", ".woff2": "font/woff2", ".xml": "application/xml", ".txt": "text/plain",
};

// Serve a directory of static files on 127.0.0.1 at a port the OS picks. `/a/b` is `a/b.html` or `a/b/index.html`, like the
// production host; a missing file is the directory's 404.html. Returns { port, stop }.
export async function serveDirectory(root, { notFound = "404.html" } = {}) {
  const resolve = (url) => {
    const path = normalize(decodeURIComponent(url.split("?")[0].split("#")[0]));
    if (path.split(sep).includes("..")) return null;
    for (const candidate of [path, `${path}.html`, join(path, "index.html")]) {
      const file = join(root, candidate);
      if (existsSync(file) && statSync(file).isFile()) return file;
    }
    return null;
  };
  const server = createServer((req, res) => {
    const file = resolve(req.url ?? "/");
    const body = file ?? join(root, notFound);
    res.statusCode = file ? 200 : 404;
    if (!existsSync(body)) return void res.end();
    res.setHeader("content-type", types[extname(body)] ?? "application/octet-stream");
    res.end(readFileSync(body));
  });
  await new Promise((done, fail) => server.once("error", fail).listen(0, "127.0.0.1", done));
  const stop = onExit(() => {
    server.closeAllConnections();
    server.close();
  });
  return { port: server.address().port, stop };
}

// Start headless Chromium with the DevTools port chosen by the browser itself (it writes it to DevToolsActivePort in
// the profile). Returns { port, stop }.
export async function startBrowser(chromium, profilePrefix) {
  const profile = mkdtempSync(join(tmpdir(), profilePrefix));
  const browser = spawn(chromium, ["--headless=new", "--no-sandbox", "--disable-gpu", "--hide-scrollbars", `--user-data-dir=${profile}`, "--remote-debugging-port=0", "about:blank"], { stdio: "ignore" });
  const stop = onExit(() => {
    browser.kill();
    // Chromium's helper processes may still be writing into the profile for a moment; a leftover temp dir must not fail the run.
    try {
      rmSync(profile, { recursive: true, force: true, maxRetries: 20, retryDelay: 100 });
    } catch {}
  });
  const file = join(profile, "DevToolsActivePort");
  for (let attempt = 0; attempt < 120; attempt++) {
    if (browser.exitCode !== null) throw new Error(`${chromium} exited with ${browser.exitCode} before it opened a DevTools port`);
    if (existsSync(file)) {
      const port = Number(readFileSync(file, "utf8").split("\n")[0]);
      if (port) return { port, stop };
    }
    await sleep(250);
  }
  stop();
  throw new Error(`${chromium} did not open a DevTools port`);
}
