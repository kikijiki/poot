import { describe, expect, test } from "bun:test";
import { spawn } from "node:child_process";
import { chmodSync, mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { serveDirectory } from "./harness.mjs";

function site() {
  const dir = mkdtempSync(join(tmpdir(), "harness-"));
  writeFileSync(join(dir, "index.html"), "home");
  writeFileSync(join(dir, "404.html"), "missing");
  mkdirSync(join(dir, "docs"));
  writeFileSync(join(dir, "docs", "status.html"), "status");
  mkdirSync(join(dir, "docs", "serve"));
  writeFileSync(join(dir, "docs", "serve", "index.html"), "serve");
  return dir;
}

describe("serveDirectory", () => {
  test("two servers at once get different ports", async () => {
    const [a, b] = [await serveDirectory(site()), await serveDirectory(site())];
    try {
      expect(a.port).not.toBe(b.port);
    } finally {
      a.stop();
      b.stop();
    }
  });

  test("resolves clean URLs like the production host and 404s the rest", async () => {
    const server = await serveDirectory(site());
    try {
      const get = async (path) => {
        const res = await fetch(`http://127.0.0.1:${server.port}${path}`);
        return [res.status, await res.text()];
      };
      expect(await get("/")).toEqual([200, "home"]);
      expect(await get("/docs/status")).toEqual([200, "status"]);
      expect(await get("/docs/serve")).toEqual([200, "serve"]);
      expect(await get("/docs/nope")).toEqual([404, "missing"]);
    } finally {
      server.stop();
    }
  });

});

describe("startBrowser", () => {
  // A stand-in for chromium: it reports a DevTools port in the profile like the real one and then idles.
  function fakeChromium(pidFile) {
    const file = join(mkdtempSync(join(tmpdir(), "harness-bin-")), "chromium");
    writeFileSync(file, `#!/usr/bin/env bash
for arg in "$@"; do case "$arg" in --user-data-dir=*) profile="\${arg#--user-data-dir=}" ;; esac; done
printf '41234\\n/devtools/browser/x\\n' > "$profile/DevToolsActivePort"
echo $$ > ${pidFile}
exec sleep 600
`);
    chmodSync(file, 0o755);
    return file;
  }
  const alive = (pid) => {
    try {
      process.kill(pid, 0);
      return true;
    } catch {
      return false;
    }
  };

  for (const signal of ["SIGTERM", "SIGINT"]) {
    test(`${signal} to a run takes its browser down with it`, async () => {
      const pidFile = join(mkdtempSync(join(tmpdir(), "harness-pid-")), "pid");
      const harness = JSON.stringify(new URL("./harness.mjs", import.meta.url).href);
      const run = spawn("node", ["--input-type=module", "-e", `import { startBrowser } from ${harness};
        const b = await startBrowser(${JSON.stringify(fakeChromium(pidFile))}, "harness-profile-"); console.log(b.port); setInterval(() => {}, 1000);`], { stdio: ["ignore", "pipe", "inherit"] });
      const port = await new Promise((resolve) => run.stdout.once("data", (d) => resolve(Number(d.toString()))));
      expect(port).toBe(41234);
      const browser = Number(readFileSync(pidFile, "utf8"));
      expect(alive(browser)).toBe(true);
      const exited = new Promise((resolve) => run.once("exit", resolve));
      run.kill(signal);
      await exited;
      for (let wait = 0; wait < 40 && alive(browser); wait++) await new Promise((resolve) => setTimeout(resolve, 50)); // init reaps it
      expect(alive(browser)).toBe(false);
    });
  }
});
