// Runs every src/**/*.test.ts(x): esbuild bundles each (the component tests import TSX and CSS modules, which node
// cannot load itself) into .test-build/, then node's own test runner runs the bundles.
import {spawnSync} from 'node:child_process';
import {readdirSync, rmSync, statSync} from 'node:fs';
import {join} from 'node:path';
import {build} from 'esbuild';

function testFiles(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return testFiles(path);
    return /\.test\.(tsx?|mjs)$/.test(name) ? [path] : [];
  });
}

const out = '.test-build';
rmSync(out, {recursive: true, force: true});
await build({
  entryPoints: testFiles('src'),
  outdir: out,
  outbase: 'src',
  bundle: true,
  platform: 'node',
  format: 'esm',
  jsx: 'automatic',
  loader: {'.css': 'text'},
  outExtension: {'.js': '.mjs'},
  banner: {js: "import {createRequire as __cr} from 'node:module'; const require = __cr(import.meta.url);"},
  logLevel: 'warning',
});
const run = spawnSync(process.execPath, ['--test', ...testFiles(out)], {stdio: 'inherit'});
rmSync(out, {recursive: true, force: true});
process.exit(run.status ?? 1);
