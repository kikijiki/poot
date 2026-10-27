// Screenshots of every view (dark, light, narrow; headless Chromium lays out no narrower than 500 px) against the fixture API, with headless Chromium
// (CHROMIUM=/path/to/chromium; default `chromium`). Usage: node scripts/screens.mjs <outdir>
// Build the page first (`npm run build`).
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {mkdirSync, mkdtempSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {startFixtureServer} from './fixture-server.mjs';

const run = promisify(execFile);
const out = process.argv[2] ?? 'screens';
const chromium = process.env.CHROMIUM ?? 'chromium';
const port = 8801;
const done = 'run-1792790000-444444';
const errors = 'run-1792789000-555555';
const exec = 'exec-1792798000-333333333-1690479-0';

const shots = [
  ['overview', '?view=overview'],
  ['runs-list', '?view=runs'],
  ['run-sweeping-summary', '?view=runs&run=run-1792800000-111111&tab=summary'],
  ['run-sweeping-results-live', '?view=runs&run=run-1792800000-111111&tab=results&sub=charts', 2000],
  ['run-waiting-summary', '?view=runs&run=run-1792799000-222222&tab=summary'],
  ['run-waiting-log', '?view=runs&run=run-1792799000-222222&tab=log'],
  ['run-done-results-charts', `?view=runs&run=${done}&tab=results&sub=charts`, 2400],
  ['run-done-results-table', `?view=runs&run=${done}&tab=results&sub=table`],
  ['run-errors-failures', `?view=runs&run=${errors}&tab=results&sub=failures`],
  ['run-incomplete-summary', '?view=runs&run=run-1792788000-666666&tab=summary'],
  ['run-failed-results-empty', '?view=runs&run=run-1792787000-777777&tab=results'],
  ['run-exec-summary', `?view=runs&run=${exec}&tab=summary`],
  ['run-exec-log', `?view=runs&run=${exec}&tab=log`],
  ['run-exec-pod', `?view=runs&run=${exec}&tab=pod`],
  ['pods', '?view=pods'],
  ['compare', `?view=compare&cmp=${done},${errors},run-1792800000-111111`, 2400],
  ['images', '?view=images&image=image-latest.log'],
  ['serve', '?view=serve'],
];

const server = await startFixtureServer(port);
const profile = mkdtempSync(join(tmpdir(), 'poot-screens-'));
mkdirSync(out, {recursive: true});
const variants = [['dark', 1440, 900, 'dark'], ['light', 1440, 900, 'light'], ['narrow', 500, 900, 'dark']];
for (const [name, url, height] of shots) {
  for (const [variant, w, h, theme] of variants) {
    if (variant !== 'dark' && !['overview', 'run-sweeping-results-live', 'run-done-results-charts', 'run-exec-summary', 'pods'].includes(name)) continue;
    const file = join(out, `${name}.${variant}.png`);
    // Asynchronous: the fixture server lives in this process and must keep answering while Chromium loads the page.
    const args = ['--headless=new', '--no-sandbox', '--disable-gpu', '--hide-scrollbars', `--user-data-dir=${profile}`, `--window-size=${w},${height ?? h}`, '--virtual-time-budget=8000', `--screenshot=${file}`, `http://127.0.0.1:${port}/${url}&__theme=${theme}`];
    try {
      await run(chromium, args, {timeout: 90_000});
      console.log(`ok   ${file}`);
    } catch (e) {
      console.log(`FAIL ${file}: ${e.stderr ?? e}`);
      process.exitCode = 1;
    }
  }
}
server.close();
