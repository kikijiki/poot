import assert from 'node:assert/strict';
import {test} from 'node:test';
import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {LOGFILES, NOW, OVERVIEW, PODS, RUNS, STATE} from './fixtures.ts';
import {PAGES} from './pages.tsx';
import {VIEWS} from './urlstate.ts';
import {Async, Empty} from './ui.tsx';
import Overview from './views/Overview.tsx';
import Pods from './views/Pods.tsx';
import {RunDetail} from './views/Runs.tsx';
import Runs from './views/Runs.tsx';
import Images from './views/Images.tsx';
import Compare from './views/Compare.tsx';
import {filterLines} from './views/LogPane.tsx';

const html = (node: React.ReactElement) => renderToStaticMarkup(node);
const noop = () => {};
const ready = <T,>(value: T) => ({kind: 'ready', value}) as const;
const run = (id: string) => RUNS.find((r) => r.id.startsWith(id))!;

test('every view has a page and every page a view', () => {
  assert.deepEqual(PAGES.map((p) => p.id), [...VIEWS]);
});

test('Async shows loading, then the error with a retry, then the content', () => {
  const loading = html(<Async what="pods" data={{kind: 'loading'}}>{() => <i>x</i>}</Async>);
  assert.match(loading, /Loading pods…/);
  const error = html(<Async what="pods" data={{kind: 'error', message: 'HTTP 500'}} onRetry={noop}>{() => <i>x</i>}</Async>);
  assert.match(error, /Could not load pods/);
  assert.match(error, /HTTP 500/);
  assert.match(error, /Retry/);
  assert.match(html(<Async what="pods" data={ready(1)}>{(n) => <i>got {n}</i>}</Async>), /got 1/);
});

test('the overview shows what runs, what waits, the spend and the failures with their reasons', () => {
  const out = html(<Overview state={ready(STATE)} overview={ready(OVERVIEW)} now={NOW} onOpen={noop} onRetry={noop} />);
  assert.match(out, /Running now/);
  assert.match(out, /decode-curve · qwen3-4b · RTX 3090/);
  assert.match(out, /Waiting for capacity/);
  assert.match(out, /\$0\.5\/h/); // the price cap of the waiting run
  assert.match(out, /\$18\.6/); // spend, last 24 h
  assert.match(out, /\$1\.61\/h/); // burn rate
  assert.match(out, /no instance has the CUDA floor 13\.0/); // a failure's reason
  assert.match(out, /missing at RunPod/); // the stale pod
});

test('the overview says why it is empty or broken: loading, error, no runs', () => {
  assert.match(html(<Overview state={{kind: 'loading'}} overview={{kind: 'loading'}} now={NOW} onOpen={noop} onRetry={noop} />), /Loading the overview/);
  assert.match(html(<Overview state={{kind: 'error', message: 'HTTP 500'}} overview={{kind: 'loading'}} now={NOW} onOpen={noop} onRetry={noop} />), /Could not load the overview/);
  const empty = html(<Overview state={ready({runs: [], pods: []})} overview={ready({...OVERVIEW, failures: [], spend_24h_usd: 0, burn_usd_per_hr: 0, billing_pods: 0})} now={NOW} onOpen={noop} onRetry={noop} />);
  assert.match(empty, /Nothing is running/);
  assert.match(empty, /Nothing is waiting/);
  assert.match(empty, /No failures/);
  assert.match(empty, /No finished runs/);
});

test('the overview still shows the runs when only the totals failed to load', () => {
  const out = html(<Overview state={ready(STATE)} overview={{kind: 'error', message: 'HTTP 500'}} now={NOW} onOpen={noop} onRetry={noop} />);
  assert.match(out, /Running now/);
  assert.match(out, /Could not load failures/);
});

test('an exec run has no Results tab; a sweep has one', () => {
  const props = {pods: PODS, logfiles: LOGFILES, tab: 'summary', sub: 'charts', now: NOW, onTab: noop, onSub: noop} as const;
  const exec = html(<RunDetail run={run('exec-1792798000')} {...props} />);
  assert.doesNotMatch(exec, />Results</);
  assert.match(exec, />Log</);
  assert.match(html(<RunDetail run={run('run-1792790000')} {...props} />), />Results</);
});

test('a link to the Results tab of an exec run shows its summary, not an empty page', () => {
  const out = html(<RunDetail run={run('exec-1792798000')} pods={PODS} logfiles={LOGFILES} tab="results" sub="charts" now={NOW} onTab={noop} onSub={noop} />);
  assert.match(out, /Command/);
  assert.doesNotMatch(out, /No result sets|Loading results/);
});

test('a run waiting for capacity shows the pass, the minutes and the price cap on its summary', () => {
  const out = html(<RunDetail run={run('run-1792799000')} pods={PODS} logfiles={LOGFILES} tab="summary" sub="charts" now={NOW} onTab={noop} onSub={noop} />);
  assert.match(out, /pass 4, 85 min so far/);
  assert.match(out, /price cap \$0\.5\/h/);
});

test('an incomplete run says which models are short of sets', () => {
  const out = html(<RunDetail run={run('run-1792788000')} pods={PODS} logfiles={LOGFILES} tab="summary" sub="charts" now={NOW} onTab={noop} onSub={noop} />);
  assert.match(out, /Incomplete/);
  assert.match(out, /qwen3-32b 0\/3/);
});

test('the run list says when there are no runs, no match, or a run that is not in the list', () => {
  const props = {pods: [], logfiles: [], selected: null, tab: 'summary', sub: 'charts', now: NOW, onSelect: noop, onTab: noop, onSub: noop} as const;
  assert.match(html(<Runs runs={[]} {...props} />), /No runs yet/);
  assert.match(html(<Runs runs={RUNS} {...props} selected="run-nope" />), /Run not found/);
  assert.match(html(<Runs runs={RUNS} {...props} />), /Select a run/);
});

test('pods lists live pods first and folds the terminated ones', () => {
  const out = html(<Pods state={ready(STATE)} now={NOW} onOpen={noop} onRetry={noop} />);
  assert.match(out, /Live \(4\)/);
  assert.match(out, /Terminated \(3\)/);
  assert.match(out, /does not list it/);
  assert.match(out, /not checked: RUNPOD_API_KEY is not set/);
  assert.match(html(<Pods state={ready({runs: [], pods: []})} now={NOW} onOpen={noop} onRetry={noop} />), /No live pods/);
});

test('the images view and the compare view have empty states', () => {
  assert.match(html(<Images logfiles={[]} selected={null} now={NOW} onSelect={noop} onTriggered={noop} />), /No image builds yet/);
  assert.match(html(<Compare state={ready({runs: [], pods: []})} selected={[]} now={NOW} onSelect={noop} onRetry={noop} />), /Nothing to compare yet/);
  assert.match(html(<Compare state={ready(STATE)} selected={[]} now={NOW} onSelect={noop} onRetry={noop} />), /No run picked/);
});

test('Empty renders its title and detail', () => {
  assert.match(html(<Empty title="Nothing">because</Empty>), /Nothing.*because/);
});

test('the log search keeps the matching lines, case-insensitively', () => {
  const text = 'a line\nERROR boom\nanother\nerror again';
  assert.deepEqual(filterLines(text, 'error'), ['ERROR boom', 'error again']);
  assert.equal(filterLines(text, '').length, 4);
  assert.deepEqual(filterLines(text, 'nope'), []);
});
