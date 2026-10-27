import assert from 'node:assert/strict';
import {test} from 'node:test';
import {OVERVIEW, PODS, RUNS, STATE} from './fixtures.ts';
import {bucketRuns, filterRuns, formatAgo, formatDuration, formatUsd, noSetsMessage, podPill, runPill, runTabs, runTitle, shortGpu, tabFor} from './model.ts';

const run = (id: string) => RUNS.find((r) => r.id.startsWith(id))!;
const pod = (id: string) => PODS.find((p) => p.id === id)!;

test('a finished run with errors is amber, an all-failed one red, never green', () => {
  assert.deepEqual(runPill(run('run-1792789000')), {label: 'done with errors (3 of 20 cells failed)', tone: 'warn'});
  assert.equal(runPill(run('run-1792790000')).tone, 'ok');
  assert.equal(runPill({...run('run-1792789000'), outcome: {kind: 'failed', label: 'done: all cells failed', failed: 3, total: 3}}).tone, 'fail');
});

test('an incomplete run keeps its text and is prominent (amber)', () => {
  assert.deepEqual(runPill(run('run-1792788000')), {label: 'incomplete: qwen3-32b 0/3', tone: 'warn'});
});

test('a run waiting for capacity says so, a sweeping one is running, a failed one is red', () => {
  assert.deepEqual(runPill(run('run-1792799000')), {label: 'waiting for capacity', tone: 'warn'});
  assert.deepEqual(runPill(run('run-1792800000')), {label: 'sweeping', tone: 'run'});
  assert.deepEqual(runPill(run('run-1792787000')), {label: 'failed', tone: 'fail'});
  assert.equal(runPill(run('run-1792785000')).tone, 'idle');
});

test('an exec run has no Results tab, and a request for it falls back to the summary', () => {
  const exec = run('exec-1792798000');
  assert.deepEqual(runTabs(exec), ['summary', 'log', 'pod']);
  assert.equal(tabFor(exec, 'results'), 'summary');
  assert.deepEqual(runTabs(run('run-1792790000')), ['summary', 'log', 'results', 'pod']);
  assert.equal(tabFor(run('run-1792790000'), 'results'), 'results');
});

test('a run is named by what it ran, not by its id', () => {
  assert.equal(runTitle(run('run-1792800000'), PODS), 'decode-curve · qwen3-4b · RTX 3090');
  assert.equal(runTitle(run('run-1792799000'), PODS), 'decode-curve · qwen3-8b · RTX 4090');
  assert.match(runTitle(run('exec-1792798000'), PODS), /^exec · bash \/root\/pod\.sh/);
  assert.equal(shortGpu('NVIDIA RTX PRO 6000 Blackwell Server Edition'), 'RTX PRO 6000 Blackwell Server');
});

test('a pod RunPod does not list is not shown as live, whatever the DB says', () => {
  assert.deepEqual(podPill(pod('podStale')), {label: 'missing at RunPod (DB: provisioning)', tone: 'fail'});
  assert.deepEqual(podPill(pod('podNoKey')), {label: 'idle · unverified', tone: 'idle'});
  assert.deepEqual(podPill(pod('pod3090busy')), {label: 'busy', tone: 'ok'});
  assert.deepEqual(podPill(pod('podH200')), {label: 'terminated', tone: 'idle'});
});

test('runs are bucketed by what they are doing now', () => {
  const b = bucketRuns(STATE);
  assert.deepEqual(b.running.map((r) => r.id.slice(0, 14)), ['run-1792800000', 'exec-179279800']);
  assert.deepEqual(b.waiting.map((r) => r.id.slice(0, 14)), ['run-1792799000']);
  assert.equal(b.recent.length, 6);
});

test('the filter matches text, kind and problems', () => {
  const all = {text: '', kind: 'all', show: 'all'} as const;
  assert.equal(filterRuns(RUNS, PODS, all).length, RUNS.length);
  assert.equal(filterRuns(RUNS, PODS, {...all, kind: 'exec'}).length, 2);
  assert.equal(filterRuns(RUNS, PODS, {...all, text: 'qwen3-8b'}).length, 1);
  assert.deepEqual(filterRuns(RUNS, PODS, {...all, show: 'problems'}).map((r) => r.id.slice(0, 14)), ['run-1792789000', 'run-1792788000', 'run-1792787000', 'exec-179278600']);
  assert.equal(filterRuns(RUNS, PODS, {...all, show: 'active'}).length, 3);
});

test('every reason a run has no sets has its own words', () => {
  const reasons = ['exec_run', 'no_results_dir', 'no_sets_yet', 'ended_without_sets', 'unknown_run'] as const;
  const texts = new Set(reasons.map(noSetsMessage));
  assert.equal(texts.size, reasons.length);
  assert.match(noSetsMessage('exec_run'), /exec run/);
});

test('durations, ages and prices read as people write them', () => {
  assert.equal(formatDuration(45), '45s');
  assert.equal(formatDuration(725), '12m 05s');
  assert.equal(formatDuration(7500), '2h 05m');
  assert.equal(formatAgo('2026-10-24T11:55:00Z', Date.parse('2026-10-24T12:00:00Z')), '5m ago');
  assert.equal(formatUsd(0.349), '$0.35');
  assert.equal(formatUsd(null), '—');
  assert.equal(formatUsd(-0), '$0.00');
  assert.equal(formatUsd(-0.001), '$0.00');
  assert.equal(OVERVIEW.failures.length, 2);
});
