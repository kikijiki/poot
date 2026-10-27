// What the pages show of a run or a pod, derived from the API types as pure functions (no React, no clock reads:
// `now` is passed in), so every rule here has a test in model.test.ts.

import type {NoSetsReason, Pod, RunView, State} from './api';
import type {RunTab} from './urlstate';

// How a pill reads: colour alone never carries the meaning, the label does, tone only reinforces it.
export type Tone = 'ok' | 'warn' | 'fail' | 'run' | 'idle';
export type PillSpec = {label: string; tone: Tone};

// A status may carry detail after its first word ("provisioning (waiting for capacity, pass 3)").
export const statusKey = (status: string): string => status.split(' ')[0];

const ACTIVE = new Set(['new', 'provisioning', 'setup', 'sweeping', 'gathering', 'running']);

// Has the run not reached a terminal status.
export const isActive = (run: RunView): boolean => ACTIVE.has(statusKey(run.status));

export const isWaitingForCapacity = (run: RunView): boolean =>
  isActive(run) && run.status.includes('waiting for capacity');

// "RTX 3090" from "NVIDIA GeForce RTX 3090"; "RTX PRO 6000 Blackwell Server" from the long RunPod name.
export function shortGpu(name: string): string {
  return name
    .replace(/^NVIDIA\s+/, '')
    .replace(/^GeForce\s+/, '')
    .replace(/\s+(Edition|Generation)$/, '')
    .replace(/\s+Workstation$/, ' WS')
    .trim();
}

// The GPU a run uses: its pod's, else the first GPU type it asked for (with "+N" for the rest).
export function runGpu(run: RunView, pods: Pod[]): string {
  const pod = pods.find((p) => p.id === run.pod_id);
  if (pod) return shortGpu(pod.gpu_type);
  const wanted = run.gpu_types.split(',').map((g) => g.trim()).filter(Boolean);
  if (wanted.length === 0) return 'any GPU';
  return wanted.length > 1 ? `${shortGpu(wanted[0])} +${wanted.length - 1}` : shortGpu(wanted[0]);
}

// What a person calls the run: "decode-curve · qwen3-4b · RTX 3090", or the command of an exec run.
export function runTitle(run: RunView, pods: Pod[]): string {
  if (run.kind === 'exec') {
    const cmd = run.scenario.replace(/\s+/g, ' ').trim();
    return `exec · ${cmd.length > 56 ? `${cmd.slice(0, 55)}…` : cmd}`;
  }
  const models = run.models.map((m) => m.model);
  const shown = models.length > 2 ? `${models.slice(0, 2).join(', ')} +${models.length - 2}` : models.join(', ');
  return [run.scenario, shown || 'no model', runGpu(run, pods)].join(' · ');
}

// The shortest suffix that tells runs apart in a table: "…744667".
export const shortId = (id: string): string => `…${id.slice(-7)}`;

// The pill of a run. A finished sweep reads by its cells (errors are never green); `incomplete:` keeps its own
// text; an unfinished run reads by its phase.
export function runPill(run: RunView): PillSpec {
  const key = statusKey(run.status);
  if (run.outcome) {
    const tone = run.outcome.kind === 'ok' ? 'ok' : run.outcome.kind === 'errors' ? 'warn' : 'fail';
    return {label: run.outcome.label, tone};
  }
  if (key === 'incomplete:') return {label: run.status.replace(/^incomplete:\s*/, 'incomplete: '), tone: 'warn'};
  if (isWaitingForCapacity(run)) return {label: 'waiting for capacity', tone: 'warn'};
  if (isActive(run)) return {label: key === 'running' ? 'running' : run.status.replace(/\s*\(.*$/, ''), tone: 'run'};
  switch (key) {
    case 'done':
      return {label: run.kind === 'sweep' ? 'done · no cells' : 'done', tone: run.kind === 'sweep' ? 'idle' : 'ok'};
    case 'failed':
    case 'aborted':
      return {label: key, tone: 'fail'};
    case 'cancelled':
      return {label: 'cancelled', tone: 'idle'};
    case 'abandoned':
      return {label: 'abandoned: owner exited', tone: 'idle'};
    default:
      return {label: run.status, tone: 'idle'};
  }
}

// The tabs a run has. An exec run has no result sets, so no Results tab.
export function runTabs(run: RunView): RunTab[] {
  return run.has_sets ? ['summary', 'log', 'results', 'pod'] : ['summary', 'log', 'pod'];
}

// The tab to show: the asked one when the run has it, else the summary.
export const tabFor = (run: RunView, tab: RunTab): RunTab => (runTabs(run).includes(tab) ? tab : 'summary');

export function noSetsMessage(reason: NoSetsReason): string {
  switch (reason) {
    case 'exec_run':
      return 'An exec run runs a command on a pod: it has a log, never benchmark result sets.';
    case 'no_results_dir':
      return 'The dashboard runs outside the poot checkout, so it has no results directory to read sets from.';
    case 'no_sets_yet':
      return 'The run has not gathered a result set yet. Sets appear here while it sweeps, as the mirror of the pod’s results.';
    case 'ended_without_sets':
      return 'The run ended before any set was gathered (see its log for why).';
    case 'unknown_run':
      return 'The state database has no run with this id.';
  }
}

// The pill of a pod: the DB's state, unless RunPod disagrees. A pod RunPod does not list is not "busy" whatever
// the DB says; one RunPod could not be asked about says so instead of passing for confirmed.
export function podPill(pod: Pod): PillSpec {
  if (pod.status === 'terminated') return {label: 'terminated', tone: 'idle'};
  const v = pod.verified;
  if (v.state === 'gone') return {label: `missing at RunPod (DB: ${pod.status})`, tone: 'fail'};
  if (v.state === 'unverified') return {label: `${pod.status} · unverified`, tone: 'idle'};
  if (v.state === 'confirmed' && v.runpod_status !== 'RUNNING') {
    return {label: `${pod.status} · RunPod: ${v.runpod_status.toLowerCase()}`, tone: 'warn'};
  }
  return {label: pod.status, tone: pod.status === 'busy' ? 'ok' : pod.status === 'provisioning' ? 'warn' : 'run'};
}

export const isLivePod = (pod: Pod): boolean => pod.status !== 'terminated' && pod.verified.state !== 'gone';

export type Buckets = {running: RunView[]; waiting: RunView[]; recent: RunView[]};

// Runs by what they are doing now. Recent keeps the order of the state (newest first).
export function bucketRuns(state: State): Buckets {
  const running = state.runs.filter((r) => isActive(r) && !isWaitingForCapacity(r));
  const waiting = state.runs.filter(isWaitingForCapacity);
  const recent = state.runs.filter((r) => !isActive(r));
  return {running, waiting, recent};
}

export type RunFilter = {text: string; kind: 'all' | 'sweep' | 'exec'; show: 'all' | 'active' | 'problems'};

// Whether a run needs a look: it failed, or finished with errors or a model short.
export function isProblem(run: RunView): boolean {
  const tone = runPill(run).tone;
  return tone === 'fail' || (tone === 'warn' && !isActive(run));
}

export function filterRuns(runs: RunView[], pods: Pod[], f: RunFilter): RunView[] {
  const text = f.text.trim().toLowerCase();
  return runs.filter((r) => {
    if (f.kind !== 'all' && r.kind !== f.kind) return false;
    if (f.show === 'active' && !isActive(r)) return false;
    if (f.show === 'problems' && !isProblem(r)) return false;
    return !text || `${r.id} ${runTitle(r, pods)} ${r.status} ${r.git_ref}`.toLowerCase().includes(text);
  });
}

// ---- formatting -------------------------------------------------------------------------------------------------

export function formatDuration(secs: number): string {
  const s = Math.max(0, Math.round(secs));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${String(s % 60).padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h ${String(m % 60).padStart(2, '0')}m`;
}

// "5m ago" of an ISO time, against `now` (ms).
export function formatAgo(iso: string, now: number): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return iso;
  const s = Math.max(0, Math.round((now - t) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  return h < 24 ? `${h}h ago` : `${Math.floor(h / 24)}d ago`;
}

// A price that rounds to zero (including -0) prints as $0.00, never "$-0.00".
export const formatUsd = (usd: number | null | undefined): string => {
  if (usd === null || usd === undefined) return '—';
  const v = Math.abs(usd) < 0.005 ? 0 : usd;
  return `$${v < 10 ? v.toFixed(2) : v.toFixed(1)}`;
};

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
}

// "14:13" in the viewer's zone, for a start time.
export function formatClock(iso: string): string {
  const t = new Date(iso);
  return Number.isNaN(t.getTime()) ? iso : t.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
}

// "repeat 2/3" progress of a sweep: sets gathered of sets to gather, or null for a run with no sets.
export function setsProgress(run: RunView): {done: number; total: number} | null {
  if (!run.has_sets || run.sets_total === undefined || run.sets_total === 0) return null;
  return {done: run.sets_done ?? 0, total: run.sets_total};
}
