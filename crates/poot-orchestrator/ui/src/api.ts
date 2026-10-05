// Typed client for the orchestrator API (same origin), plus an adapter from the per-snapshot
// /api/results response to the website's Run shape.

import type {Run, RunEnv, ResultRow} from './charts/types';

export type RunStatus =
  | 'new'
  | 'provisioning'
  | 'setup'
  | 'sweeping'
  | 'gathering'
  | 'done'
  | 'failed'
  | 'aborted';

export type ModelView = {
  model: string;
  setup_done: boolean;
  sweep_done: boolean;
  gathered: boolean;
  pod_run_id: string | null;
};

export type RunView = {
  id: string;
  created_at: string;
  updated_at: string;
  status: RunStatus;
  scenario: string;
  git_ref: string;
  gpu_types: string;
  image: string;
  pod_id: string | null;
  ssh_host: string | null;
  ssh_port: number | null;
  models: ModelView[];
};

// A pod (worker), spec 030: its lifecycle is independent of runs.
export type PodStatus = 'provisioning' | 'idle' | 'busy' | 'terminated';

export type Pod = {
  id: string;
  gpu_type: string;
  cloud: string;
  image: string;
  status: PodStatus;
  ssh_host: string | null;
  ssh_port: number | null;
  current_run: string | null;
  created_at: string;
  updated_at: string;
};

// /api/state returns runs and pods as separate lists.
export type State = {
  runs: RunView[];
  pods: Pod[];
};

export type LogState = 'running' | 'done' | 'failed';
export type LogKind = 'run' | 'image';

export type LogFile = {
  name: string;
  size: number;
  mtime: number;
  kind: LogKind;
  state: LogState;
};

// One snapshot dir (one model) of a run's gathered results.
export type Snapshot = {
  dir: string;
  env: RunEnv | null;
  rows: ResultRow[];
};

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url, {cache: 'no-store'});
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return (await r.json()) as T;
}

export async function fetchState(): Promise<State> {
  return getJson<State>('/api/state');
}

export async function fetchLogfiles(): Promise<LogFile[]> {
  return getJson<LogFile[]>('/api/logfiles');
}

export async function fetchLog(name: string, tail = 800): Promise<string> {
  const r = await fetch(
    `/api/log?name=${encodeURIComponent(name)}&tail=${tail}`,
    {cache: 'no-store'},
  );
  if (!r.ok) throw new Error(`log ${name}: ${r.status}`);
  return r.text();
}

export async function fetchResults(runId: string): Promise<Snapshot[]> {
  return getJson<Snapshot[]>(`/api/results?run=${encodeURIComponent(runId)}`);
}

// Proxy for a live poot-serve's JSON /metrics (card 051). `target` is `host:port`; the orchestrator
// fetches it, avoiding CORS and reaching remote pods. Always answers 200; an `error` key means the
// target was unreachable or invalid. The /metrics shape varies by engine version, so it stays untyped.
export async function fetchServeMetrics(
  target: string,
): Promise<Record<string, unknown>> {
  return getJson<Record<string, unknown>>(
    `/api/serve-metrics?target=${encodeURIComponent(target)}`,
  );
}

export async function postBuild(
  dockerfile: string,
  tag: string,
  push: boolean,
): Promise<{log?: string; error?: string}> {
  const body = new URLSearchParams({
    dockerfile,
    tag,
    push: push ? '1' : '0',
  }).toString();
  const r = await fetch('/api/build', {
    method: 'POST',
    headers: {'Content-Type': 'application/x-www-form-urlencoded'},
    body,
  });
  return (await r.json()) as {log?: string; error?: string};
}

const EMPTY_ENV: RunEnv = {
  captured_at_utc: null,
  gpu_name: null,
  gpu_driver: null,
  gpu_memory_total: null,
  cuda_version: null,
  host_kernel: null,
  cpu: null,
  harness_version: null,
};

// Build a Run from a run's snapshots: concatenate all rows (each carries model and scenario) and
// use the first non-null env.
export function snapshotsToRun(runId: string, snaps: Snapshot[]): Run {
  const results: ResultRow[] = [];
  let env: RunEnv | null = null;
  for (const s of snaps) {
    if (!env && s.env) env = s.env;
    for (const row of s.rows ?? []) results.push(row);
  }
  return {
    id: runId,
    source: 'results',
    sample: false,
    env: env ?? EMPTY_ENV,
    results,
  };
}
