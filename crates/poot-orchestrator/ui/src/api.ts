// Typed client for the orchestrator API (same origin), plus an adapter from the per-snapshot
// /api/results response to the website's Run shape.

import type {Run, RunEnv, ResultRow} from './charts/types';

export type ModelView = {
  model: string;
  setup_done: boolean;
  sweep_done: boolean;
  gathered: boolean;
  pod_run_id: string | null;
};

// How a finished sweep came out, derived by the server from its gathered rows (not the lifecycle status).
export type Outcome = {
  kind: 'ok' | 'errors' | 'failed';
  label: string;
  failed: number;
  total: number;
};

// A run's wait for a GPU, as its log last reported it (only while it waits for capacity).
export type CapacityWait = {
  pass: number;
  waited_min: number;
  retry_secs: number;
  price_cap: number | null;
};

export type RunView = {
  id: string;
  // `sweep` (a benchmark run) or `exec` (a command on a pod: a log, never result sets).
  kind: 'sweep' | 'exec';
  created_at: string;
  updated_at: string;
  status: string;
  scenario: string;
  git_ref: string;
  gpu_types: string;
  image: string;
  pod_id: string | null;
  ssh_host: string | null;
  ssh_port: number | null;
  repeats: number;
  models: ModelView[];
  has_sets: boolean;
  // Sets gathered of the sets the sweep will gather (repeats x models); sweeps only.
  sets_done?: number;
  sets_total?: number;
  // Seconds left, extrapolated from the sets gathered; only while the run has not ended and has a rate.
  eta_secs?: number | null;
  wait?: CapacityWait | null;
  // Its pod's accrued cost, estimated as price x lifetime.
  cost_usd?: number | null;
  // Set once the sweep has ended (status `done`).
  outcome?: Outcome | null;
  // The harness's latest progress line, while the run has not ended.
  progress?: string | null;
};

// A pod (worker), spec 030: its lifecycle is independent of runs.
export type PodStatus = 'provisioning' | 'idle' | 'busy' | 'terminated';

// What RunPod's own pod list says of a pod the DB knows.
export type Verified =
  | {state: 'terminated'}
  | {state: 'confirmed'; runpod_status: string; checked_secs_ago: number}
  | {state: 'gone'; checked_secs_ago: number}
  | {state: 'unverified'; reason: string};

export type Pod = {
  id: string;
  gpu_type: string;
  cloud: string;
  // USD per hour as RunPod reported it at create, or null.
  cost_per_hr: number | null;
  image: string;
  status: PodStatus;
  ssh_host: string | null;
  ssh_port: number | null;
  current_run: string | null;
  created_at: string;
  updated_at: string;
  verified: Verified;
  uptime_secs?: number;
  // Estimated: price x lifetime.
  accrued_usd?: number | null;
};

// /api/state returns runs and pods as separate lists.
export type State = {
  runs: RunView[];
  pods: Pod[];
};

export type Failure = {
  run: string;
  status: string;
  at: string;
  reason: string | null;
  replacements: number;
};

// /api/overview: the totals and failures of the last 24 hours.
export type Overview = {
  spend_24h_usd: number;
  burn_usd_per_hr: number;
  billing_pods: number;
  failures: Failure[];
  replacements_24h: number;
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

// Why a run has no sets: it is an exec run, the server has no results directory, none is gathered yet, it ended
// without any, or the run is unknown.
export type NoSetsReason =
  | 'exec_run'
  | 'no_results_dir'
  | 'no_sets_yet'
  | 'ended_without_sets'
  | 'unknown_run';

export type ResultsReply = {sets: Snapshot[]; reason: NoSetsReason | null};

// One snapshot dir (one model) of a run's gathered results.
export type Snapshot = {
  dir: string;
  env: RunEnv | null;
  rows: ResultRow[];
  // Gathered with no ok row (kept in failed-results/): shown with its reasons, not charted.
  failed?: boolean;
  // Still being swept: the rows so far, mirrored off the pod.
  in_progress?: boolean;
  updated_ms?: number;
  last_cell?: string | null;
};

async function getJson<T>(url: string): Promise<T> {
  const r = await fetch(url, {cache: 'no-store'});
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return (await r.json()) as T;
}

export async function fetchState(): Promise<State> {
  return getJson<State>('/api/state');
}

export async function fetchOverview(): Promise<Overview> {
  return getJson<Overview>('/api/overview');
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

export async function fetchResults(runId: string): Promise<ResultsReply> {
  const reply = await getJson<ResultsReply | {error: string}>(
    `/api/results?run=${encodeURIComponent(runId)}`,
  );
  if ('error' in reply) throw new Error(reply.error);
  return reply;
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
