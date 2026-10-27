// A fixed set of API answers covering every state the dashboard shows (running, waiting for capacity, finished
// with errors, incomplete, failed, an exec run, a pod RunPod no longer lists, ...). The component tests render
// against it and scripts/fixture-server.mjs serves it, so the committed screenshots show states that the live
// database does not hold right now.

import type {LogFile, Overview, Pod, ResultsReply, RunView, Snapshot, State} from './api';
import type {CurvePoint, ResultRow} from './charts/types';

// The tests pin the clock; the fixture server builds the same data around the moment it starts (`__NOW__`).
declare const __NOW__: number | undefined;
export const NOW = typeof __NOW__ === 'number' ? __NOW__ : Date.parse('2026-10-24T12:00:00Z');
const ago = (mins: number): string => new Date(NOW - mins * 60_000).toISOString().replace('.000Z', 'Z');

const model = (name: string, done: boolean) => ({model: name, setup_done: true, sweep_done: done, gathered: done, pod_run_id: done ? `x-${name}` : null});

const base = {
  git_ref: 'master=55d3fa180',
  image: 'ghcr.io/kikijiki/poot-bench@sha256:5d426b930de40ba14fd293079ace3c08297b56085afaba6eec3a46a6918f1b22',
  ssh_host: null,
  ssh_port: null,
  repeats: 3,
  has_sets: true,
} as const;

export const RUNS: RunView[] = [
  {...base, id: 'run-1792800000-111111', kind: 'sweep', created_at: ago(95), updated_at: ago(0), status: 'sweeping', scenario: 'decode-curve', gpu_types: 'NVIDIA GeForce RTX 3090', pod_id: 'pod3090busy', ssh_host: '99.69.17.69', ssh_port: 11100, models: [model('qwen3-4b', false)], sets_done: 2, sets_total: 3, eta_secs: 1700, cost_usd: 0.35, progress: 'qwen3-4b: [progress] poot: isl 2048 done (5/8), ETA 12m'},
  {...base, id: 'run-1792799000-222222', kind: 'sweep', created_at: ago(85), updated_at: ago(1), status: 'provisioning (waiting for capacity, pass 4)', scenario: 'decode-curve', gpu_types: 'NVIDIA GeForce RTX 4090', pod_id: null, models: [model('qwen3-8b', false)], sets_done: 0, sets_total: 3, wait: {pass: 4, waited_min: 85, retry_secs: 71, price_cap: 0.5}},
  {...base, id: 'exec-1792798000-333333333-1690479-0', kind: 'exec', created_at: ago(14), updated_at: ago(2), status: 'running', scenario: 'bash /root/pod.sh; echo ---LANE; cat /root/out/lane.rc /root/out/gate.rc', gpu_types: 'NVIDIA A100-SXM4-80GB,NVIDIA H100 PCIe', pod_id: 'podA100', ssh_host: '216.249.100.66', ssh_port: 20337, models: [], has_sets: false, repeats: 1, cost_usd: 0.33},
  {...base, id: 'run-1792790000-444444', kind: 'sweep', created_at: ago(190), updated_at: ago(120), status: 'done', scenario: 'decode-curve', gpu_types: 'NVIDIA H200', pod_id: 'podH200', models: [model('qwen3-0.6b', true), model('qwen3-4b', true)], sets_done: 6, sets_total: 6, outcome: {kind: 'ok', label: 'done', failed: 0, total: 12}, cost_usd: 6.8},
  {...base, id: 'run-1792789000-555555', kind: 'sweep', created_at: ago(260), updated_at: ago(170), status: 'done', scenario: 'decode-curve', gpu_types: 'NVIDIA RTX PRO 6000 Blackwell Server Edition', pod_id: 'podBW', models: [model('qwen3-4b', true)], sets_done: 3, sets_total: 3, outcome: {kind: 'errors', label: 'done with errors (3 of 20 cells failed)', failed: 3, total: 20}, cost_usd: 3.4},
  {...base, id: 'run-1792788000-666666', kind: 'sweep', created_at: ago(300), updated_at: ago(240), status: 'incomplete: qwen3-32b 0/3', scenario: 'decode-curve', gpu_types: 'NVIDIA GeForce RTX 3090', pod_id: null, models: [model('qwen3-4b', true), model('qwen3-32b', false)], sets_done: 3, sets_total: 6},
  {...base, id: 'run-1792787000-777777', kind: 'sweep', created_at: ago(400), updated_at: ago(380), status: 'failed', scenario: 'decode-curve', gpu_types: 'NVIDIA L40S', pod_id: null, models: [model('qwen3-4b', false)], sets_done: 0, sets_total: 3},
  {...base, id: 'exec-1792786000-888888888-1219748-0', kind: 'exec', created_at: ago(500), updated_at: ago(470), status: 'aborted', scenario: 'bash /root/legA.sh; tar -xf /root/fn.tar -C /root', gpu_types: 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', pod_id: 'podGone', models: [], has_sets: false, repeats: 1},
  {...base, id: 'run-1792785000-999999', kind: 'sweep', created_at: ago(600), updated_at: ago(560), status: 'cancelled', scenario: 'decode-curve', gpu_types: 'NVIDIA GeForce RTX 3090', pod_id: null, models: [model('qwen3-4b', false)], sets_done: 0, sets_total: 3},
];

const pod = (p: Partial<Pod> & Pick<Pod, 'id' | 'gpu_type' | 'status' | 'verified'>): Pod => ({
  cloud: 'COMMUNITY', cost_per_hr: 0.22, image: base.image, ssh_host: null, ssh_port: null, current_run: null, created_at: ago(95), updated_at: ago(1), ...p,
});

export const PODS: Pod[] = [
  pod({id: 'pod3090busy', gpu_type: 'NVIDIA GeForce RTX 3090', status: 'busy', current_run: 'run-1792800000-111111', verified: {state: 'confirmed', runpod_status: 'RUNNING', checked_secs_ago: 12}, uptime_secs: 5700, accrued_usd: 0.35, ssh_host: '99.69.17.69', ssh_port: 11100}),
  pod({id: 'podA100', gpu_type: 'NVIDIA A100-SXM4-80GB', status: 'busy', cost_per_hr: 1.39, current_run: 'exec-1792798000-333333333-1690479-0', created_at: ago(14), verified: {state: 'confirmed', runpod_status: 'RUNNING', checked_secs_ago: 12}, uptime_secs: 840, accrued_usd: 0.33, ssh_host: '216.249.100.66', ssh_port: 20337}),
  pod({id: 'podStale', gpu_type: 'NVIDIA GeForce RTX 3090', status: 'provisioning', created_at: ago(1500), updated_at: ago(1499), verified: {state: 'gone', checked_secs_ago: 12}, uptime_secs: 60, accrued_usd: 0.01}),
  pod({id: 'podNoKey', gpu_type: 'NVIDIA L40S', status: 'idle', cost_per_hr: 0.86, created_at: ago(40), verified: {state: 'unverified', reason: 'RUNPOD_API_KEY is not set'}, uptime_secs: 2400, accrued_usd: 0.57}),
  pod({id: 'podH200', gpu_type: 'NVIDIA H200', status: 'terminated', cloud: 'SECURE', cost_per_hr: 3.59, created_at: ago(190), updated_at: ago(120), verified: {state: 'terminated'}, uptime_secs: 4200, accrued_usd: 4.2}),
  pod({id: 'podGone', gpu_type: 'NVIDIA RTX PRO 6000 Blackwell Workstation Edition', status: 'terminated', cost_per_hr: 1.69, created_at: ago(500), updated_at: ago(470), verified: {state: 'terminated'}, uptime_secs: 1800, accrued_usd: 0.85}),
  pod({id: 'podBW', gpu_type: 'NVIDIA RTX PRO 6000 Blackwell Server Edition', status: 'terminated', cloud: 'SECURE', cost_per_hr: 2.49, created_at: ago(260), updated_at: ago(170), verified: {state: 'terminated'}, uptime_secs: 5400, accrued_usd: 3.4}),
];

export const STATE: State = {runs: RUNS, pods: PODS};

export const OVERVIEW: Overview = {
  spend_24h_usd: 18.62,
  burn_usd_per_hr: 1.61,
  billing_pods: 3,
  replacements_24h: 2,
  failures: [
    {run: 'run-1792787000-777777', status: 'failed', at: ago(380), reason: 'create failed: no instance has the CUDA floor 13.0 the image declares (3 passes)', replacements: 0},
    {run: 'exec-1792786000-888888888-1219748-0', status: 'aborted', at: ago(470), reason: 'pod sq3nu3f73q65ky has no working GPU, provisioning replacement, 0 of 1 repeats already gathered', replacements: 2},
  ],
};

export const LOGFILES: LogFile[] = [
  ...RUNS.map((r, i): LogFile => ({name: `${r.id}.log`, size: 12000 + i * 900, mtime: Math.floor(NOW / 1000) - i * 600, kind: 'run', state: r.status === 'failed' || r.status === 'aborted' ? 'failed' : r.status === 'done' ? 'done' : 'running'})),
  {name: 'image-latest.log', size: 48200, mtime: Math.floor(NOW / 1000) - 7200, kind: 'image', state: 'done'},
  {name: 'image-cuda13.log', size: 9100, mtime: Math.floor(NOW / 1000) - 90000, kind: 'image', state: 'failed'},
];

export function logOf(name: string): string {
  const stamp = (m: number, s: number) => `[${String(11 + Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}:${String(s).padStart(2, '0')}]`;
  const run = RUNS.find((r) => `${r.id}.log` === name);
  const lines = [`${stamp(0, 1)} run ${name.replace('.log', '')}`, `${stamp(0, 2)} provision pass 1, try 1: ${run?.gpu_types ?? 'GPU'} (ALL) image=${base.image.slice(0, 40)}…`];
  if (run?.wait) {
    for (let p = 1; p <= run.wait.pass; p++) lines.push(`${stamp(p, 10)} no capacity for any of ${run.gpu_types}, waiting ${20 + p * 17}s (pass ${p}, waiting ${p * 3} min so far)`);
  } else {
    for (let i = 0; i < 24; i++) lines.push(`${stamp(2 + i, 5)} qwen3-4b: [progress] poot: isl ${128 * 2 ** (i % 8)} done (${(i % 8) + 1}/8), ETA ${20 - i}m`);
    lines.push(`${stamp(30, 0)} pod pod3090busy has no working GPU, provisioning replacement, 1 of 3 repeats already gathered`);
  }
  return lines.join('\n');
}

const pt = (isl: number, tok: number): CurvePoint => ({isl, osl: 64, iters: 5, decode_tok_s: tok, ttft_ms: {p50: isl / 40, p90: isl / 35, p99: isl / 30}, tpot_ms: {p50: 1000 / tok, p90: 1100 / tok, p99: 1300 / tok}, itl_ms: {p50: 1000 / tok, p90: 1100 / tok, p99: 1300 / tok}});

export function curveRow(framework: ResultRow['framework'], modelName: string, peak: number, status: ResultRow['status'] = 'ok'): ResultRow {
  const isls = [128, 256, 512, 1024, 2048, 4096, 8192];
  return {
    framework, model: modelName, precision: 'q4_k_m', scenario: 'decode-curve', status, backend: framework === 'poot' ? 'ptx' : 'cuda',
    ...(status === 'error' ? {reason: 'rc=1: CUDA error: out of memory\n  at poot_ptx::launch (ptx.rs:412)'} : {curve: isls.map((isl, i) => pt(isl, peak * (1 - i * (framework === 'poot' ? 0.045 : 0.02))))}),
  };
}

const env = (gpu: string) => ({captured_at_utc: ago(100), gpu_name: gpu, gpu_driver: '580.95', gpu_memory_total: '24576 MiB', cuda_version: '13.0', host_kernel: '6.8.0', cpu: 'AMD EPYC 7742 64-Core Processor', harness_version: 'abc123'});

const set = (dir: string, gpu: string, rows: ResultRow[], extra: Partial<Snapshot> = {}): Snapshot => ({dir, env: env(gpu), rows, ...extra});

export function resultsOf(runId: string): ResultsReply {
  const run = RUNS.find((r) => r.id === runId);
  if (!run) return {sets: [], reason: 'unknown_run'};
  if (run.kind === 'exec') return {sets: [], reason: 'exec_run'};
  switch (run.id) {
    case 'run-1792800000-111111':
      return {reason: null, sets: [
        set('2026-10-24-geforce-rtx-3090-qwen3-4b-101500-r1', 'NVIDIA GeForce RTX 3090', [curveRow('poot', 'qwen3-4b', 92), curveRow('vllm', 'qwen3-4b', 118)]),
        set('2026-10-24-geforce-rtx-3090-qwen3-4b-101500-r2.partial', 'NVIDIA GeForce RTX 3090', [curveRow('poot', 'qwen3-4b', 90)], {in_progress: true, updated_ms: NOW - 40_000, last_cell: 'poot ptx decode-curve'}),
      ]};
    case 'run-1792790000-444444':
      return {reason: null, sets: [set('2026-10-24-h200-qwen3-4b-r1', 'NVIDIA H200', [curveRow('poot', 'qwen3-4b', 210), curveRow('vllm', 'qwen3-4b', 260), curveRow('llamacpp', 'qwen3-4b', 190)])]};
    case 'run-1792789000-555555':
      return {reason: null, sets: [set('2026-10-24-rtx-pro-6000-qwen3-4b-r1', 'NVIDIA RTX PRO 6000', [curveRow('poot', 'qwen3-4b', 180), curveRow('vllm', 'qwen3-4b', 0, 'error'), curveRow('candle', 'qwen3-4b', 0, 'error')])]};
    case 'run-1792788000-666666':
      return {reason: null, sets: [set('2026-10-24-geforce-rtx-3090-qwen3-4b-r1', 'NVIDIA GeForce RTX 3090', [curveRow('poot', 'qwen3-4b', 95)])]};
    case 'run-1792787000-777777':
    case 'run-1792785000-999999':
      return {sets: [], reason: 'ended_without_sets'};
    default:
      return {sets: [], reason: 'no_sets_yet'};
  }
}
