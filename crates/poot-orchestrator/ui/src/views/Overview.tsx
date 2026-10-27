import React from 'react';
import type {Overview as OverviewData, Pod, RunView, State} from '../api';
import {bucketRuns, formatAgo, formatClock, formatDuration, formatUsd, isLivePod, podPill, runGpu, runPill, runTitle, setsProgress, shortId} from '../model';
import {Async, Empty, Kpi, Pill, ProgressBar} from '../ui';
import type {Loadable} from '../ui';

type Open = (runId: string) => void;

function RunLink(props: {run: RunView; pods: Pod[]; onOpen: Open}) {
  return (
    <button type="button" className="link" onClick={() => props.onOpen(props.run.id)}>
      {runTitle(props.run, props.pods)} <span className="muted mono">{shortId(props.run.id)}</span>
    </button>
  );
}

function RunningTable(props: {runs: RunView[]; pods: Pod[]; now: number; onOpen: Open}) {
  if (props.runs.length === 0) return <Empty title="Nothing is running">No run is provisioning, sweeping or gathering.</Empty>;
  return (
    <div className="tablewrap stack">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>GPU</th>
            <th>Progress</th>
            <th className="n">Cost (est.)</th>
            <th>Started / left</th>
          </tr>
        </thead>
        <tbody>
          {props.runs.map((r) => {
            const p = setsProgress(r);
            return (
              <tr key={r.id}>
                <td data-label="Run">
                  <Pill spec={runPill(r)} /> <RunLink run={r} pods={props.pods} onOpen={props.onOpen} />
                </td>
                <td data-label="GPU">{runGpu(r, props.pods)}</td>
                <td data-label="Progress">{p ? <ProgressBar done={p.done} total={p.total} label={`${p.done}/${p.total} sets`} /> : <span className="muted">{r.progress ?? r.status}</span>}</td>
                <td data-label="Cost (est.)" className="n">{formatUsd(r.cost_usd)}</td>
                <td data-label="Started / left">
                  {formatClock(r.created_at)}
                  {r.eta_secs ? ` · ~${formatDuration(r.eta_secs)} left` : ''}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function WaitingTable(props: {runs: RunView[]; pods: Pod[]; onOpen: Open}) {
  if (props.runs.length === 0) return <Empty title="Nothing is waiting">No run is waiting for capacity.</Empty>;
  return (
    <div className="tablewrap stack">
      <table>
        <thead>
          <tr>
            <th>Run</th>
            <th>Wanted</th>
            <th className="n">Pass</th>
            <th className="n">Waited</th>
            <th className="n">Price cap</th>
            <th className="n">Next try</th>
          </tr>
        </thead>
        <tbody>
          {props.runs.map((r) => (
            <tr key={r.id}>
              <td data-label="Run">
                <Pill spec={runPill(r)} /> <RunLink run={r} pods={props.pods} onOpen={props.onOpen} />
              </td>
              <td data-label="Wanted">{runGpu(r, props.pods)}</td>
              <td data-label="Pass" className="n">{r.wait?.pass ?? '—'}</td>
              <td data-label="Waited" className="n">{r.wait ? `${r.wait.waited_min} min` : '—'}</td>
              <td data-label="Price cap" className="n">{r.wait?.price_cap != null ? `$${r.wait.price_cap}/h` : 'none'}</td>
              <td data-label="Next try" className="n">{r.wait ? `≤ ${r.wait.retry_secs}s` : '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function PodTable(props: {pods: Pod[]; runs: RunView[]; now: number; onOpen: Open}) {
  if (props.pods.length === 0) return <Empty title="No pods">No pod has been created yet.</Empty>;
  return (
    <div className="tablewrap stack">
      <table>
        <thead>
          <tr>
            <th>Pod</th>
            <th>State</th>
            <th>GPU</th>
            <th className="n">$/h</th>
            <th className="n">Uptime</th>
            <th className="n">Cost (est.)</th>
            <th>Run</th>
            <th>RunPod says</th>
          </tr>
        </thead>
        <tbody>
          {props.pods.map((p) => {
            const owner = props.runs.find((r) => r.id === p.current_run || r.pod_id === p.id);
            return (
              <tr key={p.id} className={p.status === 'terminated' ? 'dim' : ''}>
                <td data-label="Pod" className="mono">{p.id}</td>
                <td data-label="State">
                  <Pill spec={podPill(p)} />
                </td>
                <td data-label="GPU">
                  {p.gpu_type.replace('NVIDIA ', '')} <span className="muted">{p.cloud.toLowerCase()}</span>
                </td>
                <td data-label="$/h" className="n">{p.cost_per_hr === null ? '—' : p.cost_per_hr.toFixed(2)}</td>
                <td data-label="Uptime" className="n">{p.uptime_secs === undefined ? '—' : formatDuration(p.uptime_secs)}</td>
                <td data-label="Cost (est.)" className="n">{formatUsd(p.accrued_usd)}</td>
                <td data-label="Run">{owner ? <button type="button" className="link" onClick={() => props.onOpen(owner.id)}>{shortId(owner.id)}</button> : <span className="muted">none</span>}</td>
                <td data-label="RunPod says" className="muted">{verifiedText(p, props.now)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function verifiedText(pod: Pod, _now: number): string {
  const v = pod.verified;
  switch (v.state) {
    case 'terminated':
      return `ended ${formatAgo(pod.updated_at, _now)}`;
    case 'confirmed':
      return `${v.runpod_status.toLowerCase()} · checked ${v.checked_secs_ago}s ago`;
    case 'gone':
      return 'does not list it: removed or reaped';
    case 'unverified':
      return `not checked: ${v.reason}`;
  }
}

function RecentOutcomes(props: {runs: RunView[]; pods: Pod[]; now: number; onOpen: Open}) {
  if (props.runs.length === 0) return <Empty title="No finished runs">Finished runs appear here with how they ended.</Empty>;
  return (
    <ul className="outcomes">
      {props.runs.slice(0, 8).map((r) => (
        <li key={r.id}>
          <Pill spec={runPill(r)} />
          <RunLink run={r} pods={props.pods} onOpen={props.onOpen} />
          <span className="muted">{formatAgo(r.updated_at, props.now)}</span>
        </li>
      ))}
    </ul>
  );
}

function Totals(props: {overview: Loadable<OverviewData>; pods: Pod[]; running: number; waiting: number}) {
  const o = props.overview.kind === 'ready' ? props.overview.value : null;
  const dash = props.overview.kind === 'error' ? '—' : '…';
  return (
    <div className="kpis">
      <Kpi label="Running now" value={props.running} note={`${props.pods.filter((p) => p.status === 'busy' && isLivePod(p)).length} pod(s) busy`} />
      <Kpi label="Waiting for capacity" value={props.waiting} tone={props.waiting > 0 ? 'warn' : undefined} />
      <Kpi label="Spend, last 24 h" value={o ? formatUsd(o.spend_24h_usd) : dash} note="estimated: pod price × lifetime" />
      <Kpi label="Burn rate" value={o ? `${formatUsd(o.burn_usd_per_hr)}/h` : dash} note={o ? `${o.billing_pods} pod(s) billing` : undefined} />
      <Kpi
        label="Failures, last 24 h"
        value={o ? o.failures.length : dash}
        tone={o && o.failures.length > 0 ? 'fail' : undefined}
        note={o ? `${o.replacements_24h} pod replacement(s)` : undefined}
      />
    </div>
  );
}

function Failures(props: {overview: OverviewData; runs: RunView[]; pods: Pod[]; now: number; onOpen: Open}) {
  const {failures} = props.overview;
  if (failures.length === 0) return <Empty title="No failures">No run failed or aborted in the last 24 hours.</Empty>;
  return (
    <ul className="outcomes failures-list">
      {failures.map((f) => {
        const run = props.runs.find((r) => r.id === f.run);
        return (
          <li key={f.run}>
            <Pill spec={{label: f.status, tone: 'fail'}} />
            {run ? <RunLink run={run} pods={props.pods} onOpen={props.onOpen} /> : <span className="mono">{shortId(f.run)}</span>}
            <span className="muted">{formatAgo(f.at, props.now)}</span>
            {f.replacements > 0 ? <span className="muted">· {f.replacements} pod replacement(s)</span> : null}
            {f.reason ? <span className="reason-line mono">{f.reason}</span> : null}
          </li>
        );
      })}
    </ul>
  );
}

export default function Overview(props: {state: Loadable<State>; overview: Loadable<OverviewData>; now: number; onOpen: Open; onRetry: () => void}) {
  return (
    <Async what="the overview" data={props.state} onRetry={props.onRetry}>
      {(state) => {
        const {running, waiting, recent} = bucketRuns(state);
        const livePods = state.pods.filter((p) => p.status !== 'terminated');
        return (
          <div className="overview">
            {props.overview.kind === 'ready' && props.overview.value.failures.length > 0 ? (
              <div className="callout tone-fail" role="status">
                <strong>{props.overview.value.failures.length} run(s) failed in the last 24 h.</strong> See Failures below.
              </div>
            ) : null}
            <Totals overview={props.overview} pods={state.pods} running={running.length} waiting={waiting.length} />
            <div className="cols">
              <div>
                <h2>Running now</h2>
                <RunningTable runs={running} pods={state.pods} now={props.now} onOpen={props.onOpen} />
                <h2>Waiting for capacity</h2>
                <WaitingTable runs={waiting} pods={state.pods} onOpen={props.onOpen} />
              </div>
              <div>
                <h2>Failures, last 24 h</h2>
                <Async what="failures" data={props.overview} onRetry={props.onRetry}>
                  {(o) => <Failures overview={o} runs={state.runs} pods={state.pods} now={props.now} onOpen={props.onOpen} />}
                </Async>
                <h2>Recent runs</h2>
                <RecentOutcomes runs={recent} pods={state.pods} now={props.now} onOpen={props.onOpen} />
              </div>
            </div>
            <h2>Pods</h2>
            <PodTable pods={livePods} runs={state.runs} now={props.now} onOpen={props.onOpen} />
          </div>
        );
      }}
    </Async>
  );
}
