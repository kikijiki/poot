import React, {useMemo, useState} from 'react';
import type {LogFile, Pod, RunView} from '../api';
import {useResults} from '../data';
import {
  filterRuns,
  formatAgo,
  formatClock,
  formatDuration,
  formatUsd,
  isActive,
  isWaitingForCapacity,
  podPill,
  runPill,
  runTabs,
  runTitle,
  setsProgress,
  shortId,
  tabFor,
} from '../model';
import type {RunFilter} from '../model';
import type {ResultSub, RunTab} from '../urlstate';
import {Async, CopyButton, Empty, Facts, Pill, ProgressBar} from '../ui';
import LogPane from './LogPane';
import ResultsPanel from './ResultsPanel';

const TAB_LABEL: Record<RunTab, string> = {summary: 'Summary', log: 'Log', results: 'Results', pod: 'Pod'};

// The log file of a run: runs write `<id>.log`.
export function runLogName(run: RunView, logfiles: LogFile[]): string | null {
  return logfiles.find((f) => f.kind === 'run' && f.name === `${run.id}.log`)?.name ?? null;
}

function RunRow(props: {run: RunView; pods: Pod[]; selected: boolean; now: number; onSelect: () => void}) {
  const {run, now} = props;
  const progress = setsProgress(run);
  return (
    <li>
      <button type="button" className={`runrow${props.selected ? ' selected' : ''}`} aria-current={props.selected} onClick={props.onSelect}>
        <span className="runrow-top">
          <span className="runrow-title">{runTitle(run, props.pods)}</span>
          <Pill spec={runPill(run)} />
        </span>
        <span className="runrow-sub muted">
          {progress ? `${progress.done}/${progress.total} sets · ` : ''}
          {run.cost_usd !== undefined && run.cost_usd !== null ? `${formatUsd(run.cost_usd)} est. · ` : ''}
          {isActive(run) ? `started ${formatAgo(run.created_at, now)}` : `ended ${formatAgo(run.updated_at, now)}`}
          {isActive(run) && run.eta_secs ? ` · ~${formatDuration(run.eta_secs)} left` : ''}
          <span className="mono"> · {shortId(run.id)}</span>
        </span>
      </button>
    </li>
  );
}

function RunList(props: {
  runs: RunView[];
  pods: Pod[];
  selected: string | null;
  now: number;
  onSelect: (id: string) => void;
}) {
  const [filter, setFilter] = useState<RunFilter>({text: '', kind: 'all', show: 'all'});
  const shown = useMemo(() => filterRuns(props.runs, props.pods, filter), [props.runs, props.pods, filter]);
  return (
    <div className="runlist">
      <div className="filters">
        <input
          type="search"
          placeholder="Filter runs"
          aria-label="Filter runs"
          value={filter.text}
          onChange={(e) => setFilter({...filter, text: e.target.value})}
        />
        <select aria-label="Show" value={filter.show} onChange={(e) => setFilter({...filter, show: e.target.value as RunFilter['show']})}>
          <option value="all">all</option>
          <option value="active">active</option>
          <option value="problems">problems</option>
        </select>
        <select aria-label="Kind" value={filter.kind} onChange={(e) => setFilter({...filter, kind: e.target.value as RunFilter['kind']})}>
          <option value="all">sweeps + exec</option>
          <option value="sweep">sweeps</option>
          <option value="exec">exec</option>
        </select>
      </div>
      {props.runs.length === 0 ? (
        <Empty title="No runs yet">Start one with <code>poot-orchestrator run</code> or <code>exec</code>.</Empty>
      ) : shown.length === 0 ? (
        <Empty title="No run matches">Clear the filter to see all {props.runs.length} runs.</Empty>
      ) : (
        <ul>
          {shown.map((r) => (
            <RunRow key={r.id} run={r} pods={props.pods} now={props.now} selected={props.selected === r.id} onSelect={() => props.onSelect(r.id)} />
          ))}
        </ul>
      )}
    </div>
  );
}

function ModelTable(props: {run: RunView}) {
  const active = isActive(props.run);
  const cell = (done: boolean, firstPending: boolean) => (done ? '✓ done' : active && firstPending ? '… running' : '–');
  return (
    <div className="tablewrap stack">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Setup</th>
            <th>Sweep</th>
            <th>Gathered</th>
          </tr>
        </thead>
        <tbody>
          {props.run.models.map((m) => (
            <tr key={m.model}>
              <td data-label="Model">{m.model}</td>
              <td data-label="Setup">{cell(m.setup_done, true)}</td>
              <td data-label="Sweep">{cell(m.sweep_done, m.setup_done)}</td>
              <td data-label="Gathered">{cell(m.gathered, m.sweep_done)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Summary(props: {run: RunView; pods: Pod[]; now: number}) {
  const {run, pods, now} = props;
  const pod = pods.find((p) => p.id === run.pod_id);
  const progress = setsProgress(run);
  const results = useResults(run.id, false);
  const cpu = results.data.kind === 'ready' ? results.data.value.sets.find((s) => s.env?.cpu)?.env?.cpu : null;
  const wait = run.wait;
  return (
    <div className="summary">
      {isWaitingForCapacity(run) ? (
        <div className="callout tone-warn">
          <strong>Waiting for capacity</strong>
          {wait
            ? ` · pass ${wait.pass}, ${wait.waited_min} min so far, next try in ≤ ${wait.retry_secs}s${wait.price_cap !== null ? `, price cap $${wait.price_cap}/h` : ''}`
            : ' · the log has not reported a pass yet'}
        </div>
      ) : null}
      {run.status.startsWith('incomplete:') ? (
        <div className="callout tone-warn">
          <strong>Incomplete</strong> · valid sets per requested model: {run.status.replace(/^incomplete:\s*/, '')}
        </div>
      ) : null}
      {run.outcome && run.outcome.kind !== 'ok' ? (
        <div className={`callout tone-${run.outcome.kind === 'failed' ? 'fail' : 'warn'}`}>
          <strong>{run.outcome.label}</strong>
        </div>
      ) : null}
      <Facts
        items={[
          ['Run id', <><span className="mono">{run.id}</span> <CopyButton text={run.id} label="Copy id" /></>],
          ['Status', <Pill spec={runPill(run)} />],
          ['Kind', run.kind === 'exec' ? 'exec (a command on a pod)' : `sweep · ${run.scenario}`],
          ['Models', run.models.length ? run.models.map((m) => m.model).join(', ') : null],
          ['GPU', pod ? `${pod.gpu_type} · ${pod.cloud}` : `asked: ${run.gpu_types}`],
          ['Host CPU', cpu],
          ['Started', `${formatClock(run.created_at)} (${formatAgo(run.created_at, now)})`],
          [isActive(run) ? 'Left' : 'Ended', isActive(run) ? (run.eta_secs ? `~${formatDuration(run.eta_secs)}` : 'no estimate yet') : formatAgo(run.updated_at, now)],
          ['Cost', run.cost_usd !== undefined && run.cost_usd !== null ? `${formatUsd(run.cost_usd)} estimated (pod price × lifetime)` : null],
          ['Ref', run.git_ref],
          ['Repeats', run.has_sets ? String(run.repeats) : null],
        ]}
      />
      {progress ? <ProgressBar done={progress.done} total={progress.total} label={`${progress.done} of ${progress.total} sets gathered`} /> : null}
      {run.progress ? <p className="progress muted mono">{run.progress}</p> : null}
      {run.models.length > 0 ? <ModelTable run={run} /> : null}
      {run.kind === 'exec' ? (
        <>
          <h3>Command</h3>
          <pre className="cmd">{run.scenario}</pre>
        </>
      ) : null}
    </div>
  );
}

function PodTab(props: {run: RunView; pods: Pod[]; now: number}) {
  const pod = props.pods.find((p) => p.id === props.run.pod_id);
  if (!pod) return <Empty title="No pod">{isActive(props.run) ? 'The run has not been given a pod yet.' : 'The run ended without a pod, or its pod record was cleared.'}</Empty>;
  const ssh = pod.ssh_host ? `ssh root@${pod.ssh_host} -p ${pod.ssh_port ?? 22}` : null;
  return (
    <Facts
      items={[
        ['Pod', <><span className="mono">{pod.id}</span> <CopyButton text={pod.id} label="Copy id" /></>],
        ['State', <Pill spec={podPill(pod)} />],
        ['GPU', `${pod.gpu_type} · ${pod.cloud}`],
        ['Price', pod.cost_per_hr === null ? null : `$${pod.cost_per_hr.toFixed(2)}/h`],
        ['Uptime', pod.uptime_secs === undefined ? null : formatDuration(pod.uptime_secs)],
        ['Accrued', pod.accrued_usd === undefined || pod.accrued_usd === null ? null : `${formatUsd(pod.accrued_usd)} estimated`],
        ['SSH', ssh ? <><span className="mono">{ssh}</span> <CopyButton text={ssh} label="Copy ssh" /></> : null],
        ['Image', <span className="mono wrap">{pod.image}</span>],
      ]}
    />
  );
}

export function RunDetail(props: {
  run: RunView;
  pods: Pod[];
  logfiles: LogFile[];
  tab: RunTab;
  sub: ResultSub;
  now: number;
  onTab: (t: RunTab) => void;
  onSub: (s: ResultSub) => void;
}) {
  const {run} = props;
  const tab = tabFor(run, props.tab);
  const tabs = runTabs(run);
  return (
    <article className="detail" aria-label={runTitle(run, props.pods)}>
      <header className="detail-head">
        <h2>{runTitle(run, props.pods)}</h2>
        <Pill spec={runPill(run)} />
        <span className="muted mono">{shortId(run.id)}</span>
      </header>
      <div className="tabs" role="tablist" aria-label="Run sections">
        {tabs.map((t) => (
          <button key={t} type="button" role="tab" aria-selected={tab === t} className={tab === t ? 'active' : ''} onClick={() => props.onTab(t)}>
            {TAB_LABEL[t]}
          </button>
        ))}
      </div>
      <div className="tabpanel" role="tabpanel">
        {tab === 'summary' ? (
          <Summary run={run} pods={props.pods} now={props.now} />
        ) : tab === 'log' ? (
          <LogPane name={runLogName(run, props.logfiles)} live={isActive(run)} missing="This run has no log file in the state directory." />
        ) : tab === 'results' ? (
          <ResultsPanel run={run} sub={props.sub} onSub={props.onSub} now={props.now} />
        ) : (
          <PodTab run={run} pods={props.pods} now={props.now} />
        )}
      </div>
    </article>
  );
}

export default function Runs(props: {
  runs: RunView[];
  pods: Pod[];
  logfiles: LogFile[];
  selected: string | null;
  tab: RunTab;
  sub: ResultSub;
  now: number;
  onSelect: (id: string) => void;
  onTab: (t: RunTab) => void;
  onSub: (s: ResultSub) => void;
}) {
  const run = props.runs.find((r) => r.id === props.selected) ?? null;
  return (
    <div className={`split${run ? ' has-detail' : ''}`}>
      <RunList runs={props.runs} pods={props.pods} selected={props.selected} now={props.now} onSelect={props.onSelect} />
      <div className="detailpane">
        {run ? (
          <>
            <button type="button" className="btn ghost back" onClick={() => props.onSelect('')}>
              ← All runs
            </button>
            <RunDetail run={run} pods={props.pods} logfiles={props.logfiles} tab={props.tab} sub={props.sub} now={props.now} onTab={props.onTab} onSub={props.onSub} />
          </>
        ) : (
          <Empty title={props.selected ? 'Run not found' : 'Select a run'}>
            {props.selected ? `No run with id ${props.selected} among the 50 most recent.` : 'Pick one on the left to see its summary, log and results.'}
          </Empty>
        )}
      </div>
    </div>
  );
}
