import React, {useMemo} from 'react';
import {snapshotsToRun} from '../api';
import type {RunView, Snapshot} from '../api';
import {modelsInRun} from '../charts/data';
import DegradationCurve from '../charts/DegradationCurve';
import ResourceUtil from '../charts/ResourceUtil';
import RunMetadata from '../charts/RunMetadata';
import SinglePointBars from '../charts/SinglePointBars';
import type {ResultRow} from '../charts/types';
import {useResults} from '../data';
import {formatDuration, isActive, noSetsMessage, runPill} from '../model';
import type {PillSpec} from '../model';
import type {ResultSub} from '../urlstate';
import {RESULT_SUBS} from '../urlstate';
import {Async, Empty, Pill} from './../ui';

const SUB_LABEL: Record<ResultSub, string> = {table: 'Table', charts: 'Charts', failures: 'Failures'};

const cellName = (r: ResultRow): string => [r.framework, r.backend, r.scenario].filter(Boolean).join(' ');

export const rowPill = (r: ResultRow): PillSpec =>
  r.status === 'ok' ? {label: 'ok', tone: 'ok'} : r.status === 'error' ? {label: 'error', tone: 'fail'} : {label: 'unsupported', tone: 'idle'};

// The first line of a failure reason.
const reasonHead = (reason: string | undefined): string => (reason ?? '').split('\n')[0];

function headline(r: ResultRow): string {
  if (r.status !== 'ok') return '—';
  const curve = r.curve?.[0];
  const v = r.decode_tok_s ?? curve?.decode_tok_s;
  return v === undefined ? '—' : `${v.toFixed(v >= 100 ? 0 : 1)} tok/s`;
}

function Table(props: {rows: ResultRow[]}) {
  if (props.rows.length === 0) return <Empty title="No cell has finished yet">Rows appear here as the sweep measures them.</Empty>;
  return (
    <div className="tablewrap stack">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Cell</th>
            <th>Status</th>
            <th className="n">Decode</th>
            <th>Detail</th>
          </tr>
        </thead>
        <tbody>
          {props.rows.map((r, i) => (
            <tr key={i}>
              <td data-label="Model">{r.model}</td>
              <td data-label="Cell">{cellName(r)}</td>
              <td data-label="Status">
                <Pill spec={rowPill(r)} />
              </td>
              <td data-label="Decode" className="n">{headline(r)}</td>
              <td data-label="Detail" className="wrap">{reasonHead(r.reason) || (r.caveats?.length ? r.caveats.join('; ') : '')}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Failures(props: {snaps: Snapshot[]}) {
  const failing = props.snaps.flatMap((s) =>
    s.rows.filter((r) => r.status === 'error').map((r) => ({dir: s.dir, row: r, set: s.failed ? 'failed set' : 'set'})),
  );
  if (failing.length === 0) return <Empty title="No failures">Every cell that finished was measured.</Empty>;
  return (
    <div className="failures">
      {failing.map(({dir, row, set}, i) => (
        <details className="reason" key={i}>
          <summary>
            <span className="cell">{cellName(row)}</span> <span className="muted">{row.model} · {set} {dir}</span>
            <span className="head">{reasonHead(row.reason)}</span>
          </summary>
          <pre>{row.reason ?? '(no reason recorded)'}</pre>
        </details>
      ))}
    </div>
  );
}

function Charts(props: {runId: string; snaps: Snapshot[]}) {
  const sets = props.snaps.filter((s) => !s.failed);
  const run = useMemo(() => snapshotsToRun(props.runId, sets), [props.runId, sets]);
  const blocks = useMemo(
    () => sets.map((s) => ({dir: s.dir, partial: !!s.in_progress, models: modelsInRun({...run, results: s.rows ?? []})})),
    [run, sets],
  );
  if (run.results.length === 0) return <Empty title="Nothing to chart yet">Charts appear once a cell has been measured.</Empty>;
  return (
    <>
      {blocks.map((b) =>
        b.models.map((model) => (
          <section className="snap-block" key={`${b.dir}:${model}`}>
            <h3>
              {model} <span className="muted mono">{b.dir.replace(/\.partial$/, '')}</span>
              {b.partial ? <Pill spec={{label: 'partial', tone: 'run'}} /> : null}
            </h3>
            {run.results.some((r) => r.scenario !== 'decode-curve') ? <SinglePointBars run={run} model={model} /> : null}
            <DegradationCurve run={run} model={model} />
            <ResourceUtil run={run} model={model} />
            <RunMetadata run={run} model={model} />
          </section>
        )),
      )}
    </>
  );
}

// The results of one sweep: live while it runs (the partial mirror off the pod), final once it ends.
export default function ResultsPanel(props: {run: RunView; sub: ResultSub; onSub: (s: ResultSub) => void; now: number}) {
  const {run, sub} = props;
  const live = isActive(run);
  const {data, reload} = useResults(run.id, live);
  const pill = runPill(run);

  return (
    <div className="results">
      <Async what="results" data={data} onRetry={reload}>
        {(reply) => {
          if (reply.sets.length === 0) {
            return (
              <>
                <Empty title="No result sets">{reply.reason ? noSetsMessage(reply.reason) : 'This run has no sets.'}</Empty>
                {run.progress ? <p className="progress muted mono">{run.progress}</p> : null}
              </>
            );
          }
          const rows = reply.sets.flatMap((s) => s.rows);
          const partial = reply.sets.filter((s) => s.in_progress);
          const errors = rows.filter((r) => r.status === 'error').length;
          return (
            <>
              <div className={`outcome tone-${pill.tone}`}>
                <Pill spec={pill} />
                <span>
                  {run.outcome
                    ? run.outcome.failed > 0
                      ? `${run.outcome.failed} of ${run.outcome.total} cells failed`
                      : `${run.outcome.total} cells measured`
                    : `${rows.filter((r) => r.status === 'ok').length} cells measured so far`}
                </span>
                {partial.map((s) => (
                  <span className="muted" key={s.dir}>
                    partial: {s.last_cell ? `last cell ${s.last_cell}` : 'no cell yet'}
                    {s.updated_ms ? ` · mirrored ${formatDuration((props.now - s.updated_ms) / 1000)} ago` : ''}
                  </span>
                ))}
                {run.progress && live ? <span className="muted mono">{run.progress}</span> : null}
              </div>
              <div className="subtabs" role="tablist" aria-label="Results view">
                {RESULT_SUBS.map((s) => (
                  <button
                    key={s}
                    type="button"
                    role="tab"
                    aria-selected={sub === s}
                    className={sub === s ? 'active' : ''}
                    onClick={() => props.onSub(s)}>
                    {SUB_LABEL[s]}
                    {s === 'failures' ? <span className="count">{errors}</span> : null}
                  </button>
                ))}
              </div>
              {sub === 'table' ? <Table rows={rows} /> : sub === 'failures' ? <Failures snaps={reply.sets} /> : <Charts runId={run.id} snaps={reply.sets} />}
            </>
          );
        }}
      </Async>
    </div>
  );
}
