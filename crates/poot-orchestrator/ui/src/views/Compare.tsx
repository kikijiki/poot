import React, {useEffect, useMemo, useState} from 'react';
import {CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis} from 'recharts';
import {fetchResults} from '../api';
import type {Pod, RunView, Snapshot, State} from '../api';
import {METRICS, curveChoices, curveOf} from '../charts/compare';
import type {Metric} from '../charts/compare';
import {ChartCard, ChartLegend} from '../charts/ui';
import {formatAgo, runGpu, shortId} from '../model';
import {Async, Empty, ErrorBox, Loading} from '../ui';
import type {Loadable} from '../ui';

const MAX_RUNS = 6;
// Okabe-Ito: distinguishable under the common colour-vision deficiencies, readable on light and dark.
const SERIES = ['#0072B2', '#E69F00', '#009E73', '#CC79A7', '#56B4E9', '#D55E00'];
const DASHES = [undefined, '7 4', '2 3', '10 3 2 3', '4 4', '1 4'];

// Runs that can be compared: sweeps that have gathered at least one set.
const comparable = (r: RunView): boolean => r.has_sets && (r.sets_done ?? 0) > 0;

const fmtIsl = (n: number): string => (n >= 1024 ? `${Math.round(n / 1024)}k` : String(n));
const fmtNum = (n: number): string => (!Number.isFinite(n) ? '—' : n >= 100 ? n.toFixed(0) : n.toFixed(1));

function useSetsOf(runIds: string[]): Loadable<Record<string, Snapshot[]>> {
  const key = runIds.join(',');
  const [data, setData] = useState<Loadable<Record<string, Snapshot[]>>>({kind: 'ready', value: {}});
  useEffect(() => {
    if (runIds.length === 0) {
      setData({kind: 'ready', value: {}});
      return;
    }
    let stop = false;
    setData({kind: 'loading'});
    Promise.all(runIds.map(async (id) => [id, (await fetchResults(id)).sets] as const))
      .then((pairs) => !stop && setData({kind: 'ready', value: Object.fromEntries(pairs)}))
      .catch((e) => !stop && setData({kind: 'error', message: e instanceof Error ? e.message : String(e)}));
    return () => {
      stop = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  return data;
}

function Chart(props: {runs: RunView[]; pods: Pod[]; sets: Record<string, Snapshot[]>}) {
  const {runs, pods, sets} = props;
  const choices = useMemo(() => curveChoices(runs.map((r) => sets[r.id] ?? [])), [runs, sets]);
  const [model, setModel] = useState('');
  const [framework, setFramework] = useState('');
  const [metric, setMetric] = useState<Metric>('decode_tok_s');
  const m = choices.models.includes(model) ? model : (choices.models[0] ?? '');
  const f = choices.frameworks.includes(framework) ? framework : (choices.frameworks.includes('poot') ? 'poot' : (choices.frameworks[0] ?? ''));

  const series = useMemo(
    () => runs.map((r, i) => ({run: r, label: `${runGpu(r, pods)} ${shortId(r.id)}`, color: SERIES[i % SERIES.length], dash: DASHES[i % DASHES.length], points: curveOf(sets[r.id] ?? [], m, f, metric)})),
    [runs, pods, sets, m, f, metric],
  );
  const isls = [...new Set(series.flatMap((s) => s.points.map(([isl]) => isl)))].sort((a, b) => a - b);
  const data = isls.map((isl) => Object.fromEntries([['isl', isl], ...series.map((s) => [s.label, s.points.find(([i]) => i === isl)?.[1]])]));
  const meta = METRICS.find((x) => x.value === metric)!;

  if (choices.models.length === 0) return <Empty title="No decode curve in these runs">None of the selected runs measured a decode-curve cell yet.</Empty>;
  return (
    <ChartCard title={`${m} · ${f} · ${meta.label}`} sub={meta.higherIsBetter ? 'Higher is better. One line per run; the median over its repeats.' : 'Lower is better. One line per run; the median over its repeats.'}>
      <div className="controls">
        <label>
          Model
          <select value={m} onChange={(e) => setModel(e.target.value)}>
            {choices.models.map((x) => <option key={x}>{x}</option>)}
          </select>
        </label>
        <label>
          Framework
          <select value={f} onChange={(e) => setFramework(e.target.value)}>
            {choices.frameworks.map((x) => <option key={x}>{x}</option>)}
          </select>
        </label>
        <label>
          Metric
          <select value={metric} onChange={(e) => setMetric(e.target.value as Metric)}>
            {METRICS.map((x) => <option key={x.value} value={x.value}>{x.label}</option>)}
          </select>
        </label>
      </div>
      <ChartLegend items={series.map((s) => ({label: s.label, color: s.color, dashed: !!s.dash}))} />
      <div role="img" aria-label={`${meta.label} against context length for ${series.length} runs; the table below has the values`}>
        <ResponsiveContainer width="100%" height={380}>
          <LineChart data={data} margin={{top: 4, right: 24, bottom: 4, left: 8}}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
            <XAxis dataKey="isl" type="number" scale="log" domain={['auto', 'auto']} tickFormatter={fmtIsl} height={52} label={{value: 'context (ISL, tokens, log)', position: 'insideBottom', offset: 0}} />
            <YAxis width={64} label={{value: meta.label, angle: -90, position: 'insideLeft', style: {textAnchor: 'middle'}}} />
            <Tooltip formatter={((v: unknown) => fmtNum(Number(v))) as never} labelFormatter={(l) => `ISL ${fmtIsl(Number(l))}`} />
            {series.map((s) => (
              <Line key={s.label} type="monotone" dataKey={s.label} stroke={s.color} strokeDasharray={s.dash} strokeWidth={2.4} dot={{r: 3}} isAnimationActive={false} connectNulls />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="tablewrap">
        <table>
          <caption className="sr-only">Values of the chart</caption>
          <thead>
            <tr>
              <th>ISL</th>
              {series.map((s) => <th key={s.label} className="n">{s.label}</th>)}
            </tr>
          </thead>
          <tbody>
            {isls.map((isl) => (
              <tr key={isl}>
                <td>{fmtIsl(isl)}</td>
                {series.map((s) => <td key={s.label} className="n">{fmtNum(s.points.find(([i]) => i === isl)?.[1] ?? NaN)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </ChartCard>
  );
}

export default function Compare(props: {state: Loadable<State>; selected: string[]; now: number; onSelect: (ids: string[]) => void; onRetry: () => void}) {
  const sets = useSetsOf(props.selected);
  return (
    <Async what="runs" data={props.state} onRetry={props.onRetry}>
      {(state) => {
        const options = state.runs.filter(comparable);
        const chosen = options.filter((r) => props.selected.includes(r.id));
        const toggle = (id: string) =>
          props.onSelect(props.selected.includes(id) ? props.selected.filter((x) => x !== id) : [...props.selected, id].slice(-MAX_RUNS));
        if (options.length === 0) return <Empty title="Nothing to compare yet">Runs with at least one gathered result set can be compared here.</Empty>;
        return (
          <div className="compare">
            <fieldset className="picker">
              <legend>Pick up to {MAX_RUNS} runs to overlay</legend>
              {options.map((r) => (
                <label key={r.id} className="check">
                  <input type="checkbox" checked={props.selected.includes(r.id)} onChange={() => toggle(r.id)} />
                  <span>
                    {runGpu(r, state.pods)} · {r.models.map((m) => m.model).join(', ')} <span className="muted mono">{shortId(r.id)} · {formatAgo(r.created_at, props.now)}</span>
                  </span>
                </label>
              ))}
            </fieldset>
            {chosen.length === 0 ? (
              <Empty title="No run picked">Tick two or more runs above to overlay their decode curves.</Empty>
            ) : sets.kind === 'loading' ? (
              <Loading what="result sets" />
            ) : sets.kind === 'error' ? (
              <ErrorBox title="Could not load result sets" message={sets.message} />
            ) : (
              <Chart runs={chosen} pods={state.pods} sets={sets.value} />
            )}
          </div>
        );
      }}
    </Async>
  );
}
