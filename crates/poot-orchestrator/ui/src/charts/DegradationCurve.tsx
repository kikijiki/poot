import React, {useMemo, useState} from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import type {Run, ResultRow, CurvePoint, Pct} from './types';
import {frameworkColor, frameworkLabel} from './types';
import {rowsFor, frameworksInRows} from './data';
import {ToggleGroup, ControlWrap, ChartCard, Empty} from './ui';
import styles from './styles.module.css';

type Metric = 'decode_tok_s' | 'tpot_ms' | 'ttft_ms' | 'itl_ms';
type Pctile = 'p50' | 'p90' | 'p99';

const METRIC_OPTS: {value: Metric; label: string}[] = [
  {value: 'decode_tok_s', label: 'decode tok/s'},
  {value: 'tpot_ms', label: 'TPOT (ms)'},
  {value: 'itl_ms', label: 'ITL (ms)'},
  {value: 'ttft_ms', label: 'TTFT (ms)'},
];

const PCT_OPTS: {value: Pctile; label: string}[] = [
  {value: 'p50', label: 'p50'},
  {value: 'p90', label: 'p90'},
  {value: 'p99', label: 'p99'},
];

function metricValue(pt: CurvePoint, metric: Metric, pct: Pctile): number {
  if (metric === 'decode_tok_s') return pt.decode_tok_s;
  const field = pt[metric] as Pct;
  return field ? field[pct] : NaN;
}

function isCurveRow(r: ResultRow): boolean {
  return r.status === 'ok' && Array.isArray(r.curve) && r.curve.length > 0;
}

export default function DegradationCurve(props: {run: Run; model: string}) {
  const {run, model} = props;
  const [metric, setMetric] = useState<Metric>('decode_tok_s');
  const [pct, setPct] = useState<Pctile>('p50');

  const rows = rowsFor(run, model, 'decode-curve').filter(isCurveRow);
  const frameworks = frameworksInRows(rows);

  // Build a wide table keyed by isl, one column per framework.
  const data = useMemo(() => {
    const byIsl = new Map<number, Record<string, number>>();
    for (const row of rows) {
      for (const pt of row.curve!) {
        const e = byIsl.get(pt.isl) ?? {isl: pt.isl};
        e[row.framework] = metricValue(pt, metric, pct);
        byIsl.set(pt.isl, e);
      }
    }
    return Array.from(byIsl.values()).sort((a, b) => a.isl - b.isl);
  }, [rows, metric, pct]);

  if (rows.length === 0) {
    return (
      <ChartCard title="Context-degradation curve">
        <Empty>
          No <code>decode-curve</code> data in this run for <b>{model}</b>.
        </Empty>
      </ChartCard>
    );
  }

  const showPct = metric !== 'decode_tok_s';
  const yLabel =
    metric === 'decode_tok_s'
      ? 'decode tok/s (1000 / median TPOT)'
      : METRIC_OPTS.find((m) => m.value === metric)!.label;

  return (
    <ChartCard
      title="Context-degradation curve (headline)"
      sub={
        <>
          {yLabel} vs context length (ISL), one line per framework. poot uses
          naive attention, so its per-token cost rises with context (the curve
          bends); flash/paged baselines stay flat.
        </>
      }>
      <div className={styles.controls}>
        <ControlWrap label="metric">
          <ToggleGroup options={METRIC_OPTS} value={metric} onChange={setMetric} />
        </ControlWrap>
        {showPct ? (
          <ControlWrap label="percentile">
            <ToggleGroup options={PCT_OPTS} value={pct} onChange={setPct} />
          </ControlWrap>
        ) : null}
      </div>
      <ResponsiveContainer width="100%" height={380}>
        <LineChart key={`${metric}:${pct}`} data={data} margin={{top: 8, right: 24, bottom: 24, left: 8}}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            dataKey="isl"
            scale="log"
            domain={['auto', 'auto']}
            type="number"
            tickFormatter={fmtIsl}
            label={{value: 'context length (ISL, tokens, log)', position: 'bottom', offset: 8}}
          />
          <YAxis
            label={{value: yLabel, angle: -90, position: 'insideLeft', style: {textAnchor: 'middle'}}}
            width={64}
          />
          <Tooltip
            formatter={((v: unknown, name: unknown) => [
              fmtNum(Number(v)),
              frameworkLabel(String(name)),
            ]) as never}
            labelFormatter={(l) => `ISL ${fmtIsl(Number(l))}`}
          />
          <Legend formatter={(v) => frameworkLabel(String(v))} />
          {frameworks.map((f) => (
            <Line
              key={f}
              type="monotone"
              dataKey={f}
              name={f}
              stroke={frameworkColor(f)}
              strokeWidth={f === 'poot' ? 3.5 : 1.8}
              dot={{r: f === 'poot' ? 3 : 2}}
              activeDot={{r: 5}}
              isAnimationActive={false}
              connectNulls
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </ChartCard>
  );
}

function fmtIsl(n: number): string {
  if (n >= 1024) return `${Math.round(n / 1024)}k`;
  return String(n);
}

function fmtNum(n: number): string {
  if (!isFinite(n)) return '-';
  return n >= 100 ? n.toFixed(0) : n.toFixed(1);
}
