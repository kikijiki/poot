import React, {useMemo, useState} from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import type {Run, ResultRow} from './types';
import {frameworkColor, frameworkLabel, bytesToGiB} from './types';
import {rowsFor} from './data';
import {ToggleGroup, ControlWrap, ChartCard, Empty, Select} from './ui';
import styles from './styles.module.css';

type Metric = 'decode_tok_s' | 'ttft_ms' | 'vram';

const METRIC_OPTS: {value: Metric; label: string}[] = [
  {value: 'decode_tok_s', label: 'decode tok/s'},
  {value: 'ttft_ms', label: 'TTFT (ms)'},
  {value: 'vram', label: 'peak VRAM (GiB)'},
];

function metricValue(r: ResultRow, m: Metric): number | null {
  if (m === 'vram') return bytesToGiB(r.peak_vram_bytes);
  const v = r[m];
  return typeof v === 'number' ? v : null;
}

export default function SinglePointBars(props: {run: Run; model: string}) {
  const {run, model} = props;

  const scenarios = useMemo(
    () =>
      ['decode-128', 'prefill-1k'].filter((s) =>
        run.results.some((r) => r.scenario === s && r.model === model),
      ),
    [run, model],
  );
  const [scenario, setScenario] = useState(scenarios[0] ?? '');
  const [metric, setMetric] = useState<Metric>('decode_tok_s');

  if (scenarios.length === 0) {
    return (
      <ChartCard title="Single-point scenarios">
        <Empty>
          No <code>decode-128</code> / <code>prefill-1k</code> rows in this run for <b>{model}</b>.
        </Empty>
      </ChartCard>
    );
  }

  // prefill-1k has no decode rate; restrict its metric set.
  const metricOpts =
    scenario === 'prefill-1k'
      ? METRIC_OPTS.filter((m) => m.value !== 'decode_tok_s')
      : METRIC_OPTS;
  // Effective metric (falls back when the selected one is not in this scenario's set). Chart data
  // must derive from this, not the raw `metric` state, or a stale metric renders under the new label.
  const effMetric = metricOpts.some((m) => m.value === metric) ? metric : metricOpts[0].value;
  const label = METRIC_OPTS.find((m) => m.value === effMetric)!.label;

  const rows = scenario ? rowsFor(run, model, scenario).filter((r) => r.status === 'ok') : [];
  const data = rows
    .map((r) => ({framework: r.framework, value: metricValue(r, effMetric)}))
    .filter((d) => d.value != null && isFinite(d.value as number));

  return (
    <ChartCard
      title="Single-point scenarios"
      sub={`Grouped per framework. ${scenario}: a fixed short workload (near-zero context for decode-128). VRAM in GiB = bytes / 2^30.`}>
      <div className={styles.controls}>
        <Select
          label="scenario"
          value={scenario}
          onChange={setScenario}
          options={scenarios.map((s) => ({value: s, label: s}))}
        />
        <ControlWrap label="metric">
          <ToggleGroup options={metricOpts} value={effMetric} onChange={setMetric} />
        </ControlWrap>
      </div>
      {data.length === 0 ? (
        <Empty>No values for this metric.</Empty>
      ) : (
        <ResponsiveContainer width="100%" height={320}>
          <BarChart key={`${scenario}:${effMetric}`} data={data} margin={{top: 8, right: 16, bottom: 8, left: 8}}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.3} vertical={false} />
            <XAxis dataKey="framework" tickFormatter={frameworkLabel} />
            <YAxis
              label={{value: label, angle: -90, position: 'insideLeft', style: {textAnchor: 'middle'}}}
              width={64}
            />
            <Tooltip
              cursor={{fill: 'rgba(128,128,128,0.1)'}}
              formatter={((v: unknown) => [fmtNum(Number(v)), label]) as never}
              labelFormatter={(l) => frameworkLabel(String(l))}
            />
            <Bar dataKey="value" isAnimationActive={false} radius={[4, 4, 0, 0]}>
              {data.map((d) => (
                <Cell key={d.framework} fill={frameworkColor(d.framework)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
      {effMetric === 'vram' ? (
        <p className={styles.chartSub} style={{marginTop: '0.5rem'}}>
          Note: vLLM pre-allocates a large KV-cache pool, so its peak VRAM reflects the pool size,
          not the weight footprint. See caveats below.
        </p>
      ) : null}
    </ChartCard>
  );
}

function fmtNum(n: number): string {
  if (!isFinite(n)) return '-';
  if (n >= 100) return n.toFixed(0);
  if (n >= 10) return n.toFixed(1);
  return n.toFixed(2);
}
