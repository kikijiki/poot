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
import type {Run, ResultRow, UtilSample} from './types';
import {frameworkColor, frameworkLabel, bytesToGiB} from './types';
import {rowsFor, frameworksInRows} from './data';
import {ChartCard, Empty, ControlWrap, ToggleGroup} from './ui';
import styles from './styles.module.css';

// Rows in this run and model with resource-utilization summaries (recorded on every ok cell).
function utilRows(run: Run, model: string): ResultRow[] {
  const seen = new Set<string>();
  const out: ResultRow[] = [];
  for (const scenario of new Set(
    run.results.filter((r) => r.model === model).map((r) => r.scenario),
  )) {
    for (const r of rowsFor(run, model, scenario)) {
      if (r.status !== 'ok') continue;
      if (
        r.gpu_util_mean_pct == null &&
        r.cpu_util_mean_pct == null &&
        r.peak_vram_bytes == null &&
        r.peak_rss_bytes == null
      ) {
        continue;
      }
      if (seen.has(r.framework)) continue;
      seen.add(r.framework);
      out.push(r);
    }
  }
  return out;
}

function fmtPct(n: number | undefined | null): string {
  if (n == null || !isFinite(n)) return '-';
  return n >= 100 ? n.toFixed(0) : n.toFixed(1);
}

function fmtGiB(bytes: number | undefined | null): string {
  const g = bytesToGiB(bytes);
  if (g == null) return '-';
  return g.toFixed(2);
}

function fmtW(n: number | undefined | null): string {
  if (n == null || !isFinite(n)) return '-';
  return n.toFixed(0);
}

function fmtWh(n: number | undefined | null): string {
  if (n == null || !isFinite(n)) return '-';
  return n.toFixed(3);
}

export default function ResourceUtil(props: {run: Run; model: string}) {
  const {run, model} = props;
  const rows = useMemo(() => utilRows(run, model), [run, model]);

  // Frameworks that actually have a time series to plot.
  const seriesFw = useMemo(
    () =>
      frameworksInRows(
        rows.filter((r) => Array.isArray(r.util_series) && r.util_series.length > 0),
      ),
    [rows],
  );

  const [selFw, setSelFw] = useState<string>('');
  const effFw = seriesFw.includes(selFw as never)
    ? selFw
    : seriesFw.includes('poot' as never)
      ? 'poot'
      : seriesFw[0] ?? '';

  const orderedFw = useMemo(() => frameworksInRows(rows), [rows]);

  // Time-series data for the selected framework: gpu_pct and cpu_pct vs t_s.
  const seriesRow = rows.find((r) => r.framework === effFw);
  const seriesData = useMemo(() => {
    const s = seriesRow?.util_series;
    if (!Array.isArray(s)) return [];
    return s.map((p: UtilSample) => ({
      t: p[0],
      gpu: p[1],
      cpu: p[2],
    }));
  }, [seriesRow]);

  // All hooks are above this guard so the hook order stays stable.
  if (rows.length === 0) {
    return null;
  }

  return (
    <ChartCard
      title="Resource utilization"
      sub={
        <>
          Per-engine GPU and CPU utilization plus peak memory and GPU power/energy,
          sampled by the harness over the whole cell. CPU is process-tree percent
          where 100% is one full core, so it can exceed 100. This is where the
          host-bound story shows: poot can sit at a fraction of GPU while a CPU core
          pegs near 100%.
        </>
      }>
      <div style={{overflowX: 'auto'}}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>engine</th>
              <th>GPU % mean</th>
              <th>GPU % peak</th>
              <th>CPU % mean</th>
              <th>CPU % peak</th>
              <th>peak VRAM (GiB)</th>
              <th>peak RAM (GiB)</th>
              <th>peak power (W)</th>
              <th>energy (Wh)</th>
            </tr>
          </thead>
          <tbody>
            {orderedFw.map((f) => {
              const r = rows.find((row) => row.framework === f)!;
              return (
                <tr key={f}>
                  <td>
                    <span
                      className={styles.swatch}
                      style={{background: frameworkColor(f)}}
                    />
                    {frameworkLabel(f)}
                  </td>
                  <td>{fmtPct(r.gpu_util_mean_pct)}</td>
                  <td>{fmtPct(r.gpu_util_peak_pct)}</td>
                  <td>{fmtPct(r.cpu_util_mean_pct)}</td>
                  <td>{fmtPct(r.cpu_util_peak_pct)}</td>
                  <td>{fmtGiB(r.peak_vram_bytes)}</td>
                  <td>{fmtGiB(r.peak_rss_bytes)}</td>
                  <td>{fmtW(r.peak_power_w)}</td>
                  <td>{fmtWh(r.energy_wh)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className={styles.chartSub} style={{marginTop: '0.6rem', marginBottom: '0.4rem'}}>
        CPU % uses 100% = 1 full core (process-tree), so multi-threaded engines exceed
        100%. VRAM/RAM in GiB = bytes / 2^30. Peak power and energy are NVML board
        power draw, integrated over wall-clock time; NVIDIA-only, blank elsewhere.
      </p>

      {seriesFw.length > 0 && effFw ? (
        <>
          <div className={styles.controls}>
            <ControlWrap label="engine trace">
              <ToggleGroup
                options={seriesFw.map((f) => ({value: f, label: frameworkLabel(f)}))}
                value={effFw}
                onChange={setSelFw}
              />
            </ControlWrap>
          </div>
          <div className={styles.chartSub} style={{marginBottom: '0.6rem'}}>
            Utilization over time for <b>{frameworkLabel(effFw)}</b> (
            {seriesData.length} samples, ~1 Hz). The series spans every ISL of the
            cell, with no per-ISL markers; engine series differ wildly in length.
          </div>
          {seriesData.length === 0 ? (
            <Empty>No time-series samples for this engine.</Empty>
          ) : (
            <ResponsiveContainer width="100%" height={320}>
              <LineChart data={seriesData} margin={{top: 8, right: 24, bottom: 24, left: 8}}>
                <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
                <XAxis
                  dataKey="t"
                  type="number"
                  domain={['auto', 'auto']}
                  tickFormatter={(v) => `${Math.round(Number(v))}s`}
                  label={{value: 'time since cell start (s)', position: 'bottom', offset: 8}}
                />
                <YAxis
                  label={{
                    value: 'utilization %',
                    angle: -90,
                    position: 'insideLeft',
                    style: {textAnchor: 'middle'},
                  }}
                  width={64}
                />
                <Tooltip
                  formatter={((v: unknown, name: unknown) => [
                    v == null ? '-' : `${fmtPct(Number(v))}%`,
                    String(name) === 'gpu' ? 'GPU %' : 'CPU %',
                  ]) as never}
                  labelFormatter={(l) => `t = ${Number(l).toFixed(1)}s`}
                />
                <Legend formatter={(v) => (String(v) === 'gpu' ? 'GPU %' : 'CPU % (100% = 1 core)')} />
                <Line
                  type="monotone"
                  dataKey="gpu"
                  name="gpu"
                  stroke={frameworkColor(effFw)}
                  strokeWidth={2.4}
                  dot={false}
                  isAnimationActive={false}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="cpu"
                  name="cpu"
                  stroke="#888888"
                  strokeWidth={1.8}
                  strokeDasharray="5 3"
                  dot={false}
                  isAnimationActive={false}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </>
      ) : null}
    </ChartCard>
  );
}
