// The series of the Compare view: one decode curve per run, from the result sets of each.

import type {Snapshot} from '../api';
import type {CurvePoint, Pct, ResultRow} from './types';

export type Metric = 'decode_tok_s' | 'tpot_ms' | 'ttft_ms';

export const METRICS: {value: Metric; label: string; higherIsBetter: boolean}[] = [
  {value: 'decode_tok_s', label: 'decode tok/s', higherIsBetter: true},
  {value: 'tpot_ms', label: 'TPOT p50 (ms)', higherIsBetter: false},
  {value: 'ttft_ms', label: 'TTFT p50 (ms)', higherIsBetter: false},
];

function value(pt: CurvePoint, metric: Metric): number {
  return metric === 'decode_tok_s' ? pt.decode_tok_s : ((pt[metric] as Pct | undefined)?.p50 ?? NaN);
}

export function median(xs: number[]): number {
  const v = xs.filter(Number.isFinite).sort((a, b) => a - b);
  if (v.length === 0) return NaN;
  const mid = Math.floor(v.length / 2);
  return v.length % 2 ? v[mid] : (v[mid - 1] + v[mid]) / 2;
}

const curveRows = (sets: Snapshot[], model: string, framework: string): ResultRow[] =>
  sets
    .filter((s) => !s.failed)
    .flatMap((s) => s.rows)
    .filter((r) => r.status === 'ok' && r.model === model && r.framework === framework && r.scenario === 'decode-curve' && (r.curve?.length ?? 0) > 0);

// The median over a run's repeats of `metric` at each context length: [isl, value] ascending by isl.
export function curveOf(sets: Snapshot[], model: string, framework: string, metric: Metric): [number, number][] {
  const byIsl = new Map<number, number[]>();
  for (const row of curveRows(sets, model, framework)) {
    for (const pt of row.curve ?? []) {
      byIsl.set(pt.isl, [...(byIsl.get(pt.isl) ?? []), value(pt, metric)]);
    }
  }
  return [...byIsl.entries()].map(([isl, vs]): [number, number] => [isl, median(vs)]).sort((a, b) => a[0] - b[0]);
}

// Models and frameworks that at least one of the runs measured a curve for, to fill the pickers.
export function curveChoices(setsPerRun: Snapshot[][]): {models: string[]; frameworks: string[]} {
  const rows = setsPerRun.flatMap((sets) => sets.filter((s) => !s.failed).flatMap((s) => s.rows)).filter((r) => r.status === 'ok' && (r.curve?.length ?? 0) > 0);
  return {
    models: [...new Set(rows.map((r) => r.model))].sort(),
    frameworks: [...new Set(rows.map((r) => r.framework))].sort(),
  };
}
