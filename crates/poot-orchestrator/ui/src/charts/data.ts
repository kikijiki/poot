// Helpers over a Run object. Adapted from the website's data.ts; Runs come from the live
// /api/results response instead of bundled JSON.

import type {Run, ResultRow, Framework} from './types';
import {FRAMEWORK_ORDER} from './types';

// Models present in a run, sorted.
export function modelsInRun(run: Run): string[] {
  const set = new Set<string>();
  for (const r of run.results) {
    if (r.model) set.add(r.model);
  }
  return Array.from(set).sort();
}

export function scenariosInRun(run: Run): string[] {
  const set = new Set<string>();
  for (const r of run.results) {
    if (r.scenario) set.add(r.scenario);
  }
  return Array.from(set);
}

export function rowsFor(run: Run, model: string, scenario: string): ResultRow[] {
  return run.results
    .filter((r) => r.model === model && r.scenario === scenario)
    .sort(
      (a, b) =>
        FRAMEWORK_ORDER.indexOf(a.framework as Framework) -
        FRAMEWORK_ORDER.indexOf(b.framework as Framework),
    );
}

export function frameworksInRows(rows: ResultRow[]): Framework[] {
  const present = new Set(rows.map((r) => r.framework));
  return FRAMEWORK_ORDER.filter((f) => present.has(f));
}
