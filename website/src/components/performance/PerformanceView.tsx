import React, { useEffect, useState } from "react";
import data from "@site/src/data/performance.json";
import Lines from "./Lines";
import RunBars from "./RunBars";
import { CHART_METRICS, LOAD_COMPANION, metricInfo, seriesColor } from "./format";
import { chartMetrics, defaultSecond, findHardware, fromQuery, options, resolve, selectPoints, legendSeries, runBars, seriesLines, toQuery, yTop, type Selection, type Side } from "./selection";
import styles from "./performance.module.css";
import type { Hardware, PerformanceData, PointMetric } from "./types";

const report = data as unknown as PerformanceData;

// The Hardware and Model selects of one column, in one row (wrapping when the column is narrow).
function Pickers({ report: data, side, onChange }: { report: PerformanceData; side: Side; onChange: (patch: Partial<Side>) => void }) {
  const opts = options(data, side);
  return (
    <div className={styles.knobs}>
      <label className={styles.knob}>
        <span className={styles.pickerLabel}>Hardware</span>
        <select className={styles.select} value={side.hw} onChange={(e) => onChange({ hw: e.target.value })}>
          {opts.hardware.map((h) => (
            <option key={h.id} value={h.id}>
              {h.label.replace("NVIDIA ", "").replace("AMD Radeon 8060S Graphics", "AMD Radeon 8060S (Strix Halo)")}
            </option>
          ))}
        </select>
      </label>
      <label className={styles.knob}>
        <span className={styles.pickerLabel}>Model</span>
        <select className={styles.select} value={side.model} onChange={(e) => onChange({ model: e.target.value })}>
          {opts.models.map((m) => (
            <option key={m} value={m}>
              {m}
            </option>
          ))}
        </select>
      </label>
    </div>
  );
}

// The metrics a side charts.
function chartsOf(side: Side): PointMetric[] {
  return chartMetrics(report, selectPoints(findHardware(report, side.hw) as Hardware, side));
}

// The rows of the grid a column of comparison mode shares with its partner: pickers, legend, one row per chart slot. A single column has its
// pickers in the top row, beside the Compare switch.
const FIXED_ROWS = 2;

// One column: one chart per metric the side has data for, and the framework bars. `other`
// is the other column's side (comparison mode), whose charts of the same metric share this one's y axis. In comparison mode both columns
// are rows of one grid (`slots`: the metrics either side charts), so a chart sits in the same row, at the same height, as its partner;
// a chart the side lacks leaves its cell empty.
function Column({ side, other, slots, onChange }: { side: Side; other?: Side; slots?: PointMetric[]; onChange: (patch: Partial<Side>) => void }) {
  const compare = !!slots;
  const hw = findHardware(report, side.hw) as Hardware;
  const points = selectPoints(hw, side);
  const otherPoints = other ? selectPoints(findHardware(report, other.hw) as Hardware, other) : [];
  const metrics = chartMetrics(report, points);
  return (
    <div className={styles.column} style={slots ? ({ "--rows": FIXED_ROWS + Math.max(slots.length, 1) } as React.CSSProperties) : undefined}>
      {compare && <Pickers report={report} side={side} onChange={onChange} />}
      <div className={styles.legend}>
        {legendSeries(points).map((s) => (
          <span key={s.id}>
            <span className={styles.swatch} style={{ background: seriesColor(s.id) }} />
            {s.label}
          </span>
        ))}
      </div>
      <div className={compare ? styles.chartsHalf : styles.charts}>
        {CHART_METRICS.filter((m) => (slots ?? metrics).includes(m.key)).map((m) => {
          if (!metrics.includes(m.key)) return <div key={m.key} className={styles.slot} />;
          const info = metricInfo(m.key);
          const direction = `${info.higherIsBetter ? "higher" : "lower"} is better`;
          if (m.wholeRun) {
            const bars = runBars(points, m.key);
            const otherBars = other ? runBars(otherPoints, m.key) : [];
            const top = Math.max(...[...bars, ...otherBars].map((b) => b.value * info.unit.scale));
            return <RunBars key={m.key} title={m.title} subtitle={`${side.model}, ${info.unit.suffix} per run${LOAD_COMPANION[m.key] ? ", peak after the model is loaded and warmed up, load stated beside it" : ""}, ${direction}`} metric={m.key} bars={bars} top={other ? top : undefined} />;
          }
          const lines = seriesLines(points, m.key);
          return (
            <Lines
              key={m.key}
              title={m.title}
              subtitle={`${side.model}, ${info.unit.suffix} over the input tokens, ${direction}`}
              metric={m.key}
              lines={lines}
              top={other ? yTop([lines, seriesLines(otherPoints, m.key)], info.unit.scale) : undefined}
            />
          );
        })}
      </div>
      {(slots ?? metrics).length === 0 && <p className={styles.sentence}>No measurement for this model.</p>}
    </div>
  );
}

// The Performance page: hardware and model selects over the committed measurements, one chart over the input tokens for every metric
// the selection has data for, and the framework bars. The Compare switch, always at the right of the top row, makes two columns, each with its own
// selection. The selection lives in the URL query.
export default function PerformanceView() {
  const [sel, setSel] = useState<Selection>(() => resolve(report, {}));
  useEffect(() => {
    setSel(resolve(report, fromQuery(window.location.search)));
  }, []);
  const apply = (raw: Parameters<typeof resolve>[1]) => {
    const next = resolve(report, raw);
    setSel(next);
    window.history.replaceState(null, "", `${window.location.pathname}?${toQuery(next)}${window.location.hash}`);
  };
  const sideB = sel.b;
  const slots = sideB ? [...new Set([...chartsOf(sel), ...chartsOf(sideB)])] : undefined;
  return (
    <div className={`${styles.root} performance-root`}>
      <div className={styles.top}>
        {!sideB && <Pickers report={report} side={sel} onChange={(patch) => apply({ ...sel, ...patch })} />}
        <label className={styles.switch}>
          <input type="checkbox" role="switch" checked={!!sideB} onChange={(e) => apply({ ...sel, b: e.target.checked ? defaultSecond(report, sel) : undefined })} />
          Compare
        </label>
      </div>
      <div className={sel.b ? styles.sides : undefined}>
        <Column side={sel} other={sideB} slots={slots} onChange={(patch) => apply({ ...sel, ...patch })} />
        {sideB && <Column side={sideB} other={sel} slots={slots} onChange={(patch) => apply({ ...sel, b: { ...sideB, ...patch } })} />}
      </div>
    </div>
  );
}
