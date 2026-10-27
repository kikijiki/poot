import React from "react";
import { CartesianGrid, ErrorBar, Line as RLine, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { metricInfo, num, seriesColor } from "./format";
import styles from "./performance.module.css";
import { rowsAt, yTop, type Line } from "./selection";
import Tooltip from "./Tooltip";
import type { PointMetric } from "./types";

const PLOT_H = 250;
const TICKS = 4;
const X_TICKS = [128, 512, 2048, 8192];

// One row of the chart data per input length: the median of every series there and, for a point of one host, its min..max as the
// distances below and above the median (a point over several hosts shows no whisker).
type Row = { isl: number } & Record<string, number | [number, number] | undefined>;

function chartData(lines: Line[]): Row[] {
  const rows = new Map<number, Row>();
  for (const l of lines) {
    for (const p of l.points) {
      const row = rows.get(p.isl) ?? { isl: p.isl };
      row[l.id] = p.stat.median;
      if (p.point.hosts === 1) row[`${l.id}:err`] = [p.stat.median - p.stat.min, p.stat.max - p.stat.median];
      rows.set(p.isl, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.isl - b.isl);
}

// One line per series over the input tokens (log scale), a marker per point and min..max whiskers (of one host). Pointing anywhere on
// the plot (or tapping it) marks the input length nearest the pointer and lists every series at it in a tooltip. An unverified point is a
// hollow marker and its segments are dashed.
// `top` is the top of the y axis in the unit's scale; without it the axis fits this chart's lines.
export default function Lines(props: { title: string; subtitle: string; metric: PointMetric; lines: Line[]; top?: number }) {
  const { lines } = props;
  const unit = metricInfo(props.metric).unit;
  const top = props.top ?? yTop([lines], unit.scale);
  const isls = lines.flatMap((l) => l.points.map((p) => p.isl));
  const lo = Math.min(...isls) / 2 ** 0.25;
  const hi = Math.max(...isls) * 2 ** 0.25;
  const yTicks = Array.from({ length: TICKS + 1 }, (_, i) => (top * i) / TICKS / unit.scale);
  return (
    <section className={styles.frame}>
      <h2 className={styles.frameTitle}>{props.title}</h2>
      <div className={styles.frameSub}>{props.subtitle}</div>
      <div
        className={styles.plot}
        role="img"
        aria-label={`${props.title}: ${lines.map((l) => `${l.label}: ${l.points.map((p) => `${p.isl} tokens ${num(p.stat.median, unit)} ${unit.suffix}`).join(", ")}`).join("; ")}`}
      >
        <ResponsiveContainer width="100%" height={PLOT_H} initialDimension={{ width: 640, height: PLOT_H }}>
          <LineChart data={chartData(lines)} margin={{ top: 14, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--border)" />
            <XAxis
              dataKey="isl"
              type="number"
              scale="log"
              domain={[lo, hi]}
              ticks={X_TICKS.filter((t) => t >= lo && t <= hi)}
              allowDataOverflow
              tick={{ className: styles.axisText }}
              label={{ value: "Input tokens", position: "insideBottom", offset: -2, className: styles.axisText }}
              height={44}
            />
            <YAxis
              type="number"
              domain={[0, top / unit.scale]}
              ticks={yTicks}
              tickFormatter={(v: number) => num(v, unit)}
              tick={{ className: styles.axisText }}
              width={52}
            />
            <RTooltip
              isAnimationActive={false}
              cursor={{ stroke: "var(--text-secondary)", strokeDasharray: "3 3" }}
              allowEscapeViewBox={{ x: false, y: false }}
              wrapperStyle={{ zIndex: 2, outline: "none" }}
              content={({ active, label }) => (active && typeof label === "number" ? <Tooltip title={`${label} input tokens`} rows={rowsAt(lines, label, unit)} /> : null)}
            />
            {lines.map((l) => (
              <RLine
                key={l.id}
                dataKey={l.id}
                stroke={seriesColor(l.id)}
                strokeWidth={2}
                strokeDasharray={l.verified ? undefined : "5 3"}
                connectNulls
                isAnimationActive={false}
                dot={({ cx, cy, payload, index }) => {
                  const verified = l.points.find((p) => p.isl === payload.isl)?.point.verified;
                  return <circle key={index} cx={cx} cy={cy} r={4.5} fill={verified ? seriesColor(l.id) : "var(--surface)"} stroke={verified ? "var(--surface)" : seriesColor(l.id)} strokeWidth={2} />;
                }}
                activeDot={{ r: 6.5, stroke: "var(--surface)", strokeWidth: 2 }}
              >
                <ErrorBar dataKey={`${l.id}:err`} stroke={seriesColor(l.id)} strokeWidth={1.5} width={8} />
              </RLine>
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      {lines.some((l) => !l.verified) && <div className={styles.frameNote}>Dashed, hollow: unverified, noisy host</div>}
    </section>
  );
}
