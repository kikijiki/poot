import React from "react";
import { CartesianGrid, ErrorBar, Line as RLine, LineChart, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { metricInfo, num, seriesColor } from "./format";
import styles from "./performance.module.css";
import { serveRowsAt, serveTop, type ServeLine } from "./selection";
import Tooltip from "./Tooltip";
import type { PointMetric } from "./types";

const PLOT_H = 250;
const TICKS = 4;

// One row of the chart data per concurrency: the median of every series there and, for a point of one host, its min..max as the
// distances below and above the median.
type Row = { concurrency: number } & Record<string, number | [number, number] | undefined>;

function chartData(lines: ServeLine[]): Row[] {
  const rows = new Map<number, Row>();
  for (const l of lines) {
    for (const p of l.points) {
      const row = rows.get(p.concurrency) ?? { concurrency: p.concurrency };
      row[l.id] = p.stat.median;
      if (p.point.hosts === 1) row[`${l.id}:err`] = [p.stat.median - p.stat.min, p.stat.max - p.stat.median];
      rows.set(p.concurrency, row);
    }
  }
  return [...rows.values()].sort((a, b) => a.concurrency - b.concurrency);
}

// One line per series over the requests in flight, a marker per concurrency measured and min..max whiskers (of one host). The x axis
// holds exactly the concurrencies the data has. An unverified point is a hollow marker and its segments are dashed.
export default function ServeLines(props: { title: string; subtitle: string; metric: PointMetric; lines: ServeLine[] }) {
  const { lines } = props;
  const unit = metricInfo(props.metric).unit;
  const top = serveTop(lines, unit.scale);
  const levels = [...new Set(lines.flatMap((l) => l.points.map((p) => p.concurrency)))].sort((a, b) => a - b);
  const yTicks = Array.from({ length: TICKS + 1 }, (_, i) => (top * i) / TICKS / unit.scale);
  return (
    <section className={styles.frame}>
      <h2 className={styles.frameTitle}>{props.title}</h2>
      <div className={styles.frameSub}>{props.subtitle}</div>
      <div
        className={styles.plot}
        role="img"
        aria-label={`${props.title}: ${lines.map((l) => `${l.label}: ${l.points.map((p) => `${p.concurrency} requests ${num(p.stat.median, unit)} ${unit.suffix}`).join(", ")}`).join("; ")}`}
      >
        <ResponsiveContainer width="100%" height={PLOT_H} initialDimension={{ width: 640, height: PLOT_H }}>
          <LineChart data={chartData(lines)} margin={{ top: 14, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid stroke="var(--border)" />
            <XAxis
              dataKey="concurrency"
              type="number"
              scale="linear"
              domain={[0, Math.max(...levels) + 1]}
              ticks={levels}
              allowDataOverflow
              tick={{ className: styles.axisText }}
              label={{ value: "Concurrent requests", position: "insideBottom", offset: -2, className: styles.axisText }}
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
              content={({ active, label }) => (active && typeof label === "number" ? <Tooltip title={`${label} concurrent requests`} rows={serveRowsAt(lines, label, unit)} /> : null)}
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
                  const verified = l.points.find((p) => p.concurrency === payload.concurrency)?.point.verified;
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
