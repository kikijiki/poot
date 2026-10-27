import React from "react";
import { Bar as RBar, BarChart, Cell, LabelList, ResponsiveContainer, Tooltip as RTooltip, XAxis, YAxis } from "recharts";
import { metricInfo, num, seriesColor } from "./format";
import styles from "./performance.module.css";
import type { Bar } from "./selection";
import Tooltip from "./Tooltip";
import type { PointMetric } from "./types";

const BAR_H = 16;
const BAR_GAP = 6;

// A whole-run metric: one value per run, so one horizontal bar per framework and backend and no input-length axis. An unverified
// point is a muted bar. Hovering a bar shows its label and value. `top` is the end of the axis in the unit's scale; without it the axis
// fits this chart's bars.
export default function RunBars(props: { title: string; subtitle: string; metric: PointMetric; bars: Bar[]; top?: number }) {
  const { bars } = props;
  const info = metricInfo(props.metric);
  const top = props.top ?? Math.max(...bars.map((b) => b.value * info.unit.scale));
  const data = bars.map((b) => ({ ...b, shown: b.value * info.unit.scale, text: `${num(b.value, info.unit)}${b.note ? ` (${b.note})` : ""}` }));
  return (
    <section className={styles.frame}>
      <h2 className={styles.frameTitle}>{props.title}</h2>
      <div className={styles.frameSub}>{props.subtitle}</div>
      <div role="img" aria-label={`${props.title}: ${bars.map((b) => `${b.label} ${num(b.value, info.unit)} ${info.unit.suffix}${b.note ? `, ${b.note}` : ""}`).join(", ")}`}>
        <ResponsiveContainer width="100%" height={bars.length * (BAR_H + BAR_GAP)} initialDimension={{ width: 640, height: bars.length * (BAR_H + BAR_GAP) }}>
          <BarChart data={data} layout="vertical" barCategoryGap={BAR_GAP} margin={{ top: 0, right: 200, bottom: 0, left: 8 }}>
            <XAxis type="number" domain={[0, top]} hide />
            <YAxis type="category" dataKey="id" hide />
            <RTooltip
              isAnimationActive={false}
              cursor={false}
              allowEscapeViewBox={{ x: false, y: false }}
              wrapperStyle={{ zIndex: 2, outline: "none" }}
              content={({ active, payload }) => {
                const b = active ? (payload?.[0]?.payload as Bar | undefined) : undefined;
                return b ? <Tooltip rows={[{ id: b.id, label: b.label, value: `${num(b.value, info.unit)} ${info.unit.suffix}${b.note ? `, ${b.note}` : ""}` }]} /> : null;
              }}
            />
            <RBar dataKey="shown" radius={2} minPointSize={2} isAnimationActive={false}>
              {data.map((b) => (
                <Cell key={b.id} fill={seriesColor(b.id)} fillOpacity={b.verified ? 1 : 0.35} stroke={b.verified ? undefined : seriesColor(b.id)} strokeDasharray={b.verified ? undefined : "4 2"} />
              ))}
              <LabelList dataKey="text" position="right" className={styles.axisText} />
            </RBar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
