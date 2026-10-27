import React from "react";
import { seriesColor } from "./format";
import styles from "./performance.module.css";
import type { TipRow } from "./selection";

// The body of a chart tooltip: an optional title and one row per series (swatch, label, value). The chart places it.
export default function Tooltip(props: { title?: string; rows: TipRow[] }) {
  return (
    <div className={styles.tooltip}>
      {props.title && <div className={styles.tooltipTitle}>{props.title}</div>}
      {props.rows.map((r) => (
        <div key={r.id} className={styles.tooltipRow}>
          <span className={styles.swatch} style={{ background: seriesColor(r.id) }} />
          <span className={styles.tooltipLabel}>{r.label}</span>
          <span className={styles.tooltipValue}>{r.value}</span>
        </div>
      ))}
    </div>
  );
}
