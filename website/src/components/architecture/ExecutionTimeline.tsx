import React, { useState } from "react";
import styles from "./architecture.module.css";

type Strategy = {
  id: string;
  label: string;
  description: string;
  dispatches: number;
  hostCalls: number;
  color: string;
};

const STRATEGIES: Strategy[] = [
  {
    id: "naive",
    label: "Naive",
    description:
      "One dispatch per primitive eqn, per token: over a thousand for Qwen2.5-0.5B (illustrative count).",
    dispatches: 1400,
    hostCalls: 1400,
    color: "var(--ifm-color-danger)",
  },
  {
    id: "fused",
    label: "Fused",
    description:
      "Pointwise, reduction, and flash-attention fusion: tens of dispatches (illustrative count).",
    dispatches: 80,
    hostCalls: 80,
    color: "var(--ifm-color-warning)",
  },
  {
    id: "captured",
    label: "Native replay",
    description:
      "Illustrative native replay: one graph launch for the recorded dispatch sequence on PTX. ROCm replays AQL packets; wgpu re-encodes cached dispatches and may split submissions. Full token steps also update inputs and read outputs.",
    dispatches: 80,
    hostCalls: 1,
    color: "var(--ifm-color-success)",
  },
];

const MAX_DISPATCHES = 1400;

export default function ExecutionTimeline(): React.ReactElement {
  const [active, setActive] = useState<string>("captured");
  const strategy = STRATEGIES.find((s) => s.id === active)!;

  return (
    <div className={styles.timelineWrap}>
      <div className={styles.timelineTabs}>
        {STRATEGIES.map((s) => (
          <button
            key={s.id}
            className={`${styles.timelineTab} ${active === s.id ? styles.timelineTabActive : ""}`}
            onClick={() => setActive(s.id)}
            style={
              active === s.id
                ? { borderColor: s.color, color: s.color }
                : undefined
            }
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className={styles.timelineBody}>
        <div className={styles.timelineDesc}>{strategy.description}</div>
        <div className={styles.timelineRow}>
          <span className={styles.timelineMetaLabel}>Dispatches per token</span>
          <div className={styles.timelineBar}>
            <div
              className={styles.timelineBarFill}
              style={{
                width: `${(strategy.dispatches / MAX_DISPATCHES) * 100}%`,
                background: strategy.color,
              }}
            />
          </div>
          <span className={styles.timelineMetaValue}>
            {strategy.dispatches.toLocaleString()}
          </span>
        </div>
        <div className={styles.timelineRow}>
          <span className={styles.timelineMetaLabel}>Dispatch or replay calls</span>
          <div className={styles.timelineBar}>
            <div
              className={styles.timelineBarFill}
              style={{
                width: `${(strategy.hostCalls / MAX_DISPATCHES) * 100}%`,
                background: strategy.color,
                minWidth: strategy.hostCalls <= 10 ? 4 : undefined,
              }}
            />
          </div>
          <span className={styles.timelineMetaValue}>
            {strategy.hostCalls.toLocaleString()}
          </span>
        </div>
        <div
          style={{
            fontSize: "0.72rem",
            color: "var(--ifm-color-emphasis-600)",
            marginTop: 6,
          }}
        >
          Counts are illustrative and exclude slot writes, sampling and readback.{" "}
          <code>poot_graph_ir::analysis::dispatch_count</code> estimates graph dispatches, not
          host calls or token latency.
        </div>
      </div>
    </div>
  );
}
