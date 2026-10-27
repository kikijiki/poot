import React from "react";
import styles from "./architecture.module.css";

type Layer = {
  label: string;
  sublabel: string;
  color: "compile" | "runtime";
};

const BRANCHES: Layer[] = [
  { label: "wgpu", sublabel: "llc -> SPIR-V -> shader module", color: "runtime" },
  { label: "Raw Vulkan", sublabel: "llc -> SPIR-V -> ash pipeline", color: "runtime" },
  { label: "PTX", sublabel: "llc -> PTX -> cudarc JIT", color: "runtime" },
  { label: "ROCm", sublabel: "clang -> HSACO -> HSA queue", color: "runtime" },
];

const COLOR_CLASS = {
  compile: styles.layerCompile,
  runtime: styles.layerRuntime,
};

function Box({ layer, href }: { layer: Layer; href?: string }) {
  return (
    <a
      href={href}
      className={`${styles.layerBox} ${COLOR_CLASS[layer.color]}`}
    >
      <span className={styles.layerLabel}>{layer.label}</span>
      <span className={styles.layerSublabel}>{layer.sublabel}</span>
    </a>
  );
}

export default function BackendDiagram(): React.ReactElement {
  return (
    <div className={styles.pipelineWrap}>
      <div className={styles.pipelineTitle}>From kernel IR to each backend</div>
      <div className={styles.pipeline}>
        <Box
          layer={{
            label: "Kernel IR",
            sublabel: "one source for every backend",
            color: "compile",
          }}
          href="./kernel-import"
        />
        <div className={styles.arrow} />
        <Box
          layer={{
            label: "poot-codegen",
            sublabel: "LLVM IR, per-target address spaces and intrinsics",
            color: "compile",
          }}
        />
        <div className={styles.arrow} />
        <div className={styles.forkRow}>
          {BRANCHES.map((b) => (
            <div key={b.label} className={styles.forkBranch}>
              <Box layer={b} />
            </div>
          ))}
        </div>
        <div className={styles.arrow} />
        <Box
          layer={{
            label: "Executor",
            sublabel: "one Engine over a per-backend Device",
            color: "runtime",
          }}
          href="./execution"
        />
      </div>
    </div>
  );
}
