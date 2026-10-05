import React, { useState } from "react";
import styles from "./architecture.module.css";

type Layer = {
  id: string;
  label: string;
  sublabel?: string;
  href?: string;
  color: "model" | "ir" | "compile" | "runtime";
};

type Fork = { fork: Layer[] };

type Step = Layer | Fork;

const STEPS: Step[] = [
  {
    id: "model",
    label: "Model definition",
    sublabel: "Rust tracer fns over Builder + ops::*",
    href: "./tracing",
    color: "model",
  },
  {
    id: "tracer",
    label: "Builder",
    sublabel: "records SSA eqns into graph",
    href: "./tracing",
    color: "model",
  },
  {
    id: "graph",
    label: "Graph IR",
    sublabel: "flat SSA of primitive eqns",
    href: "./graph-ir",
    color: "ir",
  },
  {
    id: "optimize",
    label: "compile(graph, target, options)",
    sublabel: "passes, packed claims, legalization, planning",
    href: "./fusion",
    color: "ir",
  },
  {
    id: "plan",
    label: "Program",
    sublabel: "equation plans, storage, slots and numerics",
    href: "./fusion",
    color: "ir",
  },
  {
    fork: [
      {
        id: "pootc",
        label: "pootc",
        sublabel: "#[kernel] fn -> Stable MIR -> kernel IR",
        href: "./kernel-import",
        color: "compile",
      },
      {
        id: "kernelgen",
        label: "poot-kernelgen",
        sublabel: "kernel IR synthesized from shapes",
        href: "./fusion",
        color: "compile",
      },
    ],
  },
  {
    id: "codegen",
    label: "poot-codegen",
    sublabel: "kernel IR -> LLVM IR (one shared emitter)",
    href: "./backends",
    color: "compile",
  },
  {
    fork: [
      {
        id: "spirv",
        label: "SPIR-V",
        sublabel: "llc, wgpu",
        href: "./backends",
        color: "runtime",
      },
      {
        id: "ptx",
        label: "PTX",
        sublabel: "llc, cudarc / CUDA graph",
        href: "./backends",
        color: "runtime",
      },
      {
        id: "rocm",
        label: "HSACO",
        sublabel: "ROCm clang, raw HSA / AQL",
        href: "./backends",
        color: "runtime",
      },
    ],
  },
  {
    id: "executor",
    label: "Backend executors",
    sublabel: "native replay or cached encoding; shared contract planned",
    href: "./execution",
    color: "runtime",
  },
];

function isFork(step: Step): step is Fork {
  return "fork" in step;
}

const COLOR_CLASS: Record<Layer["color"], string> = {
  model: styles.layerModel,
  ir: styles.layerIr,
  compile: styles.layerCompile,
  runtime: styles.layerRuntime,
};

function LayerBox({
  layer,
  active,
  onHover,
}: {
  layer: Layer;
  active: boolean;
  onHover: (id: string | null) => void;
}) {
  return (
    <a
      href={layer.href}
      className={`${styles.layerBox} ${COLOR_CLASS[layer.color]} ${active ? styles.layerBoxActive : ""}`}
      onMouseEnter={() => onHover(layer.id)}
      onMouseLeave={() => onHover(null)}
    >
      <span className={styles.layerLabel}>{layer.label}</span>
      {layer.sublabel && (
        <span className={styles.layerSublabel}>{layer.sublabel}</span>
      )}
    </a>
  );
}

export default function PipelineDiagram(): React.ReactElement {
  const [activeId, setActiveId] = useState<string | null>(null);

  return (
    <div className={styles.pipelineWrap}>
      <div className={styles.pipelineTitle}>
        Full compilation and execution pipeline
      </div>
      <div className={styles.pipeline}>
        {STEPS.map((step, i) => {
          if (isFork(step)) {
            return (
              <React.Fragment key={step.fork[0].id}>
                <div className={styles.arrow} />
                <div className={styles.forkRow}>
                  {step.fork.map((layer) => (
                    <div key={layer.id} className={styles.forkBranch}>
                      <LayerBox
                        layer={layer}
                        active={activeId === layer.id}
                        onHover={setActiveId}
                      />
                    </div>
                  ))}
                </div>
              </React.Fragment>
            );
          }
          return (
            <React.Fragment key={step.id}>
              {i > 0 && <div className={styles.arrow} />}
              <LayerBox
                layer={step}
                active={activeId === step.id}
                onHover={setActiveId}
              />
            </React.Fragment>
          );
        })}
      </div>
      <div className={styles.legend}>
        <span className={`${styles.legendDot} ${styles.layerModel}`} />
        Model
        <span className={`${styles.legendDot} ${styles.layerIr}`} />
        IR &amp; transforms
        <span className={`${styles.legendDot} ${styles.layerCompile}`} />
        Codegen
        <span className={`${styles.legendDot} ${styles.layerRuntime}`} />
        Runtime
      </div>
    </div>
  );
}
