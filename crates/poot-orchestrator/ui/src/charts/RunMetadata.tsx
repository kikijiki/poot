import React, {useMemo} from 'react';
import type {Run} from './types';
import {frameworkColor, frameworkLabel, shortDate} from './types';
import styles from './styles.module.css';

export default function RunMetadata(props: {run: Run; model: string}) {
  const {run, model} = props;
  const env = run.env;

  // The provenance is on the rows: each row names the build, backend and device that produced it.
  const provenance = useMemo(() => {
    const distinct = (values: (string | null | undefined)[]) =>
      [...new Set(values.filter((v): v is string => !!v))].join(', ') || null;
    return {
      builds: distinct(run.results.map((r) => r.build_sha?.replace(/^([0-9a-f]{12})[0-9a-f]+/, '$1'))),
      backends: distinct(run.results.map((r) => r.backend)),
      devices: distinct(run.results.map((r) => r.device)),
    };
  }, [run]);

  const meta: {k: string; v: string | null}[] = [
    {k: 'GPU', v: env.gpu_name},
    {k: 'GPU driver', v: env.gpu_driver},
    {k: 'GPU memory', v: env.gpu_memory_total},
    {k: 'CUDA', v: env.cuda_version},
    {k: 'poot build', v: provenance.builds},
    {k: 'backend', v: provenance.backends},
    {k: 'device', v: provenance.devices},
    {k: 'captured', v: shortDate(env.captured_at_utc)},
    {k: 'CPU', v: env.cpu},
    {k: 'host kernel', v: env.host_kernel},
    {k: 'harness', v: env.harness_version},
  ].filter((m) => m.v);

  // Collect caveats and unsupported/error rows for this model.
  const notes = useMemo(() => {
    const out: {framework: string; text: string; kind: 'caveat' | 'unsupported' | 'error'}[] = [];
    const seen = new Set<string>();
    for (const r of run.results) {
      if (model && r.model !== model) continue;
      if (r.status !== 'ok' && r.reason) {
        const key = `${r.framework}|${r.scenario}|${r.reason}`;
        if (!seen.has(key)) {
          seen.add(key);
          out.push({
            framework: r.framework,
            text: `${r.scenario}: ${r.reason}`,
            kind: r.status,
          });
        }
      }
      for (const c of r.caveats ?? []) {
        const key = `${r.framework}|${c}`;
        if (!seen.has(key)) {
          seen.add(key);
          out.push({framework: r.framework, text: c, kind: 'caveat'});
        }
      }
    }
    return out;
  }, [run, model]);

  return (
    <div className={styles.chartCard}>
      <div className={styles.chartTitle}>Run metadata</div>
      <div className={styles.metaGrid}>
        {meta.map((m) => (
          <div key={m.k} className={styles.metaItem}>
            <span className={styles.metaKey}>{m.k}</span>
            <span className={styles.metaVal}>{m.v}</span>
          </div>
        ))}
      </div>
      <div className={styles.chartTitle} style={{fontSize: '0.95rem', marginTop: '0.5rem'}}>
        Caveats ({notes.length})
      </div>
      <div className={styles.chartSub} style={{marginTop: 0}}>
        This suite measures actual performance, so the per-cell notes matter.
      </div>
      {notes.length === 0 ? (
        <p className={styles.chartSub}>No caveats recorded for {model}.</p>
      ) : (
        <ul className={styles.caveatList}>
          {notes.map((n, i) => (
            <li key={i}>
              <span
                className={styles.swatch}
                style={{background: frameworkColor(n.framework)}}
              />
              <span className={styles.caveatFw}>{frameworkLabel(n.framework)}</span>
              {n.kind !== 'caveat' ? <em> ({n.kind})</em> : null}: {n.text}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
