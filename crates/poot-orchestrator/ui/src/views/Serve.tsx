import React, {useEffect, useMemo, useState} from 'react';
import {fetchServeMetrics} from '../api';
import {Empty, ErrorBox} from '../ui';

const SERVE_POLL_MS = 3000;

// Live serve panel (card 051): polls a running poot-serve's JSON /metrics through the orchestrator
// proxy (/api/serve-metrics). Request-outcome counters get tiles; everything else renders as generic
// flattened key/value rows, so new metrics fields need no UI change.

// The four request-outcome counters (poot_requests{,_completed,_cancelled,_errored}_total in
// Prometheus) get their own tiles.
const OUTCOME_KEYS: [key: string, label: string][] = [
  ['requests', 'total'],
  ['requests_completed', 'completed'],
  ['requests_cancelled', 'cancelled'],
  ['requests_errored', 'errored'],
];

// Flatten a JSON object into dot-path "key: value" rows, recursing into nested objects (e.g. the
// {count, avg} histograms) but not arrays.
function flattenMetrics(
  obj: Record<string, unknown>,
  prefix = '',
): [string, string][] {
  const rows: [string, string][] = [];
  for (const [k, v] of Object.entries(obj)) {
    const key = prefix ? `${prefix}.${k}` : k;
    if (v !== null && typeof v === 'object' && !Array.isArray(v)) {
      rows.push(...flattenMetrics(v as Record<string, unknown>, key));
    } else {
      rows.push([key, Array.isArray(v) ? JSON.stringify(v) : String(v)]);
    }
  }
  return rows;
}

// Polls /api/serve-metrics every SERVE_POLL_MS while `polling` is true and `target` is non-empty.
// `error` is set from either a proxy-reported {"error":...} body or a network/parse failure.
function useServeMetrics(target: string, polling: boolean) {
  const [metrics, setMetrics] = useState<Record<string, unknown> | null>(
    null,
  );
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    if (!polling || !target.trim()) {
      return;
    }
    let stop = false;
    const load = async () => {
      try {
        const m = await fetchServeMetrics(target.trim());
        if (stop) return;
        if (typeof m.error === 'string') {
          setMetrics(null);
          setErr(m.error);
        } else {
          setMetrics(m);
          setErr(null);
        }
      } catch (e) {
        if (!stop) {
          setMetrics(null);
          setErr(String(e));
        }
      }
    };
    load();
    const id = setInterval(load, SERVE_POLL_MS);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [target, polling]);

  return {metrics, err};
}


// The live serve page: a target and a poll toggle, then the metrics of whatever answers there.
export default function Serve() {
  const [target, setTarget] = useState('127.0.0.1:8080');
  const [polling, setPolling] = useState(true);
  const {metrics, err} = useServeMetrics(target, polling);

  const outcomeTiles = useMemo(() => (metrics ? OUTCOME_KEYS.filter(([k]) => k in metrics) : []), [metrics]);
  const rest = useMemo(() => {
    if (!metrics) return [];
    const skip = new Set(OUTCOME_KEYS.map(([k]) => k));
    return flattenMetrics(Object.fromEntries(Object.entries(metrics).filter(([k]) => !skip.has(k))));
  }, [metrics]);

  return (
    <div className="serve">
      <form className="inlineform" onSubmit={(e) => e.preventDefault()}>
        <label>
          Target (host:port)
          <input type="text" value={target} onChange={(e) => setTarget(e.target.value)} placeholder="127.0.0.1:8080" />
        </label>
        <label className="inline">
          <input type="checkbox" checked={polling} onChange={(e) => setPolling(e.target.checked)} />
          poll every 3 s
        </label>
      </form>
      {!target.trim() ? (
        <Empty title="No target">Enter the host:port of a running poot-serve.</Empty>
      ) : err ? (
        <ErrorBox title="Serve unreachable" message={err} />
      ) : !metrics ? (
        <Empty title={polling ? 'Waiting for metrics…' : 'Polling is paused'}>{polling ? 'The first answer arrives within a few seconds.' : 'Enable polling to fetch.'}</Empty>
      ) : (
        <>
          <div className="tiles">
            {outcomeTiles.map(([k, label]) => (
              <div className="kpi" key={k}>
                <span className="kpi-label">{label}</span>
                <strong className="kpi-value">{String(metrics[k])}</strong>
              </div>
            ))}
          </div>
          <div className="kv-grid">
            {rest.map(([k, v]) => (
              <div className="kv-row" key={k}>
                <span className="k mono">{k}</span>
                <span className="v mono">{v}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
