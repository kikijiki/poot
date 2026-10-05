import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {
  fetchState,
  fetchLogfiles,
  fetchLog,
  fetchResults,
  fetchServeMetrics,
  postBuild,
  snapshotsToRun,
} from './api';
import type {RunView, LogFile, RunStatus, LogState, Snapshot, Pod} from './api';
import type {Run} from './charts/types';
import {modelsInRun} from './charts/data';
import DegradationCurve from './charts/DegradationCurve';
import SinglePointBars from './charts/SinglePointBars';
import ResourceUtil from './charts/ResourceUtil';
import RunMetadata from './charts/RunMetadata';

const POLL_MS = 3000;
const SERVE_POLL_MS = 3000;

type Tab = 'runs' | 'pods' | 'images' | 'serve';
type SubTab = 'log' | 'results';

// Status -> badge color. Active phases also pulse.
const STATUS_COLOR: Record<RunStatus, string> = {
  new: '#6b7688',
  provisioning: '#62b0ef',
  setup: '#62b0ef',
  sweeping: '#62b0ef',
  gathering: '#62b0ef',
  done: '#5bc492',
  failed: '#ef6e78',
  aborted: '#ef6e78',
};

const ACTIVE_STATUS: Record<string, boolean> = {
  provisioning: true,
  setup: true,
  sweeping: true,
  gathering: true,
};

const LOG_STATE_COLOR: Record<LogState, string> = {
  running: '#62b0ef',
  done: '#5bc492',
  failed: '#ef6e78',
};

function agoStr(iso: string): string {
  const t = Date.parse(iso);
  if (isNaN(t)) return iso;
  const s = Math.max(0, Math.round((Date.now() - t) / 1000));
  if (s < 60) return `${s}s ago`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ago`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ago`;
  return `${Math.floor(h / 24)}d ago`;
}

function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
  return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function fmtMtime(epochSecs: number): string {
  return agoStr(new Date(epochSecs * 1000).toISOString());
}

function Badge(props: {text: string; color: string; pulse?: boolean}) {
  return (
    <span
      className={`badge${props.pulse ? ' active-phase' : ''}`}
      style={{background: props.color, color: '#0e1217'}}>
      {props.text}
    </span>
  );
}

// One model's three-segment setup -> sweep -> gather track.
function ModelTrack(props: {
  model: string;
  setup_done: boolean;
  sweep_done: boolean;
  gathered: boolean;
  pod_run_id: string | null;
  runActive: boolean;
}) {
  // A segment is active (pulsing) if it is the first not-done one and the run is live.
  const states: ('done' | 'active' | 'pending')[] = [];
  const flags = [props.setup_done, props.sweep_done, props.gathered];
  let activeAssigned = false;
  for (const f of flags) {
    if (f) {
      states.push('done');
    } else if (!activeAssigned && props.runActive) {
      states.push('active');
      activeAssigned = true;
    } else {
      states.push('pending');
    }
  }
  return (
    <div className="model-track">
      <div className="model-name">{props.model}</div>
      <div className="segs">
        {states.map((s, i) => (
          <div key={i} className={`seg ${s}`} />
        ))}
      </div>
      <div className="seg-labels">
        <span>setup</span>
        <span>sweep</span>
        <span>gather</span>
      </div>
      {props.pod_run_id ? (
        <code className="snap-id">{props.pod_run_id}</code>
      ) : null}
    </div>
  );
}

const POD_STATUS_COLOR: Record<string, string> = {
  provisioning: '#caa23a',
  idle: '#3a86c8',
  busy: '#3fb37f',
  terminated: '#6b7688',
};

// A pod (worker) card (spec 030). `current_run` is the run using the pod, if any.
function PodCard({pod}: {pod: Pod}) {
  const facts: [string, string | null][] = [
    ['gpu', pod.gpu_type],
    ['cloud', pod.cloud],
    ['run', pod.current_run],
    ['ssh', pod.ssh_host ? `${pod.ssh_host}:${pod.ssh_port ?? ''}` : null],
    ['image', pod.image],
  ];
  return (
    <div className="card">
      <div className="card-top">
        <span className="card-id">{pod.id}</span>
        <Badge
          text={pod.status}
          color={POD_STATUS_COLOR[pod.status] ?? '#6b7688'}
          pulse={pod.status === 'provisioning' || pod.status === 'busy'}
        />
      </div>
      <div className="card-sub">
        <span className="ago">updated {agoStr(pod.updated_at)}</span>
      </div>
      <div className="facts">
        {facts.map(([k, v]) =>
          v ? (
            <div className="fact" key={k}>
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </div>
          ) : null,
        )}
      </div>
    </div>
  );
}

function RunCard(props: {
  run: RunView;
  selected: boolean;
  onSelect: () => void;
}) {
  const {run} = props;
  const active = !!ACTIVE_STATUS[run.status];
  const facts: [string, string | null][] = [
    ['pod', run.pod_id],
    ['gpu', run.gpu_types],
    [
      'ssh',
      run.ssh_host ? `${run.ssh_host}:${run.ssh_port ?? ''}` : null,
    ],
    ['image', run.image],
  ];
  return (
    <div
      className={`card${props.selected ? ' selected' : ''}`}
      onClick={props.onSelect}>
      <div className="card-top">
        <span className="card-id">{run.id}</span>
        <Badge
          text={run.status}
          color={STATUS_COLOR[run.status] ?? '#6b7688'}
          pulse={active}
        />
      </div>
      <div className="card-sub">
        {run.scenario} @ <span className="ref">{run.git_ref}</span>
        <span className="ago"> updated {agoStr(run.updated_at)}</span>
      </div>
      <div className="facts">
        {facts.map(([k, v]) =>
          v ? (
            <div className="fact" key={k}>
              <span className="k">{k}</span>
              <span className="v">{v}</span>
            </div>
          ) : null,
        )}
      </div>
      {run.models.map((m) => (
        <ModelTrack key={m.model} {...m} runActive={active} />
      ))}
    </div>
  );
}

function BuildForm(props: {onTriggered: () => void}) {
  const [dockerfile, setDockerfile] = useState('docker/Dockerfile');
  const [tag, setTag] = useState('latest');
  const [push, setPush] = useState(true);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{text: string; ok: boolean} | null>(null);

  const submit = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await postBuild(dockerfile, tag, push);
      if (r.error) {
        setMsg({text: r.error, ok: false});
      } else {
        setMsg({text: `started: ${r.log ?? 'build'}`, ok: true});
        props.onTriggered();
      }
    } catch (e) {
      setMsg({text: String(e), ok: false});
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="build-form">
      <h3>Trigger image build</h3>
      <label>
        dockerfile
        <input
          type="text"
          value={dockerfile}
          onChange={(e) => setDockerfile(e.target.value)}
        />
      </label>
      <label>
        tag
        <input
          type="text"
          value={tag}
          onChange={(e) => setTag(e.target.value)}
        />
      </label>
      <div className="row-check">
        <input
          type="checkbox"
          id="push"
          checked={push}
          onChange={(e) => setPush(e.target.checked)}
        />
        <label htmlFor="push" style={{margin: 0}}>
          push
        </label>
      </div>
      <button className="btn" disabled={busy} onClick={submit}>
        {busy ? 'starting...' : 'build'}
      </button>
      {msg ? (
        <div className={`build-msg ${msg.ok ? 'ok' : 'err'}`}>{msg.text}</div>
      ) : null}
    </div>
  );
}

function ImageCard(props: {
  file: LogFile;
  selected: boolean;
  onSelect: () => void;
}) {
  const {file} = props;
  return (
    <div
      className={`card${props.selected ? ' selected' : ''}`}
      onClick={props.onSelect}>
      <div className="card-top">
        <span className="card-id">{file.name}</span>
        <Badge
          text={file.state}
          color={LOG_STATE_COLOR[file.state] ?? '#6b7688'}
          pulse={file.state === 'running'}
        />
      </div>
      <div className="card-sub">
        {fmtBytes(file.size)} <span className="ago">{fmtMtime(file.mtime)}</span>
      </div>
    </div>
  );
}

function LogView(props: {name: string | null}) {
  const {name} = props;
  const [text, setText] = useState('');
  const [follow, setFollow] = useState(true);
  const boxRef = useRef<HTMLPreElement>(null);

  useEffect(() => {
    let stop = false;
    const load = async () => {
      if (!name) {
        setText('');
        return;
      }
      try {
        const t = await fetchLog(name, 800);
        if (!stop) setText(t);
      } catch (e) {
        if (!stop) setText(String(e));
      }
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [name]);

  useEffect(() => {
    if (follow && boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  }, [text, follow]);

  const toBottom = () => {
    if (boxRef.current) {
      boxRef.current.scrollTop = boxRef.current.scrollHeight;
    }
  };

  return (
    <>
      <div className="log-bar">
        <span className="name">{name ?? 'no selection'}</span>
        <span className="spacer" />
        <label className="log-toggle">
          <input
            type="checkbox"
            checked={follow}
            onChange={(e) => setFollow(e.target.checked)}
          />
          follow
        </label>
        <button className="btn ghost" onClick={toBottom}>
          to bottom
        </button>
      </div>
      <pre className="log" ref={boxRef}>
        {text || (name ? 'waiting for log...' : 'select a card to view its log')}
      </pre>
    </>
  );
}

function ResultsView(props: {runId: string | null; isImage: boolean}) {
  const {runId, isImage} = props;
  const [run, setRun] = useState<Run | null>(null);
  const [snaps, setSnaps] = useState<Snapshot[]>([]);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let stop = false;
    if (!runId || isImage) {
      setRun(null);
      setSnaps([]);
      return;
    }
    const load = async () => {
      try {
        const s = await fetchResults(runId);
        if (stop) return;
        setSnaps(s);
        setRun(snapshotsToRun(runId, s));
        setErr(null);
      } catch (e) {
        if (!stop) setErr(String(e));
      }
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      stop = true;
      clearInterval(id);
    };
  }, [runId, isImage]);

  // Models per snapshot dir, grouped per snapshot in the header.
  const blocks = useMemo(() => {
    if (!run) return [];
    return snaps.map((s) => {
      const mdls = modelsInRun({...run, results: s.rows ?? []});
      return {dir: s.dir, models: mdls};
    });
  }, [run, snaps]);

  if (isImage) {
    return <div className="placeholder">results are per run</div>;
  }
  if (!runId) {
    return <div className="placeholder">select a run to view results</div>;
  }
  if (err) {
    return <div className="placeholder">error loading results: {err}</div>;
  }
  if (!run || run.results.length === 0) {
    return (
      <div className="placeholder">
        no gathered results yet for this run
      </div>
    );
  }

  return (
    <div className="results">
      {blocks.map((b) =>
        b.models.map((model) => (
          <div className="snap-block" key={`${b.dir}:${model}`}>
            <div className="snap-head">
              <span className="model">{model}</span>
              <span className="dir">{b.dir}</span>
            </div>
            <SinglePointBars run={run} model={model} />
            <DegradationCurve run={run} model={model} />
            <ResourceUtil run={run} model={model} />
            <RunMetadata run={run} model={model} />
          </div>
        )),
      )}
    </div>
  );
}

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

// Left-column controls: target input + poll toggle + a one-line connection status.
function ServeControls(props: {
  target: string;
  onTarget: (v: string) => void;
  polling: boolean;
  onPolling: (v: boolean) => void;
  metrics: Record<string, unknown> | null;
  error: string | null;
}) {
  const {target, onTarget, polling, onPolling, metrics, error} = props;
  const status = error
    ? `unreachable: ${error}`
    : metrics
      ? 'connected'
      : polling
        ? 'waiting for metrics...'
        : 'polling paused';
  return (
    <div className="build-form">
      <h3>Live serve</h3>
      <label>
        target (host:port)
        <input
          type="text"
          value={target}
          onChange={(e) => onTarget(e.target.value)}
          placeholder="127.0.0.1:8080"
        />
      </label>
      <div className="row-check">
        <input
          type="checkbox"
          id="serve-poll"
          checked={polling}
          onChange={(e) => onPolling(e.target.checked)}
        />
        <label htmlFor="serve-poll" style={{margin: 0}}>
          poll every 3s
        </label>
      </div>
      <div className={`build-msg ${error ? 'err' : metrics ? 'ok' : ''}`}>
        {status}
      </div>
    </div>
  );
}

// Right-pane metrics view: outcome tiles, then a flattened key/value grid for the rest.
function ServeMetricsPanel(props: {
  metrics: Record<string, unknown> | null;
  error: string | null;
  polling: boolean;
}) {
  const {metrics, error, polling} = props;

  const outcomeTiles = useMemo(
    () => (metrics ? OUTCOME_KEYS.filter(([k]) => k in metrics) : []),
    [metrics],
  );
  const rest = useMemo(() => {
    if (!metrics) return [];
    const skip = new Set(OUTCOME_KEYS.map(([k]) => k));
    const filtered: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(metrics)) {
      if (!skip.has(k)) filtered[k] = v;
    }
    return flattenMetrics(filtered);
  }, [metrics]);

  if (error) {
    return <div className="placeholder">serve unreachable: {error}</div>;
  }
  if (!metrics) {
    return (
      <div className="placeholder">
        {polling
          ? 'waiting for metrics...'
          : 'polling paused - enable poll to fetch'}
      </div>
    );
  }
  return (
    <div className="pane serve-metrics">
      <div className="outcome-tiles">
        {outcomeTiles.map(([k, label]) => (
          <div className="outcome-tile" key={k}>
            <span className="v">{String(metrics[k])}</span>
            <span className="k">{label}</span>
          </div>
        ))}
      </div>
      <div className="kv-grid">
        {rest.map(([k, v]) => (
          <div className="kv-row" key={k}>
            <span className="k">{k}</span>
            <span className="v">{v}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>('runs');
  const [subTab, setSubTab] = useState<SubTab>('log');
  const [runs, setRuns] = useState<RunView[]>([]);
  const [pods, setPods] = useState<Pod[]>([]);
  const [logfiles, setLogfiles] = useState<LogFile[]>([]);
  const [live, setLive] = useState(false);

  // Selection remembered per tab.
  const [selRun, setSelRun] = useState<string | null>(null);
  const [selImage, setSelImage] = useState<string | null>(null);

  // Live serve panel state (card 051).
  const [serveTarget, setServeTarget] = useState('127.0.0.1:8080');
  const [servePolling, setServePolling] = useState(true);
  const {metrics: serveMetrics, err: serveErr} = useServeMetrics(
    serveTarget,
    servePolling,
  );

  const images = useMemo(
    () => logfiles.filter((f) => f.kind === 'image'),
    [logfiles],
  );

  // The log file for the selected run (run logs are named after the run id).
  const runLogName = useMemo(() => {
    if (!selRun) return null;
    const exact = logfiles.find(
      (f) => f.kind === 'run' && (f.name === `${selRun}.log` || f.name === selRun),
    );
    if (exact) return exact.name;
    const pref = logfiles.find(
      (f) => f.kind === 'run' && f.name.startsWith(selRun),
    );
    return pref ? pref.name : null;
  }, [selRun, logfiles]);

  const poll = useCallback(async () => {
    try {
      const [s, lf] = await Promise.all([fetchState(), fetchLogfiles()]);
      setRuns(s.runs);
      setPods(s.pods);
      setLogfiles(lf);
      setLive(true);
    } catch {
      setLive(false);
    }
  }, []);

  useEffect(() => {
    poll();
    const id = setInterval(poll, POLL_MS);
    return () => clearInterval(id);
  }, [poll]);

  // Auto-select the newest of the active tab if nothing is selected.
  useEffect(() => {
    if (tab === 'runs' && !selRun && runs.length > 0) {
      setSelRun(runs[0].id);
    }
  }, [tab, runs, selRun]);

  useEffect(() => {
    if (tab === 'images' && !selImage && images.length > 0) {
      setSelImage(images[0].name);
    }
  }, [tab, images, selImage]);

  const isImageTab = tab === 'images';
  const selectedLogName = isImageTab ? selImage : runLogName;

  return (
    <div className="app">
      <header className="header">
        <h1>
          <span className="poot-dot">poot</span> bench orchestrator
        </h1>
        <span className="conn">
          <span className={`dot${live ? ' live' : ''}`} />
          {live ? 'live' : 'disconnected'}
        </span>
      </header>

      <div className="body">
        <div className="left">
          <div className="tabs">
            <button
              className={tab === 'runs' ? 'active' : ''}
              onClick={() => setTab('runs')}>
              Runs<span className="tab-count">{runs.length}</span>
            </button>
            <button
              className={tab === 'pods' ? 'active' : ''}
              onClick={() => setTab('pods')}>
              Pods<span className="tab-count">{pods.filter((p) => p.status !== 'terminated').length}</span>
            </button>
            <button
              className={tab === 'images' ? 'active' : ''}
              onClick={() => setTab('images')}>
              Image builds<span className="tab-count">{images.length}</span>
            </button>
            <button
              className={tab === 'serve' ? 'active' : ''}
              onClick={() => setTab('serve')}>
              Live serve
            </button>
          </div>
          <div className="list">
            {tab === 'serve' ? (
              <ServeControls
                target={serveTarget}
                onTarget={setServeTarget}
                polling={servePolling}
                onPolling={setServePolling}
                metrics={serveMetrics}
                error={serveErr}
              />
            ) : tab === 'runs' ? (
              runs.length === 0 ? (
                <div className="placeholder">no runs yet</div>
              ) : (
                runs.map((r) => (
                  <RunCard
                    key={r.id}
                    run={r}
                    selected={selRun === r.id}
                    onSelect={() => setSelRun(r.id)}
                  />
                ))
              )
            ) : tab === 'pods' ? (
              pods.length === 0 ? (
                <div className="placeholder">no pods yet</div>
              ) : (
                pods.map((p) => <PodCard key={p.id} pod={p} />)
              )
            ) : (
              <>
                <BuildForm onTriggered={poll} />
                {images.length === 0 ? (
                  <div className="placeholder">no image builds yet</div>
                ) : (
                  images.map((f) => (
                    <ImageCard
                      key={f.name}
                      file={f}
                      selected={selImage === f.name}
                      onSelect={() => setSelImage(f.name)}
                    />
                  ))
                )}
              </>
            )}
          </div>
        </div>

        <div className="right">
          {tab === 'serve' ? (
            <ServeMetricsPanel
              metrics={serveMetrics}
              error={serveErr}
              polling={servePolling}
            />
          ) : (
            <>
              <div className="sub-tabs">
                <button
                  className={subTab === 'log' ? 'active' : ''}
                  onClick={() => setSubTab('log')}>
                  Log
                </button>
                <button
                  className={subTab === 'results' ? 'active' : ''}
                  onClick={() => setSubTab('results')}>
                  Results
                </button>
              </div>
              {subTab === 'log' ? (
                <LogView name={selectedLogName} />
              ) : (
                <div className="pane">
                  <ResultsView runId={selRun} isImage={isImageTab} />
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
