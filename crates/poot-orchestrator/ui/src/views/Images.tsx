import React, {useState} from 'react';
import {postBuild} from '../api';
import type {LogFile} from '../api';
import {formatAgo, formatBytes} from '../model';
import {Empty, Pill} from '../ui';
import type {PillSpec} from '../model';
import LogPane from './LogPane';

const STATE_PILL: Record<LogFile['state'], PillSpec> = {
  running: {label: 'running', tone: 'run'},
  done: {label: 'done', tone: 'ok'},
  failed: {label: 'failed', tone: 'fail'},
};

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
    <form
      className="inlineform"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}>
      <label>
        Dockerfile
        <input type="text" value={dockerfile} onChange={(e) => setDockerfile(e.target.value)} />
      </label>
      <label>
        Tag
        <input type="text" value={tag} onChange={(e) => setTag(e.target.value)} />
      </label>
      <label className="inline">
        <input type="checkbox" checked={push} onChange={(e) => setPush(e.target.checked)} />
        push
      </label>
      <button type="submit" className="btn primary" disabled={busy}>
        {busy ? 'Starting…' : 'Build image'}
      </button>
      {msg ? <span className={`build-msg ${msg.ok ? 'tone-ok' : 'tone-fail'}`} role="status">{msg.text}</span> : null}
    </form>
  );
}

export default function Images(props: {logfiles: LogFile[]; selected: string | null; now: number; onSelect: (name: string) => void; onTriggered: () => void}) {
  const images = props.logfiles.filter((f) => f.kind === 'image');
  const selected = images.find((f) => f.name === props.selected) ?? images[0] ?? null;
  return (
    <div className="images">
      <BuildForm onTriggered={props.onTriggered} />
      {images.length === 0 ? (
        <Empty title="No image builds yet">Start one above; its log appears here while it runs.</Empty>
      ) : (
        <div className="split has-detail">
          <ul className="runlist builds">
            {images.map((f) => (
              <li key={f.name}>
                <button type="button" className={`runrow${selected?.name === f.name ? ' selected' : ''}`} aria-current={selected?.name === f.name} onClick={() => props.onSelect(f.name)}>
                  <span className="runrow-top">
                    <span className="runrow-title mono">{f.name}</span>
                    <Pill spec={STATE_PILL[f.state]} />
                  </span>
                  <span className="runrow-sub muted">
                    {formatBytes(f.size)} · {formatAgo(new Date(f.mtime * 1000).toISOString(), props.now)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
          <div className="detailpane">
            <LogPane name={selected?.name ?? null} live={selected?.state === 'running'} missing="Select a build." />
          </div>
        </div>
      )}
    </div>
  );
}
