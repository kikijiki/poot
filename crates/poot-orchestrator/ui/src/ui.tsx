// Small shared components: every view builds on these, so loading, empty and error read the same everywhere.

import React, {useEffect, useState} from 'react';
import type {PillSpec} from './model';

export function Pill(props: {spec: PillSpec; title?: string}) {
  const {label, tone} = props.spec;
  return (
    <span className={`pill tone-${tone}`} title={props.title}>
      {tone === 'run' ? <span className="pulse" aria-hidden="true" /> : null}
      {label}
    </span>
  );
}

export function Kpi(props: {label: string; value: React.ReactNode; note?: React.ReactNode; tone?: 'fail' | 'warn'}) {
  return (
    <div className="kpi">
      <span className="kpi-label">{props.label}</span>
      <strong className={`kpi-value${props.tone ? ` tone-${props.tone}` : ''}`}>{props.value}</strong>
      {props.note ? <span className="kpi-note">{props.note}</span> : null}
    </div>
  );
}

// What a view that loads something shows: the fetch's own state, then the content when there is some.
export type Loadable<T> = {kind: 'loading'} | {kind: 'error'; message: string} | {kind: 'ready'; value: T};

export function Loading(props: {what: string}) {
  return (
    <div className="state state-loading" role="status">
      <span className="spinner" aria-hidden="true" />
      Loading {props.what}…
    </div>
  );
}

export function ErrorBox(props: {title: string; message: string; onRetry?: () => void}) {
  return (
    <div className="state state-error" role="alert">
      <strong>{props.title}</strong>
      <span>{props.message}</span>
      {props.onRetry ? (
        <button type="button" className="btn" onClick={props.onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function Empty(props: {title: string; children?: React.ReactNode}) {
  return (
    <div className="state state-empty">
      <strong>{props.title}</strong>
      {props.children ? <span>{props.children}</span> : null}
    </div>
  );
}

// Renders the loading / error state of a `Loadable`, or `children(value)` once it is ready (which decides its own
// empty state).
export function Async<T>(props: {
  what: string;
  data: Loadable<T>;
  onRetry?: () => void;
  children: (value: T) => React.ReactNode;
}) {
  const {data} = props;
  if (data.kind === 'loading') return <Loading what={props.what} />;
  if (data.kind === 'error') return <ErrorBox title={`Could not load ${props.what}`} message={data.message} onRetry={props.onRetry} />;
  return <>{props.children(data.value)}</>;
}

export function CopyButton(props: {text: string; label?: string}) {
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (!done) return;
    const id = setTimeout(() => setDone(false), 1500);
    return () => clearTimeout(id);
  }, [done]);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(props.text);
      setDone(true);
    } catch {
      setDone(false);
    }
  };
  return (
    <button type="button" className="btn ghost small" onClick={copy} aria-label={`Copy ${props.label ?? 'text'}`}>
      {done ? 'Copied' : (props.label ?? 'Copy')}
    </button>
  );
}

// The wall clock in ms, re-read every `everyMs`, so "5m ago" and uptimes advance between polls.
export function useNow(everyMs = 5000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(id);
  }, [everyMs]);
  return now;
}

// Rows in the key/value form of a facts grid; a missing value is left out rather than shown empty.
export function Facts(props: {items: [string, React.ReactNode | null | undefined][]}) {
  return (
    <dl className="facts">
      {props.items
        .filter(([, v]) => v !== null && v !== undefined && v !== '')
        .map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
    </dl>
  );
}

export function ProgressBar(props: {done: number; total: number; label: string}) {
  const pct = props.total > 0 ? Math.min(100, (props.done / props.total) * 100) : 0;
  return (
    <div className="progress-cell">
      <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={props.total} aria-valuenow={props.done} aria-label={props.label}>
        <i style={{width: `${pct}%`}} />
      </div>
      <span>{props.label}</span>
    </div>
  );
}
