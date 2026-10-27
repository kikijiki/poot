import React, {useEffect, useMemo, useRef, useState} from 'react';
import {fetchLog} from '../api';
import {usePolled} from '../data';
import {Async, CopyButton, Empty} from '../ui';

const TAILS = [400, 2000, 10000] as const;

// Lines of `text` that contain `query` (case-insensitive); all of them for an empty query.
export function filterLines(text: string, query: string): string[] {
  const lines = text.split('\n');
  const q = query.trim().toLowerCase();
  return q ? lines.filter((l) => l.toLowerCase().includes(q)) : lines;
}

// A line with the matches of `query` wrapped in <mark>.
function Highlighted(props: {line: string; query: string}) {
  const q = props.query.trim();
  if (!q) return <>{props.line}</>;
  const parts = props.line.split(new RegExp(`(${q.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')})`, 'i'));
  return (
    <>
      {parts.map((p, i) => (i % 2 === 1 ? <mark key={i}>{p}</mark> : <React.Fragment key={i}>{p}</React.Fragment>))}
    </>
  );
}

// The live tail of one log file with search, follow and copy.
export default function LogPane(props: {name: string | null; live: boolean; missing: string}) {
  const {name} = props;
  const [tail, setTail] = useState<number>(TAILS[0]);
  const [query, setQuery] = useState('');
  const [follow, setFollow] = useState(true);
  const box = useRef<HTMLPreElement>(null);
  const {data, reload} = usePolled(() => (name ? fetchLog(name, tail) : Promise.resolve('')), [name, tail], 3000, props.live);

  const text = data.kind === 'ready' ? data.value : '';
  const lines = useMemo(() => filterLines(text, query), [text, query]);

  useEffect(() => {
    if (follow && box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines, follow]);

  if (!name) return <Empty title="No log">{props.missing}</Empty>;
  return (
    <div className="logpane">
      <div className="logbar">
        <span className="logname mono">{name}</span>
        <input
          type="search"
          value={query}
          placeholder="Search the log"
          aria-label="Search the log"
          onChange={(e) => setQuery(e.target.value)}
        />
        {query ? (
          <span className="muted" aria-live="polite">
            {lines.length} of {text.split('\n').length} lines
          </span>
        ) : null}
        <label className="inline">
          tail
          <select value={tail} onChange={(e) => setTail(Number(e.target.value))}>
            {TAILS.map((t) => (
              <option key={t} value={t}>
                {t} lines
              </option>
            ))}
          </select>
        </label>
        <label className="inline">
          <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} />
          follow
        </label>
        <CopyButton text={lines.join('\n')} label={query ? 'Copy matches' : 'Copy log'} />
      </div>
      <Async what="the log" data={data} onRetry={reload}>
        {() =>
          text.trim() === '' ? (
            <Empty title="The log is empty">Nothing has been written to it yet.</Empty>
          ) : lines.length === 0 ? (
            <Empty title="No line matches">Nothing in the last {tail} lines contains “{query}”.</Empty>
          ) : (
            <pre className="log" ref={box} tabIndex={0} aria-label={`Log ${name}`}>
              {lines.map((l, i) => (
                <div key={i}>
                  <Highlighted line={l} query={query} />
                </div>
              ))}
            </pre>
          )
        }
      </Async>
    </div>
  );
}
