// Polling hooks: the app state every few seconds, a result set while its run is live.

import {useCallback, useEffect, useRef, useState} from 'react';
import {fetchLogfiles, fetchOverview, fetchResults, fetchState} from './api';
import type {LogFile, Overview, ResultsReply, State} from './api';
import type {Loadable} from './ui';

export const POLL_MS = 3000;

// Fetches with `load` now and every `everyMs` while `enabled`; keeps the last value across a failed poll but
// reports the failure, so a stale page says it is stale instead of looking live.
export function usePolled<T>(load: () => Promise<T>, deps: unknown[], everyMs = POLL_MS, enabled = true) {
  const [data, setData] = useState<Loadable<T>>({kind: 'loading'});
  const [failing, setFailing] = useState<string | null>(null);
  const loadRef = useRef(load);
  loadRef.current = load;
  const [tick, setTick] = useState(0);
  const reload = useCallback(() => setTick((t) => t + 1), []);

  useEffect(() => {
    let stop = false;
    setData({kind: 'loading'});
    setFailing(null);
    const run = async () => {
      try {
        const value = await loadRef.current();
        if (stop) return;
        setData({kind: 'ready', value});
        setFailing(null);
      } catch (e) {
        if (stop) return;
        const message = e instanceof Error ? e.message : String(e);
        setFailing(message);
        setData((prev) => (prev.kind === 'ready' ? prev : {kind: 'error', message}));
      }
    };
    run();
    if (!enabled) return () => { stop = true; };
    const id = setInterval(run, everyMs);
    return () => {
      stop = true;
      clearInterval(id);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, everyMs, enabled, tick]);

  return {data, failing, reload};
}

export type AppData = {
  state: Loadable<State>;
  overview: Loadable<Overview>;
  logfiles: LogFile[];
  // The message of the latest failed poll, or null while the server answers.
  failing: string | null;
  reload: () => void;
};

export function useAppData(): AppData {
  const state = usePolled(fetchState, []);
  const overview = usePolled(fetchOverview, []);
  const logs = usePolled(fetchLogfiles, []);
  return {
    state: state.data,
    overview: overview.data,
    logfiles: logs.data.kind === 'ready' ? logs.data.value : [],
    failing: state.failing,
    reload: () => {
      state.reload();
      overview.reload();
      logs.reload();
    },
  };
}

// A run's result sets, re-read every poll while `live`.
export function useResults(runId: string | null, live: boolean) {
  return usePolled<ResultsReply>(() => fetchResults(runId ?? ''), [runId], POLL_MS, runId !== null && live);
}
