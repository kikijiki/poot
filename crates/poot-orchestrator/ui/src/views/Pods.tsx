import React from 'react';
import type {RunView, State} from '../api';
import {Async, Empty} from '../ui';
import type {Loadable} from '../ui';
import {PodTable} from './Overview';

export default function Pods(props: {state: Loadable<State>; now: number; onOpen: (runId: string) => void; onRetry: () => void}) {
  return (
    <Async what="pods" data={props.state} onRetry={props.onRetry}>
      {(state) => {
        const live = state.pods.filter((p) => p.status !== 'terminated');
        const ended = state.pods.filter((p) => p.status === 'terminated');
        const runs: RunView[] = state.runs;
        return (
          <div className="pods">
            <p className="muted">
              The state is the DB’s; the last column is what RunPod’s own list says, checked every 30 s. Cost is estimated: pod price × lifetime.
            </p>
            <h2>Live ({live.length})</h2>
            {live.length === 0 ? <Empty title="No live pods">Nothing is rented right now.</Empty> : <PodTable pods={live} runs={runs} now={props.now} onOpen={props.onOpen} />}
            <details className="ended">
              <summary>Terminated ({ended.length})</summary>
              <PodTable pods={ended} runs={runs} now={props.now} onOpen={props.onOpen} />
            </details>
          </div>
        );
      }}
    </Async>
  );
}
