// The page registry: the navigation and the page body of every view are one record here. A view that is not in
// this list does not exist, and urlstate's VIEWS must name exactly these ids (pages.test.tsx checks it). The
// scheduler views of POOT-1358 (job queue, attempts, pool leases) are added as further records, with their id in
// VIEWS; nothing else in the shell changes.

import React from 'react';
import type {AppData} from './data';
import {isActive, isLivePod} from './model';
import Compare from './views/Compare';
import Images from './views/Images';
import Overview from './views/Overview';
import Pods from './views/Pods';
import Runs from './views/Runs';
import Serve from './views/Serve';
import type {View, ViewId} from './urlstate';
import {Async} from './ui';

export type PageCtx = {
  data: AppData;
  view: View;
  now: number;
  // Applies a change to the view; `push` false replaces the history entry instead of adding one.
  go: (patch: Partial<View>, push?: boolean) => void;
};

export type Page = {
  id: ViewId;
  label: string;
  // The count shown next to the label, when there is one.
  count?: (data: AppData) => number | null;
  render: (ctx: PageCtx) => React.ReactNode;
};

const ready = <T,>(l: {kind: string; value?: T}): T | null => (l.kind === 'ready' ? (l.value as T) : null);

export const PAGES: Page[] = [
  {
    id: 'overview',
    label: 'Overview',
    render: ({data, now, go}) => (
      <Overview
        state={data.state}
        overview={data.overview}
        now={now}
        onRetry={data.reload}
        onOpen={(run) => go({view: 'runs', run, tab: 'summary'})}
      />
    ),
  },
  {
    id: 'runs',
    label: 'Runs',
    count: (d) => ready(d.state)?.runs.filter(isActive).length ?? null,
    render: ({data, view, now, go}) => (
      <Async what="runs" data={data.state} onRetry={data.reload}>
        {(state) => (
          <Runs
            runs={state.runs}
            pods={state.pods}
            logfiles={data.logfiles}
            selected={view.run}
            tab={view.tab}
            sub={view.sub}
            now={now}
            onSelect={(run) => go({run: run || null})}
            onTab={(tab) => go({tab})}
            onSub={(sub) => go({sub})}
          />
        )}
      </Async>
    ),
  },
  {
    id: 'pods',
    label: 'Pods',
    count: (d) => ready(d.state)?.pods.filter(isLivePod).length ?? null,
    render: ({data, now, go}) => <Pods state={data.state} now={now} onRetry={data.reload} onOpen={(run) => go({view: 'runs', run, tab: 'pod'})} />,
  },
  {
    id: 'compare',
    label: 'Compare',
    render: ({data, view, now, go}) => <Compare state={data.state} selected={view.cmp} now={now} onSelect={(cmp) => go({cmp}, false)} onRetry={data.reload} />,
  },
  {
    id: 'images',
    label: 'Images',
    count: (d) => d.logfiles.filter((f) => f.kind === 'image').length,
    render: ({data, view, now, go}) => (
      <Images logfiles={data.logfiles} selected={view.image} now={now} onSelect={(image) => go({image})} onTriggered={data.reload} />
    ),
  },
  {id: 'serve', label: 'Serve', render: () => <Serve />},
];
