// The dashboard's view lives in the URL query (?view=&run=&tab=&sub=&image=&cmp=), so a reload, the back and
// forward buttons and a shared link keep it. Pure functions: parse never throws, an unknown value falls back to
// the default.

export const VIEWS = ['overview', 'runs', 'pods', 'compare', 'images', 'serve'] as const;
export const RUN_TABS = ['summary', 'log', 'results', 'pod'] as const;
export const RESULT_SUBS = ['table', 'charts', 'failures'] as const;

export type ViewId = (typeof VIEWS)[number];
export type RunTab = (typeof RUN_TABS)[number];
export type ResultSub = (typeof RESULT_SUBS)[number];

export type View = {
  view: ViewId;
  // The selected run (Runs view), image build log (Images view).
  run: string | null;
  tab: RunTab;
  sub: ResultSub;
  image: string | null;
  // The runs overlaid by the Compare view.
  cmp: string[];
};

export const DEFAULT_VIEW: View = {
  view: 'overview',
  run: null,
  tab: 'summary',
  sub: 'charts',
  image: null,
  cmp: [],
};

function oneOf<T extends string>(allowed: readonly T[], value: string | null, fallback: T): T {
  return allowed.find((a) => a === value) ?? fallback;
}

// `search` is `location.search` (with or without the leading `?`).
export function parseView(search: string): View {
  const q = new URLSearchParams(search);
  return {
    view: oneOf(VIEWS, q.get('view'), DEFAULT_VIEW.view),
    run: q.get('run') || null,
    tab: oneOf(RUN_TABS, q.get('tab'), DEFAULT_VIEW.tab),
    sub: oneOf(RESULT_SUBS, q.get('sub'), DEFAULT_VIEW.sub),
    image: q.get('image') || null,
    cmp: (q.get('cmp') ?? '').split(',').filter(Boolean),
  };
}

// The query string (with its leading `?`) of a view; what the view does not use is left out.
export function viewToSearch(v: View): string {
  const q = new URLSearchParams();
  q.set('view', v.view);
  if (v.view === 'runs') {
    if (v.run) q.set('run', v.run);
    q.set('tab', v.tab);
    if (v.tab === 'results') q.set('sub', v.sub);
  }
  if (v.view === 'images' && v.image) q.set('image', v.image);
  if (v.view === 'compare' && v.cmp.length > 0) q.set('cmp', v.cmp.join(','));
  return `?${q.toString()}`;
}
