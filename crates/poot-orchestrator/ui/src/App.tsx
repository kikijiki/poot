import React, {useCallback, useEffect, useMemo, useRef, useState} from 'react';
import {useAppData} from './data';
import {PAGES} from './pages';
import {parseView, viewToSearch} from './urlstate';
import type {View} from './urlstate';
import {useNow} from './ui';

type Theme = 'auto' | 'light' | 'dark';

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() => {
    try {
      const saved = window.localStorage.getItem('poot-theme');
      return saved === 'light' || saved === 'dark' ? saved : 'auto';
    } catch {
      return 'auto';
    }
  });
  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'auto') root.removeAttribute('data-theme');
    else root.setAttribute('data-theme', theme);
    try {
      if (theme === 'auto') window.localStorage.removeItem('poot-theme');
      else window.localStorage.setItem('poot-theme', theme);
    } catch {
      // storage unavailable (private window): the choice lasts for this page only
    }
  }, [theme]);
  const next: Record<Theme, Theme> = {auto: 'light', light: 'dark', dark: 'auto'};
  return [theme, () => setTheme(next[theme])];
}

export default function App() {
  const [view, setView] = useState<View>(() => parseView(window.location.search));
  const data = useAppData();
  const now = useNow();
  const [theme, cycleTheme] = useTheme();
  // The next URL update replaces the history entry (first load, auto-selection) instead of adding one.
  const replaceNext = useRef(true);

  // Keep the URL in step with the view: a user's change is a history entry, so back/forward restore it.
  useEffect(() => {
    const search = viewToSearch(view);
    if (search === window.location.search) return;
    const url = `${window.location.pathname}${search}`;
    if (replaceNext.current) window.history.replaceState(null, '', url);
    else window.history.pushState(null, '', url);
    replaceNext.current = false;
  }, [view]);

  useEffect(() => {
    const onPop = () => {
      replaceNext.current = false;
      setView(parseView(window.location.search));
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, []);

  const go = useCallback((patch: Partial<View>, push = true) => {
    replaceNext.current = !push;
    setView((v) => ({...v, ...patch}));
  }, []);

  // The Runs view with no run selected opens the newest one on a wide screen only through the list; nothing is
  // auto-selected, so a link to the list stays the list.
  const page = PAGES.find((p) => p.id === view.view) ?? PAGES[0];
  const ctx = useMemo(() => ({data, view, now, go}), [data, view, now, go]);
  const live = data.failing === null && data.state.kind === 'ready';

  return (
    <div className="app">
      <a className="skip" href="#main">
        Skip to content
      </a>
      <header className="topbar">
        <h1>
          <span className="brand">poot</span> orchestrator
        </h1>
        <nav aria-label="Primary">
          {PAGES.map((p) => {
            const n = p.count?.(data) ?? null;
            return (
              <button key={p.id} type="button" className={p.id === page.id ? 'active' : ''} aria-current={p.id === page.id ? 'page' : undefined} onClick={() => go({view: p.id})}>
                {p.label}
                {n !== null ? <span className="count">{n}</span> : null}
              </button>
            );
          })}
        </nav>
        <span className={`conn ${live ? 'live' : 'down'}`} role="status">
          <span className="dot" aria-hidden="true" />
          {live ? 'live' : data.state.kind === 'loading' ? 'connecting…' : `disconnected${data.failing ? `: ${data.failing}` : ''}`}
        </span>
        <button type="button" className="btn ghost small" onClick={cycleTheme} aria-label={`Theme: ${theme}. Switch theme`}>
          {theme === 'auto' ? '◐ auto' : theme === 'light' ? '☀ light' : '☾ dark'}
        </button>
      </header>
      <main id="main" tabIndex={-1}>
        <h2 className="sr-only">{page.label}</h2>
        {page.render(ctx)}
      </main>
    </div>
  );
}
