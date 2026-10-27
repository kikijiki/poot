import assert from 'node:assert/strict';
import {test} from 'node:test';
import {DEFAULT_VIEW, parseView, viewToSearch} from './urlstate.ts';

test('a run view survives the round trip through the query string', () => {
  const view = {...DEFAULT_VIEW, view: 'runs', run: 'run-1792660009-2575640', tab: 'results', sub: 'table'} as const;
  assert.deepEqual(parseView(viewToSearch(view)), view);
  assert.equal(viewToSearch(view), '?view=runs&run=run-1792660009-2575640&tab=results&sub=table');
});

test('no query is the overview', () => {
  assert.deepEqual(parseView(''), DEFAULT_VIEW);
  assert.deepEqual(parseView('?'), DEFAULT_VIEW);
  assert.equal(viewToSearch(DEFAULT_VIEW), '?view=overview');
});

test('an unknown view, tab or sub-tab falls back to its default, keeping the rest', () => {
  assert.deepEqual(parseView('?view=runs&run=r1&tab=bogus&sub=nope'), {...DEFAULT_VIEW, view: 'runs', run: 'r1'});
  assert.equal(parseView('?view=bogus').view, 'overview');
});

test('an empty run is no run, and an id with reserved characters round-trips', () => {
  assert.equal(parseView('?view=runs&run=').run, null);
  const view = {...DEFAULT_VIEW, view: 'runs', run: 'a b&c=d'} as const;
  assert.deepEqual(parseView(viewToSearch(view)), view);
});

test('the compare selection is a list in the query and nothing else is kept for other views', () => {
  const view = {...DEFAULT_VIEW, view: 'compare', cmp: ['run-1', 'run-2']} as const;
  assert.equal(viewToSearch(view), '?view=compare&cmp=run-1%2Crun-2');
  assert.deepEqual(parseView(viewToSearch(view)).cmp, ['run-1', 'run-2']);
  assert.equal(viewToSearch({...DEFAULT_VIEW, view: 'pods', run: 'r1', cmp: ['x']}), '?view=pods');
});

test('the image log of the images view is kept', () => {
  const view = {...DEFAULT_VIEW, view: 'images', image: 'image-latest.log'} as const;
  assert.deepEqual(parseView(viewToSearch(view)), view);
});
