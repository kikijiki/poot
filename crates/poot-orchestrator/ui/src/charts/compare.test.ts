import assert from 'node:assert/strict';
import {test} from 'node:test';
import type {Snapshot} from '../api';
import {curveChoices, curveOf, median} from './compare.ts';

const pt = (isl: number, tok: number) => ({isl, osl: 64, decode_tok_s: tok, ttft_ms: {p50: 1, p90: 1, p99: 1}, tpot_ms: {p50: 1000 / tok, p90: 1, p99: 1}, itl_ms: {p50: 1, p90: 1, p99: 1}});
const row = (model: string, tok: number[], framework = 'poot', status = 'ok') => ({
  framework, model, precision: 'f16', scenario: 'decode-curve', status, curve: tok.map((t, i) => pt(128 * 2 ** i, t)),
});
const set = (rows: unknown[], failed = false) => ({dir: 'd', env: null, rows, failed}) as unknown as Snapshot;

test('the median of the repeats at each context length, ascending', () => {
  const sets = [set([row('m', [100, 80])]), set([row('m', [120, 60])]), set([row('m', [110, 70])])];
  assert.deepEqual(curveOf(sets, 'm', 'poot', 'decode_tok_s'), [[128, 110], [256, 70]]);
});

test('other models, frameworks, failed sets and errored rows are not in the curve', () => {
  const sets = [set([row('other', [1]), row('m', [2], 'vllm'), row('m', [3], 'poot', 'error')]), set([row('m', [4])], true)];
  assert.deepEqual(curveOf(sets, 'm', 'poot', 'decode_tok_s'), []);
});

test('a latency metric reads the p50 of its percentile block', () => {
  assert.deepEqual(curveOf([set([row('m', [100])])], 'm', 'poot', 'tpot_ms'), [[128, 10]]);
});

test('the pickers list what some run measured', () => {
  const sets = [[set([row('b', [1]), row('a', [1], 'vllm')])], [set([row('c', [1], 'poot', 'error')])]];
  assert.deepEqual(curveChoices(sets), {models: ['a', 'b'], frameworks: ['poot', 'vllm']});
});

test('the median ignores non-finite values and is NaN of none', () => {
  assert.equal(median([3, NaN, 1]), 2);
  assert.ok(Number.isNaN(median([NaN])));
});
