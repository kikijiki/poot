import React, { useState } from 'react';

// Visualizes the flash attention decode algorithm.
// Shows Q (single row), K/V cache (chunked), LDS accumulator, and the chunk-by-chunk update.

const N_CHUNKS = 5;      // total K/V chunks shown
const HEAD_DIM = 8;      // visual head_dim (actual is larger)
const CHUNK_TOKENS = 4;  // tokens per chunk (visual)

// Colors
const C = {
  q:    { fill: '#d4edda', stroke: '#28a745', text: '#155724' },
  k:    { fill: '#cce5ff', stroke: '#007bff', text: '#004085' },
  v:    { fill: '#e2d9f3', stroke: '#6f42c1', text: '#2d1a5e' },
  acc:  { fill: '#fff3cd', stroke: '#ffc107', text: '#856404' },
  out:  { fill: '#d1ecf1', stroke: '#17a2b8', text: '#0c5460' },
  active: { fill: '#ffc107', stroke: '#e0a800' },
  done:   { fill: '#c3e6cb', stroke: '#28a745' },
};

const PAD = 20;
const COL1 = PAD;           // Q column
const COL2 = COL1 + 90;    // K cache column
const COL3 = COL2 + 210;   // V cache column
const COL4 = COL3 + 210;   // Accumulator column
const SVG_W = COL4 + 160;

const ROW_H = 22;
const CHUNK_H = CHUNK_TOKENS * ROW_H;
const CACHE_H = N_CHUNKS * CHUNK_H + (N_CHUNKS - 1) * 4;
const SVG_H = PAD * 3 + CACHE_H + 80;

function chunkY(chunk: number) {
  return PAD + 60 + chunk * (CHUNK_H + 4);
}

export default function FlashAttentionDiagram(): React.ReactElement {
  const [step, setStep] = useState(0); // 0 = start, 1..N_CHUNKS = after chunk i
  const maxStep = N_CHUNKS;

  function stepLabel(s: number) {
    if (s === 0) return 'Before processing any chunk';
    if (s === maxStep) return `Done: output = acc / l_running`;
    return `After chunk ${s} of ${N_CHUNKS}`;
  }

  return (
    <div style={{ margin: '1.5rem 0' }}>
      <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ifm-color-emphasis-600)', textAlign: 'center', marginBottom: '0.5rem' }}>
        Flash attention decode: one workgroup, streaming over K/V chunks
      </div>

      <svg width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        style={{ display: 'block', margin: '0 auto', maxWidth: '100%', overflow: 'visible' }}>

        {/* Column headers */}
        {[
          [COL1 + 35, 'Q (query)'],
          [COL2 + 90, 'K cache (chunked)'],
          [COL3 + 90, 'V cache (chunked)'],
          [COL4 + 65, 'LDS Accumulator'],
        ].map(([x, label]) => (
          <text key={label as string} x={x as number} y={PAD + 44} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ifm-color-emphasis-700)">
            {label as string}
          </text>
        ))}

        {/* Q: a single row */}
        <rect x={COL1} y={PAD + 55} width={70} height={ROW_H} rx={4}
          fill={C.q.fill} stroke={C.q.stroke} strokeWidth={1.5} />
        <text x={COL1 + 35} y={PAD + 55 + ROW_H / 2 + 4} textAnchor="middle" fontSize={10} fill={C.q.text} fontWeight={600}>
          q [D]
        </text>

        {/* K and V cache chunks */}
        {Array.from({ length: N_CHUNKS }).map((_, chunk) => {
          const y = chunkY(chunk);
          const isActive = step > 0 && step - 1 === chunk;
          const isDone = step > chunk + 1;
          const kColor = isDone ? C.done : isActive ? { fill: C.active.fill, stroke: C.active.stroke } : C.k;
          const vColor = isDone ? C.done : isActive ? { fill: C.active.fill, stroke: C.active.stroke } : C.v;

          return (
            <g key={chunk}>
              {/* K chunk */}
              <rect x={COL2} y={y} width={180} height={CHUNK_H} rx={4}
                fill={kColor.fill} stroke={kColor.stroke} strokeWidth={isActive ? 2 : 1.5} />
              <text x={COL2 + 90} y={y + CHUNK_H / 2 + 4} textAnchor="middle" fontSize={10} fill={C.k.text} fontWeight={isActive ? 700 : 400}>
                K[{chunk * CHUNK_TOKENS}..{(chunk + 1) * CHUNK_TOKENS}]
              </text>

              {/* V chunk */}
              <rect x={COL3} y={y} width={180} height={CHUNK_H} rx={4}
                fill={vColor.fill} stroke={vColor.stroke} strokeWidth={isActive ? 2 : 1.5} />
              <text x={COL3 + 90} y={y + CHUNK_H / 2 + 4} textAnchor="middle" fontSize={10} fill={C.v.text} fontWeight={isActive ? 700 : 400}>
                V[{chunk * CHUNK_TOKENS}..{(chunk + 1) * CHUNK_TOKENS}]
              </text>

              {/* arrow from active chunk */}
              {isActive && (
                <>
                  <line x1={COL2} y1={y + CHUNK_H / 2} x2={COL2 - 30} y2={y + CHUNK_H / 2} stroke="#e0a800" strokeWidth={1.5} markerEnd="url(#faArrow)" />
                  <line x1={COL3 + 180} y1={y + CHUNK_H / 2} x2={COL4 - 4} y2={y + CHUNK_H / 2} stroke="#e0a800" strokeWidth={1.5} markerEnd="url(#faArrow)" />
                </>
              )}
            </g>
          );
        })}

        {/* Accumulator box */}
        <rect x={COL4} y={PAD + 55} width={130} height={CACHE_H + 2} rx={6}
          fill={step > 0 ? C.acc.fill : '#f8f9fa'} stroke={C.acc.stroke} strokeWidth={step > 0 ? 2 : 1.5}
          style={{ transition: 'fill 0.3s' }}
        />
        {/* Accumulator contents */}
        {[
          ['m_running', step === 0 ? '-inf' : step < maxStep ? 'updating...' : 'final'],
          ['l_running', step === 0 ? '0' : step < maxStep ? 'updating...' : 'final'],
          ['o [D]', step === 0 ? '0...0' : step < maxStep ? 'accumulating...' : '/ l_running'],
        ].map(([key, val], i) => (
          <g key={key}>
            <text x={COL4 + 65} y={PAD + 55 + 40 + i * 38} textAnchor="middle" fontSize={11} fontWeight={700} fill={C.acc.text}>{key}</text>
            <text x={COL4 + 65} y={PAD + 55 + 56 + i * 38} textAnchor="middle" fontSize={10} fill={C.acc.text} opacity={0.8}>{val}</text>
          </g>
        ))}
        <text x={COL4 + 65} y={PAD + 55 + CACHE_H - 10} textAnchor="middle" fontSize={9} fill={C.acc.text} opacity={0.6}>
          workgroup-local (LDS)
        </text>

        {/* Output arrow */}
        {step === maxStep && (
          <>
            <line x1={COL4 + 65} y1={PAD + 55 + CACHE_H + 4}
              x2={COL4 + 65} y2={SVG_H - PAD - 20}
              stroke={C.out.stroke} strokeWidth={2} markerEnd="url(#faArrow)" />
            <rect x={COL4} y={SVG_H - PAD - 20} width={130} height={24} rx={5}
              fill={C.out.fill} stroke={C.out.stroke} strokeWidth={1.5} />
            <text x={COL4 + 65} y={SVG_H - PAD - 5} textAnchor="middle" fontSize={11} fontWeight={600} fill={C.out.text}>
              out [D] written
            </text>
          </>
        )}

        <defs>
          <marker id="faArrow" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
            <polygon points="0 0, 7 2.5, 0 5" fill="#e0a800" />
          </marker>
        </defs>
      </svg>

      {/* Controls */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', marginTop: '0.75rem' }}>
        <button onClick={() => setStep(s => Math.max(0, s - 1))} disabled={step === 0}
          style={{ padding: '4px 14px', borderRadius: 5, border: '1px solid var(--ifm-color-emphasis-300)', background: 'transparent', cursor: step === 0 ? 'default' : 'pointer', opacity: step === 0 ? 0.4 : 1 }}>
          &lt;- Prev
        </button>
        <span style={{ fontSize: '0.82rem', color: 'var(--ifm-color-emphasis-700)', minWidth: 230, textAlign: 'center' }}>
          {stepLabel(step)}
        </span>
        <button onClick={() => setStep(s => Math.min(maxStep, s + 1))} disabled={step === maxStep}
          style={{ padding: '4px 14px', borderRadius: 5, border: '1px solid var(--ifm-color-emphasis-300)', background: 'transparent', cursor: step === maxStep ? 'default' : 'pointer', opacity: step === maxStep ? 0.4 : 1 }}>
          Next -&gt;
        </button>
      </div>
      <div style={{ fontSize: '0.73rem', color: 'var(--ifm-color-emphasis-500)', textAlign: 'center', marginTop: 6 }}>
        No [H x cap] score matrix is ever materialized. Peak memory: O(head_dim) per workgroup.
      </div>
    </div>
  );
}
