import React, { useState } from 'react';

type Mode = 'fixed' | 'paged';

const MAX_SEQ = 16;         // visual capacity
const ACTUAL_TOKENS = 6;    // tokens currently in cache
const BLOCK_SIZE = 4;       // paged: tokens per block
const N_BLOCKS_POOL = 6;    // paged: total blocks in pool
const CELL_W = 28, CELL_H = 28;

// Colors
const C_FILLED  = { fill: '#cce5ff', stroke: '#007bff' };
const C_MASKED  = { fill: '#f8f9fa', stroke: '#dee2e6' };
const C_BLOCK   = { fill: '#cce5ff', stroke: '#007bff' };
const C_FREE    = { fill: '#f8f9fa', stroke: '#dee2e6' };
const C_SHARED  = { fill: '#d4edda', stroke: '#28a745' };  // shared prefix block

// Fixed: left-to-right fill, right tail masked
// Paged: pool of blocks, block table for a sequence

export default function KVCacheDiagram(): React.ReactElement {
  const [mode, setMode] = useState<Mode>('fixed');

  return (
    <div style={{ margin: '1.5rem 0' }}>
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginBottom: '1rem' }}>
        {(['fixed', 'paged'] as Mode[]).map(m => (
          <button key={m}
            onClick={() => setMode(m)}
            style={{
              padding: '5px 18px', borderRadius: 6, fontSize: '0.85rem', cursor: 'pointer',
              border: mode === m ? '2px solid var(--ifm-color-primary)' : '1px solid var(--ifm-color-emphasis-300)',
              background: mode === m ? 'color-mix(in srgb, var(--ifm-color-primary) 10%, transparent)' : 'transparent',
              fontWeight: mode === m ? 700 : 400,
              color: mode === m ? 'var(--ifm-color-primary)' : 'var(--ifm-font-color-base)',
            }}
          >
            {m === 'fixed' ? 'Fixed-capacity + masking' : 'Paged KV cache'}
          </button>
        ))}
      </div>

      {mode === 'fixed' ? <FixedView /> : <PagedView />}
    </div>
  );
}

function FixedView() {
  const [showMask, setShowMask] = useState(false);
  const W = MAX_SEQ * (CELL_W + 2) + 40;

  return (
    <div>
      <svg width={W} height={160} viewBox={`0 0 ${W} 160`} style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}>
        {/* Allocation bar */}
        <text x={W / 2} y={18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ifm-color-emphasis-700)">
          K/V allocation: cap = {MAX_SEQ} positions (pre-allocated)
        </text>
        {Array.from({ length: MAX_SEQ }).map((_, i) => {
          const x = 20 + i * (CELL_W + 2);
          const filled = i < ACTUAL_TOKENS;
          const col = filled ? C_FILLED : C_MASKED;
          return (
            <g key={i}>
              <rect x={x} y={28} width={CELL_W} height={CELL_H} rx={4}
                fill={col.fill} stroke={col.stroke} strokeWidth={1.5} />
              {filled && (
                <text x={x + CELL_W / 2} y={28 + CELL_H / 2 + 4} textAnchor="middle" fontSize={9} fill="#004085">{i}</text>
              )}
            </g>
          );
        })}

        {/* Labels */}
        <text x={20 + (ACTUAL_TOKENS - 1) * (CELL_W + 2) / 2 + ACTUAL_TOKENS * (CELL_W + 2) / 2} y={75} textAnchor="middle" fontSize={10} fill="#004085">
          {ACTUAL_TOKENS} filled
        </text>
        <text x={20 + ACTUAL_TOKENS * (CELL_W + 2) + (MAX_SEQ - ACTUAL_TOKENS) * (CELL_W + 2) / 2} y={75} textAnchor="middle" fontSize={10} fill="#6c757d">
          {MAX_SEQ - ACTUAL_TOKENS} empty (still allocated)
        </text>

        {/* Brace: filled region */}
        <line x1={20} y1={68} x2={20 + ACTUAL_TOKENS * (CELL_W + 2) - 2} y2={68} stroke="#007bff" strokeWidth={1.5} />
        <line x1={20 + ACTUAL_TOKENS * (CELL_W + 2)} y1={68} x2={20 + MAX_SEQ * (CELL_W + 2) - 2} y2={68} stroke="#adb5bd" strokeWidth={1.5} strokeDasharray="3 2" />

        {/* Mask row */}
        <text x={W / 2} y={96} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ifm-color-emphasis-700)">
          Additive mask: positions after pos get a large negative value before softmax
        </text>
        {Array.from({ length: MAX_SEQ }).map((_, i) => {
          const x = 20 + i * (CELL_W + 2);
          const valid = i < ACTUAL_TOKENS;
          return (
            <g key={i}>
              <rect x={x} y={104} width={CELL_W} height={CELL_H} rx={4}
                fill={valid ? '#d4edda' : '#fce8e8'} stroke={valid ? '#28a745' : '#dc3545'} strokeWidth={1.5} />
              <text x={x + CELL_W / 2} y={104 + CELL_H / 2 + 4} textAnchor="middle" fontSize={9}
                fill={valid ? '#155724' : '#721c24'}>
                {valid ? '0' : '-inf'}
              </text>
            </g>
          );
        })}
        <text x={W / 2} y={152} textAnchor="middle" fontSize={10} fill="var(--ifm-color-emphasis-600)">
          Grid dimensions are constant, the same every token, so one CUDA graph serves all positions
        </text>
      </svg>
    </div>
  );
}

function PagedView() {
  // Pool: 6 blocks (0-5). Seq A uses blocks 0,1 (8 tokens). Seq B shares block 0 (prefix) + uses block 2.
  const blocks = [
    { id: 0, label: 'Block 0', style: C_SHARED, note: 'shared prefix' },
    { id: 1, label: 'Block 1', style: C_BLOCK,  note: 'seq A' },
    { id: 2, label: 'Block 2', style: C_BLOCK,  note: 'seq B' },
    { id: 3, label: 'Block 3', style: C_FREE,   note: 'free' },
    { id: 4, label: 'Block 4', style: C_FREE,   note: 'free' },
    { id: 5, label: 'Block 5', style: C_FREE,   note: 'free' },
  ];
  const BW = 90, BH = 52, PAD = 16;
  const W = N_BLOCKS_POOL * (BW + 8) + PAD * 2 + 10;

  return (
    <div>
      <svg width={W} height={240} viewBox={`0 0 ${W} 240`} style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}>
        {/* Pool */}
        <text x={W / 2} y={18} textAnchor="middle" fontSize={11} fontWeight={600} fill="var(--ifm-color-emphasis-700)">
          Shared block pool: {BLOCK_SIZE} tokens/block
        </text>
        {blocks.map((b, i) => {
          const x = PAD + i * (BW + 8);
          return (
            <g key={b.id}>
              <rect x={x} y={26} width={BW} height={BH} rx={6} fill={b.style.fill} stroke={b.style.stroke} strokeWidth={1.5} />
              <text x={x + BW / 2} y={26 + BH / 2 - 5} textAnchor="middle" fontSize={10} fontWeight={600} fill="#383d41">{b.label}</text>
              <text x={x + BW / 2} y={26 + BH / 2 + 10} textAnchor="middle" fontSize={9} fill="#6c757d">{b.note}</text>
            </g>
          );
        })}

        {/* Seq A block table */}
        <text x={PAD + 25} y={102} fontSize={10} fontWeight={700} fill="var(--ifm-color-emphasis-700)">Seq A block table:</text>
        {[0, 1].map((logicalIdx, i) => {
          const physBlock = [0, 1][i];
          const x = PAD + 120 + i * 70;
          return (
            <g key={logicalIdx}>
              <rect x={x} y={110} width={60} height={26} rx={4} fill="#e9ecef" stroke="#adb5bd" strokeWidth={1} />
              <text x={x + 30} y={128} textAnchor="middle" fontSize={10} fill="#383d41">-&gt; blk {physBlock}</text>
              {/* arrow to pool */}
              <line x1={x + 30} y1={110}
                x2={PAD + physBlock * (BW + 8) + BW / 2}
                y2={26 + BH + 2}
                stroke={physBlock === 0 ? C_SHARED.stroke : C_BLOCK.stroke}
                strokeWidth={1.5} strokeDasharray="3 2"
                markerEnd="url(#kvArrow)"
              />
            </g>
          );
        })}

        {/* Seq B block table */}
        <text x={PAD + 25} y={154} fontSize={10} fontWeight={700} fill="var(--ifm-color-emphasis-700)">Seq B block table:</text>
        {[0, 2].map((physBlock, i) => {
          const x = PAD + 120 + i * 70;
          return (
            <g key={i}>
              <rect x={x} y={162} width={60} height={26} rx={4} fill="#e9ecef" stroke="#adb5bd" strokeWidth={1} />
              <text x={x + 30} y={180} textAnchor="middle" fontSize={10} fill="#383d41">-&gt; blk {physBlock}</text>
              <line x1={x + 30} y1={162}
                x2={PAD + physBlock * (BW + 8) + BW / 2}
                y2={26 + BH + 2}
                stroke={physBlock === 0 ? C_SHARED.stroke : C_BLOCK.stroke}
                strokeWidth={1.5} strokeDasharray="3 2"
                markerEnd="url(#kvArrow)"
              />
            </g>
          );
        })}

        {/* Shared block label */}
        <text x={PAD + 0 * (BW + 8) + BW / 2} y={210} textAnchor="middle" fontSize={9} fill={C_SHARED.stroke} fontWeight={600}>shared</text>
        <text x={PAD + 0 * (BW + 8) + BW / 2} y={221} textAnchor="middle" fontSize={9} fill={C_SHARED.stroke}>prefix cache</text>

        <text x={W / 2} y={238} textAnchor="middle" fontSize={10} fill="var(--ifm-color-emphasis-600)">
          Freed blocks return to the pool immediately, so memory scales with actual usage
        </text>

        <defs>
          <marker id="kvArrow" markerWidth="6" markerHeight="5" refX="6" refY="2.5" orient="auto">
            <polygon points="0 0, 6 2.5, 0 5" fill="#adb5bd" />
          </marker>
        </defs>
      </svg>
    </div>
  );
}
