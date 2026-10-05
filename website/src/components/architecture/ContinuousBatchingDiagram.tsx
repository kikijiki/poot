import React from 'react';

// Shows continuous batching vs static batching for 5 requests of varying length.
// X-axis = decode step (token number). Y-axis = batch slot.

const REQUESTS = [
  { id: 'A', start: 0, end: 12, color: '#007bff' },
  { id: 'B', start: 0, end: 6,  color: '#28a745' },
  { id: 'C', start: 0, end: 18, color: '#dc3545' },
  // In static batching, D and E wait for the whole first batch to finish (step 18)
  // In continuous batching, D starts immediately when B finishes (step 6), E at step 12
  { id: 'D', start: 6,  end: 22, color: '#fd7e14' },
  { id: 'E', start: 12, end: 28, color: '#6f42c1' },
];
const STATIC_REQ = [
  { id: 'A', start: 0, end: 12, slot: 0, color: '#007bff' },
  { id: 'B', start: 0, end: 6,  slot: 1, color: '#28a745' },
  { id: 'C', start: 0, end: 18, slot: 2, color: '#dc3545' },
  // In static batching: batch 1 runs until step 18 (slowest), then batch 2 starts
  { id: 'D', start: 18, end: 30, slot: 0, color: '#fd7e14' },
  { id: 'E', start: 18, end: 40, slot: 1, color: '#6f42c1' },
];

const CONTINUOUS_SLOTS: { slot: number; reqId: string; start: number; end: number; color: string }[] = [
  { slot: 0, reqId: 'A', start: 0,  end: 12, color: '#007bff' },
  { slot: 1, reqId: 'B', start: 0,  end: 6,  color: '#28a745' },
  { slot: 2, reqId: 'C', start: 0,  end: 18, color: '#dc3545' },
  { slot: 1, reqId: 'D', start: 6,  end: 22, color: '#fd7e14' }, // B finishes at 6 -> D takes slot 1
  { slot: 0, reqId: 'E', start: 12, end: 28, color: '#6f42c1' }, // A finishes at 12 -> E takes slot 0
];

const N_SLOTS = 3;
const MAX_T_CONT = 28;
const MAX_T_STATIC = 40;

const SLOT_H = 28, T_SCALE = 14, PAD = { left: 52, top: 32, right: 20, bottom: 20 };

type Mode = 'continuous' | 'static';

function BatchChart({ mode }: { mode: Mode }) {
  const slots = mode === 'continuous' ? CONTINUOUS_SLOTS : STATIC_REQ;
  const MAX_T = mode === 'continuous' ? MAX_T_CONT : MAX_T_STATIC;
  const W = PAD.left + MAX_T * T_SCALE + PAD.right;
  const H = PAD.top + N_SLOTS * (SLOT_H + 4) + PAD.bottom;

  // GPU utilization: fraction of token steps where at least one slot is active
  const totalSlotSteps = slots.reduce((s, r) => s + (r.end - r.start), 0);
  const maxSlotSteps = N_SLOTS * MAX_T;
  const util = Math.round(100 * totalSlotSteps / maxSlotSteps);

  // Idle gaps in static: slots 0 and 1 are idle between step 6/12 and step 18
  const idleGaps = mode === 'static' ? [
    { slot: 1, start: 6,  end: 18 },
    { slot: 0, start: 12, end: 18 },
  ] : [];

  return (
    <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ display: 'block', maxWidth: '100%' }}>
      {/* Slot labels */}
      {Array.from({ length: N_SLOTS }).map((_, s) => (
        <text key={s} x={PAD.left - 6} y={PAD.top + s * (SLOT_H + 4) + SLOT_H / 2 + 4}
          textAnchor="end" fontSize={10} fill="var(--ifm-color-emphasis-700)">
          slot {s}
        </text>
      ))}

      {/* Time axis ticks */}
      {Array.from({ length: MAX_T + 1 }).map((_, t) => t % 4 === 0 && (
        <g key={t}>
          <line x1={PAD.left + t * T_SCALE} y1={PAD.top - 4}
            x2={PAD.left + t * T_SCALE} y2={PAD.top + N_SLOTS * (SLOT_H + 4)}
            stroke="var(--ifm-color-emphasis-200)" strokeWidth={1} />
          <text x={PAD.left + t * T_SCALE} y={PAD.top - 7} textAnchor="middle" fontSize={9} fill="var(--ifm-color-emphasis-500)">
            {t}
          </text>
        </g>
      ))}

      {/* Idle gaps (static only) */}
      {idleGaps.map((g, i) => (
        <rect key={i}
          x={PAD.left + g.start * T_SCALE} y={PAD.top + g.slot * (SLOT_H + 4)}
          width={(g.end - g.start) * T_SCALE} height={SLOT_H}
          fill="#fff3cd" stroke="#ffc107" strokeWidth={1} rx={3}
        />
      ))}
      {idleGaps.map((g, i) => (
        <text key={`t${i}`}
          x={PAD.left + (g.start + g.end) / 2 * T_SCALE}
          y={PAD.top + g.slot * (SLOT_H + 4) + SLOT_H / 2 + 4}
          textAnchor="middle" fontSize={9} fill="#856404">
          idle
        </text>
      ))}

      {/* Request bars */}
      {slots.map((r, i) => {
        const slotIdx = typeof r.slot === 'number' ? r.slot : CONTINUOUS_SLOTS.indexOf(r as typeof CONTINUOUS_SLOTS[0]);
        const actualSlot = 'slot' in r ? (r as typeof STATIC_REQ[0]).slot : (r as typeof CONTINUOUS_SLOTS[0]).slot;
        // the two modes label a request differently (STATIC_REQ.id vs CONTINUOUS_SLOTS.reqId); normalize.
        const label = 'reqId' in r ? r.reqId : r.id;
        return (
          <g key={`${label}-${i}`}>
            <rect
              x={PAD.left + r.start * T_SCALE}
              y={PAD.top + actualSlot * (SLOT_H + 4)}
              width={(r.end - r.start) * T_SCALE - 2}
              height={SLOT_H}
              rx={4}
              fill={r.color + '33'}
              stroke={r.color}
              strokeWidth={1.8}
            />
            <text
              x={PAD.left + r.start * T_SCALE + (r.end - r.start) * T_SCALE / 2 - 1}
              y={PAD.top + actualSlot * (SLOT_H + 4) + SLOT_H / 2 + 4}
              textAnchor="middle" fontSize={10} fontWeight={700} fill={r.color}
            >
              {label}
            </text>
          </g>
        );
      })}

      {/* Axis label */}
      <text x={PAD.left + MAX_T * T_SCALE / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--ifm-color-emphasis-500)">
        token step
      </text>

      {/* Utilization badge */}
      <rect x={W - PAD.right - 72} y={PAD.top} width={72} height={22} rx={4}
        fill={util > 85 ? '#d4edda' : '#fff3cd'} stroke={util > 85 ? '#28a745' : '#ffc107'} />
      <text x={W - PAD.right - 36} y={PAD.top + 14} textAnchor="middle" fontSize={10} fontWeight={700}
        fill={util > 85 ? '#155724' : '#856404'}>
        {util}% util
      </text>
    </svg>
  );
}

export default function ContinuousBatchingDiagram(): React.ReactElement {
  const [mode, setMode] = React.useState<Mode>('static');

  return (
    <div style={{ margin: '1.5rem 0' }}>
      <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center', marginBottom: '0.75rem' }}>
        {(['static', 'continuous'] as Mode[]).map(m => (
          <button key={m} onClick={() => setMode(m)}
            style={{
              padding: '5px 16px', borderRadius: 6, fontSize: '0.83rem', cursor: 'pointer',
              border: mode === m ? '2px solid var(--ifm-color-primary)' : '1px solid var(--ifm-color-emphasis-300)',
              background: mode === m ? 'color-mix(in srgb, var(--ifm-color-primary) 10%, transparent)' : 'transparent',
              fontWeight: mode === m ? 700 : 400,
              color: mode === m ? 'var(--ifm-color-primary)' : 'var(--ifm-font-color-base)',
            }}
          >
            {m === 'static' ? 'Static batching' : 'Continuous batching'}
          </button>
        ))}
      </div>
      <div style={{ fontSize: '0.74rem', color: 'var(--ifm-color-emphasis-600)', textAlign: 'center', marginBottom: '0.5rem' }}>
        {mode === 'static'
          ? 'Static: slot idles until the whole batch finishes. Next batch waits for the slowest request.'
          : 'Continuous: request D claims slot 1 the moment B finishes (step 6). No idle time.'}
      </div>
      <div style={{ overflowX: 'auto' }}>
        <BatchChart mode={mode} />
      </div>
    </div>
  );
}
