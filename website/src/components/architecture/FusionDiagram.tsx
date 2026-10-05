import React, { useState } from 'react';
import styles from './architecture.module.css';

// Shows the same 9-eqn attention softmax subgraph in two states:
// unfused (individual dispatch per eqn) vs fused (one flash-attention dispatch).

type View = 'unfused' | 'fused';

type NodeSpec = {
  id: string;
  label: string;
  sublabel: string;
  col: number; // 0-based
  row: number; // 0-based
  group: 'none' | 'pointwise' | 'reduction' | 'flash';
};

const NODES: NodeSpec[] = [
  { id: 'qk',     label: 'MatMul',      sublabel: 'Q @ K^T',      col: 1, row: 0, group: 'none' },
  { id: 'scale',  label: 'Mul',         sublabel: '* 1/sqrt(D)',  col: 1, row: 1, group: 'pointwise' },
  { id: 'mask',   label: 'Add',         sublabel: '+ causal mask', col: 1, row: 2, group: 'pointwise' },
  { id: 'rmax',   label: 'Reduce Max',  sublabel: 'axis=-1',       col: 0, row: 3, group: 'reduction' },
  { id: 'sub',    label: 'Sub',         sublabel: 'x - max',       col: 1, row: 3, group: 'reduction' },
  { id: 'exp',    label: 'Exp',         sublabel: 'e^x',           col: 1, row: 4, group: 'reduction' },
  { id: 'rsum',   label: 'Reduce Sum',  sublabel: 'axis=-1',       col: 0, row: 4, group: 'reduction' },
  { id: 'div',    label: 'Div',         sublabel: 'p = e/sum(e)',  col: 1, row: 5, group: 'reduction' },
  { id: 'pv',     label: 'MatMul',      sublabel: 'p @ V',         col: 1, row: 6, group: 'none' },
];

type Edge = { from: string; to: string };
const EDGES: Edge[] = [
  { from: 'qk', to: 'scale' }, { from: 'scale', to: 'mask' },
  { from: 'mask', to: 'rmax' }, { from: 'mask', to: 'sub' }, { from: 'rmax', to: 'sub' },
  { from: 'sub', to: 'exp' }, { from: 'exp', to: 'rsum' }, { from: 'exp', to: 'div' }, { from: 'rsum', to: 'div' },
  { from: 'div', to: 'pv' },
];

const GROUP_STYLE: Record<string, React.CSSProperties> = {
  none:      { background: '#e2e3e5', border: '1.5px solid #6c757d', color: '#383d41' },
  pointwise: { background: '#cce5ff', border: '1.5px solid #007bff', color: '#004085' },
  reduction: { background: '#fff3cd', border: '1.5px solid #ffc107', color: '#856404' },
  flash:     { background: '#d4edda', border: '2px solid #28a745',   color: '#155724' },
};

const CELL_W = 130, CELL_H = 50, GAP_X = 20, GAP_Y = 16;
const COLS = 2, ROWS = 7;
const SVG_W = COLS * (CELL_W + GAP_X) + 20;
const SVG_H = ROWS * (CELL_H + GAP_Y) + 20;

function nodeX(col: number) { return 10 + col * (CELL_W + GAP_X); }
function nodeY(row: number) { return 10 + row * (CELL_H + GAP_Y); }
function ncx(n: NodeSpec) { return nodeX(n.col) + CELL_W / 2; }
function ncy(n: NodeSpec) { return nodeY(n.row) + CELL_H / 2; }

export default function FusionDiagram(): React.ReactElement {
  const [view, setView] = useState<View>('unfused');
  const [hovered, setHovered] = useState<string | null>(null);
  const nodeMap = Object.fromEntries(NODES.map(n => [n.id, n]));

  const isFused = view === 'fused';

  // In fused view the whole chain (both matmuls included) is one FlashAttention eqn.
  const fusedNodes: { id: string; label: string; sublabel: string; col: number; row: number; group: 'none' | 'flash' }[] = [
    { id: 'flash_fused', label: 'FlashAttention dispatch', sublabel: 'one eqn, one dispatch', col: 1, row: 0, group: 'flash' },
  ];

  const FUSED_SVG_H = 2 * (CELL_H + GAP_Y) + 40;

  return (
    <div style={{ margin: '1.5rem 0' }}>
      <div className={styles.fusionToggleHeader}>
        <button className={`${styles.fusionBtn} ${view === 'unfused' ? styles.fusionBtnActive : ''}`} onClick={() => setView('unfused')}>
          Unfused: {NODES.length} dispatches
        </button>
        <button className={`${styles.fusionBtn} ${view === 'fused' ? styles.fusionBtnActive : ''}`} onClick={() => setView('fused')}>
          Fused: 1 dispatch
        </button>
      </div>

      {!isFused ? (
        <div style={{ overflowX: 'auto' }}>
          <div style={{ fontSize: '0.74rem', color: 'var(--ifm-color-emphasis-600)', marginBottom: 8, textAlign: 'center' }}>
            9 primitive eqns -&gt; 9 separate kernel dispatches. Hover to highlight.
          </div>
          <svg width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}>
            <defs>
              <marker id="arr2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#bbb" />
              </marker>
            </defs>
            {EDGES.map(e => {
              const a = nodeMap[e.from], b = nodeMap[e.to];
              if (!a || !b) return null;
              const active = hovered === e.from || hovered === e.to;
              return (
                <line key={`${e.from}-${e.to}`}
                  x1={ncx(a)} y1={ncy(a) + CELL_H / 2 - 2}
                  x2={ncx(b)} y2={ncy(b) - CELL_H / 2 + 2}
                  stroke={active ? '#666' : '#ddd'} strokeWidth={active ? 1.8 : 1.2}
                  markerEnd="url(#arr2)"
                />
              );
            })}
            {NODES.map(n => {
              const st = GROUP_STYLE[n.group];
              const active = hovered === n.id;
              const x = nodeX(n.col), y = nodeY(n.row);
              return (
                <g key={n.id} onMouseEnter={() => setHovered(n.id)} onMouseLeave={() => setHovered(null)} style={{ cursor: 'default' }}>
                  <rect x={x} y={y} width={CELL_W} height={CELL_H} rx={6}
                    fill={st.background as string} stroke={st.border as string}
                    strokeWidth={active ? 2.5 : 1.5}
                    filter={active ? 'drop-shadow(0 2px 4px rgba(0,0,0,0.15))' : undefined}
                  />
                  <text x={x + CELL_W / 2} y={y + CELL_H / 2 - 7} textAnchor="middle" fontSize={12} fontWeight={600} fill={st.color as string}>{n.label}</text>
                  <text x={x + CELL_W / 2} y={y + CELL_H / 2 + 8} textAnchor="middle" fontSize={10} fill={st.color as string} opacity={0.8}>{n.sublabel}</text>
                </g>
              );
            })}
          </svg>
          {/* Group legend */}
          <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', marginTop: 10, flexWrap: 'wrap', fontSize: '0.78rem' }}>
            {[['none', '#6c757d', 'Stand-alone (MatMul boundary)'], ['pointwise', '#007bff', 'Pointwise chain -> fuses with reduction'], ['reduction', '#ffc107', 'Reduction-rooted region']] .map(([g, c, label]) => (
              <span key={g} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                <span style={{ width: 12, height: 12, borderRadius: 3, background: GROUP_STYLE[g].background as string, border: `1.5px solid ${c}`, display: 'inline-block' }} />
                <span style={{ color: 'var(--ifm-color-emphasis-700)' }}>{label}</span>
              </span>
            ))}
          </div>
        </div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <div style={{ fontSize: '0.74rem', color: 'var(--ifm-color-emphasis-600)', marginBottom: 8, textAlign: 'center' }}>
            After compile's flash-attention rewrite: both matmuls, scale, mask, and softmax -&gt; one fused dispatch.
          </div>
          <svg width={SVG_W} height={FUSED_SVG_H} viewBox={`0 0 ${SVG_W} ${FUSED_SVG_H}`} style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}>
            <defs>
              <marker id="arr3" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
                <polygon points="0 0, 7 2.5, 0 5" fill="#bbb" />
              </marker>
            </defs>
            {fusedNodes.map((n, i) => {
              const st = GROUP_STYLE[n.group];
              const x = nodeX(n.col);
              const y = nodeY(n.row);
              const h = n.group === 'flash' ? CELL_H * 2.2 : CELL_H;
              if (i > 0) {
                const prev = fusedNodes[i - 1];
                const ay = nodeY(prev.row) + (prev.group === 'flash' ? CELL_H * 2.2 : CELL_H);
                const by = y;
              }
              return (
                <g key={n.id}>
                  {i > 0 && (() => {
                    const prev = fusedNodes[i - 1];
                    const prevH = prev.group === 'flash' ? CELL_H * 2.2 : CELL_H;
                    const ay = nodeY(prev.row) + prevH - 2;
                    const by = y;
                    return <line x1={nodeX(prev.col) + CELL_W / 2} y1={ay} x2={x + CELL_W / 2} y2={by} stroke="#ddd" strokeWidth={1.2} markerEnd="url(#arr3)" />;
                  })()}
                  <rect x={x} y={y} width={CELL_W} height={h} rx={8}
                    fill={st.background as string} stroke={st.border as string} strokeWidth={2}
                  />
                  <text x={x + CELL_W / 2} y={y + h / 2 - 8} textAnchor="middle" fontSize={12} fontWeight={700} fill={st.color as string}>{n.label}</text>
                  <text x={x + CELL_W / 2} y={y + h / 2 + 8} textAnchor="middle" fontSize={10} fill={st.color as string} opacity={0.85}>{n.sublabel}</text>
                  {n.group === 'flash' && (
                    <text x={x + CELL_W / 2} y={y + h / 2 + 24} textAnchor="middle" fontSize={9} fill={st.color as string} opacity={0.65}>replaces 9 eqns</text>
                  )}
                </g>
              );
            })}
          </svg>
          <div style={{ fontSize: '0.78rem', color: 'var(--ifm-color-emphasis-600)', textAlign: 'center', marginTop: 8 }}>
            The planner chooses the kernel for this one op: a synthesized region kernel within the LDS head-dim cap; a softcapped prefill uses the imported <code>#[kernel]</code> flash kernel.
          </div>
        </div>
      )}
    </div>
  );
}
