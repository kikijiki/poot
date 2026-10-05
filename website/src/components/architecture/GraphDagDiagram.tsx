import React, { useState } from 'react';

type NodeDef = {
  id: string;
  label: string;
  shape: string;
  category: 'matmul' | 'reduce' | 'elemwise' | 'io';
  x: number;
  y: number;
};

type Edge = { from: string; to: string };

// A small slice of a decode attention step: Q@K^T -> scale -> +mask -> softmax -> @V
const NODES: NodeDef[] = [
  { id: 'q',      label: 'q',           shape: '[H,D]',      category: 'io',       x: 80,  y: 20 },
  { id: 'k',      label: 'k_cache',     shape: '[H,cap,D]',  category: 'io',       x: 280, y: 20 },
  { id: 'v',      label: 'v_cache',     shape: '[H,cap,D]',  category: 'io',       x: 480, y: 20 },
  { id: 'mask',   label: 'causal_mask', shape: '[cap]',      category: 'io',       x: 380, y: 130 },
  { id: 'qk',     label: 'MatMul',      shape: '[H,cap]',    category: 'matmul',   x: 180, y: 130 },
  { id: 'scale',  label: 'Mul / sqrt(D)', shape: '[H,cap]',    category: 'elemwise', x: 180, y: 230 },
  { id: 'masked', label: 'Add mask',    shape: '[H,cap]',    category: 'elemwise', x: 280, y: 330 },
  { id: 'rmax',   label: 'Reduce Max',  shape: '[H,1]',      category: 'reduce',   x: 80,  y: 440 },
  { id: 'sub',    label: 'Sub',         shape: '[H,cap]',    category: 'elemwise', x: 280, y: 440 },
  { id: 'exp',    label: 'Exp',         shape: '[H,cap]',    category: 'elemwise', x: 280, y: 540 },
  { id: 'rsum',   label: 'Reduce Sum',  shape: '[H,1]',      category: 'reduce',   x: 80,  y: 540 },
  { id: 'div',    label: 'Div',         shape: '[H,cap]',    category: 'elemwise', x: 280, y: 640 },
  { id: 'out',    label: 'MatMul',      shape: '[H,D]',      category: 'matmul',   x: 380, y: 740 },
];

const EDGES: Edge[] = [
  { from: 'q',      to: 'qk' },
  { from: 'k',      to: 'qk' },
  { from: 'qk',     to: 'scale' },
  { from: 'scale',  to: 'masked' },
  { from: 'mask',   to: 'masked' },
  { from: 'masked', to: 'rmax' },
  { from: 'masked', to: 'sub' },
  { from: 'rmax',   to: 'sub' },
  { from: 'sub',    to: 'exp' },
  { from: 'exp',    to: 'rsum' },
  { from: 'exp',    to: 'div' },
  { from: 'rsum',   to: 'div' },
  { from: 'div',    to: 'out' },
  { from: 'v',      to: 'out' },
];

const CAT_COLOR: Record<NodeDef['category'], { fill: string; stroke: string; text: string }> = {
  matmul:  { fill: '#d4edda', stroke: '#28a745', text: '#155724' },
  reduce:  { fill: '#fff3cd', stroke: '#ffc107', text: '#856404' },
  elemwise:{ fill: '#cce5ff', stroke: '#007bff', text: '#004085' },
  io:      { fill: '#e2e3e5', stroke: '#6c757d', text: '#383d41' },
};

const W = 130, H = 38, R = 7;
const SVG_W = 620, SVG_H = 820;

function cx(n: NodeDef) { return n.x + W / 2; }
function cy(n: NodeDef) { return n.y + H / 2; }

export default function GraphDagDiagram(): React.ReactElement {
  const [hovered, setHovered] = useState<string | null>(null);
  const nodeMap = Object.fromEntries(NODES.map(n => [n.id, n]));

  return (
    <div style={{ margin: '1.5rem 0', overflowX: 'auto' }}>
      <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--ifm-color-emphasis-600)', textAlign: 'center', marginBottom: '0.5rem' }}>
        Primitive eqns for one attention layer (decode step): hover a node for details
      </div>
      <svg width={SVG_W} height={SVG_H} viewBox={`0 0 ${SVG_W} ${SVG_H}`} style={{ display: 'block', margin: '0 auto', maxWidth: '100%' }}>
        <defs>
          <marker id="arrowhead" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="#aaa" />
          </marker>
        </defs>

        {/* Edges */}
        {EDGES.map(e => {
          const a = nodeMap[e.from];
          const b = nodeMap[e.to];
          if (!a || !b) return null;
          const ax = cx(a), ay = cy(a) + H / 2 - 2;
          const bx = cx(b), by = cy(b) - H / 2 + 2;
          const mx = (ax + bx) / 2, my = (ay + by) / 2;
          const active = hovered === e.from || hovered === e.to;
          return (
            <line
              key={`${e.from}-${e.to}`}
              x1={ax} y1={ay} x2={bx} y2={by}
              stroke={active ? '#555' : '#ccc'}
              strokeWidth={active ? 1.8 : 1.2}
              markerEnd="url(#arrowhead)"
            />
          );
        })}

        {/* Nodes */}
        {NODES.map(n => {
          const col = CAT_COLOR[n.category];
          const active = hovered === n.id;
          return (
            <g
              key={n.id}
              onMouseEnter={() => setHovered(n.id)}
              onMouseLeave={() => setHovered(null)}
              style={{ cursor: 'default' }}
            >
              <rect
                x={n.x} y={n.y} width={W} height={H} rx={R}
                fill={col.fill}
                stroke={col.stroke}
                strokeWidth={active ? 2.5 : 1.5}
                filter={active ? 'drop-shadow(0 2px 4px rgba(0,0,0,0.18))' : undefined}
              />
              <text x={n.x + W / 2} y={n.y + H / 2 - 5} textAnchor="middle" fontSize={12} fontWeight={600} fill={col.text}>
                {n.label}
              </text>
              <text x={n.x + W / 2} y={n.y + H / 2 + 9} textAnchor="middle" fontSize={10} fill={col.text} opacity={0.75}>
                {n.shape}
              </text>
            </g>
          );
        })}
      </svg>

      {/* Legend */}
      <div style={{ display: 'flex', gap: '1.25rem', justifyContent: 'center', marginTop: '0.75rem', flexWrap: 'wrap', fontSize: '0.78rem' }}>
        {(Object.entries(CAT_COLOR) as [NodeDef['category'], typeof CAT_COLOR[keyof typeof CAT_COLOR]][]).map(([cat, col]) => (
          <span key={cat} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <span style={{ display: 'inline-block', width: 12, height: 12, borderRadius: 3, background: col.fill, border: `1.5px solid ${col.stroke}` }} />
            <span style={{ color: 'var(--ifm-color-emphasis-700)', textTransform: 'capitalize' }}>{cat === 'io' ? 'Input / const' : cat === 'elemwise' ? 'Elementwise' : cat.charAt(0).toUpperCase() + cat.slice(1)}</span>
          </span>
        ))}
      </div>
    </div>
  );
}
