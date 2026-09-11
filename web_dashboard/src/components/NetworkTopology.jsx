import React, { useMemo, useState } from 'react';
import { Network } from 'lucide-react';

const VIEW_W = 760;
const VIEW_H = 410;
const MAX_RENDER_NODES = 80;

function layoutNodes(nodes) {
  const external = nodes.filter((n) => n.role === 'external');
  const internal = nodes.filter((n) => n.role !== 'external');

  const placeColumn = (list, x0, x1) => {
    const n = list.length;
    if (n === 0) return [];
    const cols = Math.min(3, Math.max(1, Math.ceil(Math.sqrt(n))));
    const rows = Math.ceil(n / cols);
    return list.map((node, i) => {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const x = x0 + ((col + 0.5) / cols) * (x1 - x0);
      const y = 70 + ((row + 0.5) / rows) * (VIEW_H - 100);
      return { ...node, x, y };
    });
  };

  return [
    ...placeColumn(external, 40, 280),
    ...placeColumn(internal, 320, 720),
  ];
}

function shortLabel(node) {
  if (node.label) return node.label;
  const ip = node.ip || node.id || '';
  if (ip.length <= 15) return ip;
  return `${ip.slice(0, 12)}…`;
}

export default function NetworkTopology({
  latestPrediction,
  status,
  topology,
  isMlActive,
}) {
  const [selectedNode, setSelectedNode] = useState(null);

  const isSensorActive = Boolean(status?.sensor === 'running' || status?.sensor_active);
  const isMlLive = Boolean(isMlActive && (status?.ml === 'running' || status?.ml_status === 'live'));
  const labMode = Boolean(status?.lab_mode);

  const focusIps = useMemo(() => {
    const fromPred = latestPrediction?.focus_ips || [];
    const target = latestPrediction?.target_ip ? [latestPrediction.target_ip] : [];
    return new Set([...fromPred, ...target].filter(Boolean));
  }, [latestPrediction]);

  const focusEdgeKeys = useMemo(() => {
    const edges = latestPrediction?.focus_edges || [];
    return new Set(edges.map((e) => `${e.src}->${e.dst}`));
  }, [latestPrediction]);

  const alertLive = Boolean(isMlLive && latestPrediction?.prediction?.alert);
  const risk = Number(latestPrediction?.prediction?.max_future_risk || latestPrediction?.prediction?.risk || 0);

  const rawNodes = topology?.nodes || [];
  const rawEdges = topology?.edges || [];
  const stats = topology?.stats || { nodes: 0, edges: 0, external_nodes: 0 };

  const truncated = rawNodes.length > MAX_RENDER_NODES;
  const visibleNodes = truncated
    ? [...rawNodes]
        .sort((a, b) => (b.bytes_in + b.bytes_out) - (a.bytes_in + a.bytes_out))
        .slice(0, MAX_RENDER_NODES)
    : rawNodes;

  const visibleIds = new Set(visibleNodes.map((n) => n.id || n.ip));
  const laidOut = layoutNodes(visibleNodes);
  const byId = Object.fromEntries(laidOut.map((n) => [n.id || n.ip, n]));

  const visibleEdges = rawEdges.filter(
    (e) => visibleIds.has(e.src) && visibleIds.has(e.dst)
  );

  const waiting = !isSensorActive && rawNodes.length === 0;
  const emptyLive = isSensorActive && rawNodes.length === 0;

  const getNodeColor = (node) => {
    const id = node.id || node.ip;
    const focused = focusIps.has(id);
    if (alertLive && focused) {
      if (risk >= 0.7) return 'var(--color-red)';
      if (risk >= 0.4) return 'var(--color-amber)';
      return 'var(--color-purple)';
    }
    if (node.role === 'external') return 'var(--color-amber)';
    if (node.stale) return 'var(--text-muted)';
    if ((node.risk || 0) >= 0.4) return 'var(--color-amber)';
    return 'var(--color-green)';
  };

  return (
    <div className="soc-card topology-card">
      <div className="soc-card-header">
        <div className="soc-card-title">
          <Network size={18} color={isSensorActive || rawNodes.length ? 'var(--color-cyan)' : 'var(--text-muted)'} />
          LIVE TOPOLOGY (SPAN DISCOVERY)
        </div>

        <div className="topology-legend">
          <span className="legend-item">
            <span className={`legend-dot ${isSensorActive ? 'dot-sensor' : 'dot-standby'}`} />
            SPAN {isSensorActive ? 'Live' : 'Off'}
          </span>
          <span className="legend-item font-mono">
            {stats.nodes || 0} hosts · {stats.edges || 0} edges
            {stats.external_nodes ? ` · ${stats.external_nodes} ext` : ''}
          </span>
        </div>
      </div>

      <div className="topology-svg-wrapper" style={{ position: 'relative' }}>
        <svg viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} className="topology-svg">
          <defs>
            <filter id="glowRed" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="5" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          <rect x="30" y="30" width="250" height={VIEW_H - 60} rx="8" className="zone-rect zone-attacker" />
          <text x="40" y="48" className="zone-label text-amber">EXTERNAL</text>

          <rect x="300" y="30" width="430" height={VIEW_H - 60} rx="8" className="zone-rect zone-users" />
          <text x="310" y="48" className="zone-label text-green">INTERNAL / ENTERPRISE</text>

          {visibleEdges.map((e, idx) => {
            const a = byId[e.src];
            const b = byId[e.dst];
            if (!a || !b) return null;
            const focused = focusEdgeKeys.has(`${e.src}->${e.dst}`) || focusEdgeKeys.has(`${e.dst}->${e.src}`);
            const hot = alertLive && focused;
            return (
              <line
                key={`${e.src}-${e.dst}-${e.dst_port}-${idx}`}
                x1={a.x}
                y1={a.y}
                x2={b.x}
                y2={b.y}
                className={hot ? 'net-link net-link-attack' : 'net-link net-link-workload'}
              />
            );
          })}

          {laidOut.map((n) => {
            const id = n.id || n.ip;
            const color = getNodeColor(n);
            const focused = focusIps.has(id);
            const isHovered = selectedNode?.id === id;
            return (
              <g
                key={id}
                className="topo-node-group"
                onClick={() => setSelectedNode(n)}
                style={{ cursor: 'pointer' }}
              >
                {alertLive && focused && (
                  <circle cx={n.x} cy={n.y} r="18" fill="none" stroke="var(--color-red)" strokeWidth="1.5" className="halo-pulse" />
                )}
                <circle
                  cx={n.x}
                  cy={n.y}
                  r={focused ? 13 : 10}
                  fill="var(--bg-surface)"
                  stroke={color}
                  strokeWidth={isHovered || focused ? 3 : 2}
                  strokeDasharray={n.role === 'external' && !focused ? '3 2' : 'none'}
                  filter={alertLive && focused ? 'url(#glowRed)' : 'none'}
                />
                <circle cx={n.x} cy={n.y} r={4} fill={color} />
                <text
                  x={n.x}
                  y={n.y + 20}
                  textAnchor="middle"
                  className="node-svg-label"
                  fill={isHovered ? '#fff' : 'var(--text-secondary)'}
                >
                  {shortLabel(n)}
                </text>
              </g>
            );
          })}
        </svg>

        {waiting && (
          <div className="topology-offline-banner font-mono text-xs text-muted">
            {labMode
              ? '[WAITING — start sensor after network deploy; topology fills from SPAN flows]'
              : '[WAITING FOR FLOWS — start sensor on the SPAN/mirror interface]'}
          </div>
        )}
        {emptyLive && (
          <div className="topology-offline-banner font-mono text-xs text-muted">
            [SENSOR LIVE — no hosts yet; generate traffic or wait for conversations]
          </div>
        )}
        {truncated && (
          <div className="topology-offline-banner font-mono text-xs text-muted" style={{ bottom: 40 }}>
            [Showing top {MAX_RENDER_NODES} of {rawNodes.length} hosts by volume]
          </div>
        )}

        {selectedNode && (
          <div className="node-detail-popover">
            <div className="popover-header">
              <strong>{selectedNode.label || selectedNode.ip}</strong>
              <button className="popover-close" onClick={() => setSelectedNode(null)}>&times;</button>
            </div>
            <div className="popover-row"><span>IP:</span> <code className="font-mono">{selectedNode.ip}</code></div>
            <div className="popover-row"><span>Role:</span> <span>{selectedNode.role}</span></div>
            <div className="popover-row"><span>Zone:</span> <span>{selectedNode.zone}</span></div>
            <div className="popover-row"><span>Bytes in/out:</span> <span className="font-mono">{selectedNode.bytes_in}/{selectedNode.bytes_out}</span></div>
            <div className="popover-row"><span>Risk:</span> <span className="font-mono">{Number(selectedNode.risk || 0).toFixed(2)}</span></div>
          </div>
        )}
      </div>
    </div>
  );
}
