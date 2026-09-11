import React, { useState, useEffect } from 'react';
import { Shield, Activity, Radio, Cpu, Clock, CheckCircle2, AlertTriangle } from 'lucide-react';

export default function Header({ status, latestPrediction, mode, connected, isMlActive, topology }) {
  const [currentTime, setCurrentTime] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const modelMeta = latestPrediction?.model || status?.model_meta || null;

  const isLive = mode === 'LIVE' || status?.network_online || status?.sensor_active;
  const labMode = Boolean(status?.lab_mode);
  const networkOnline = status?.network_online ?? false;
  const sensorActive = status?.sensor_active ?? false;
  const latency = latestPrediction?.latency || { telemetry_ms: 0, inference_ms: 0, total_ms: 0 };
  const topoNodes = topology?.stats?.nodes ?? status?.topology_nodes ?? 0;
  const topoEdges = topology?.stats?.edges ?? status?.topology_edges ?? 0;
  const iface = status?.sensor_interface || 'SPAN';
  // TGNE stage output from the latest inference pass (never a frontend-derived value).
  const tgne = isMlActive ? (latestPrediction?.stages?.tgne ?? null) : null;

  return (
    <header className="soc-header">
      <div className="header-brand">
        <div className="brand-logo-container">
          <Shield className="brand-icon" size={28} />
          <div className="brand-pulse-ring" />
        </div>
        <div>
          <div className="brand-title">
            ANTI<span className="brand-highlight">GRAVITY</span>
          </div>
          <div className="brand-subtitle">DUAL-BRANCH WORLD MODEL &amp; DEEPOP CWA DECODER</div>
        </div>
      </div>

      <div className="header-telemetry-strip">
        <div className="strip-item">
          <div className="strip-label">OPERATING MODE</div>
          <div className={`badge ${isLive ? 'badge-live' : 'badge-standby'}`}>
            <span className={isLive ? 'dot-live' : 'dot-standby'} />
            {isLive ? 'LIVE' : 'STANDBY'}
          </div>
        </div>

        <div className="strip-item">
          <div className="strip-label">{labMode ? 'ENTERPRISE RANGE' : 'OBSERVED GRAPH'}</div>
          <div className="strip-value-row">
            {labMode ? (
              networkOnline ? (
                <CheckCircle2 size={16} color="var(--color-green)" />
              ) : (
                <AlertTriangle size={16} color="var(--color-amber)" />
              )
            ) : (
              <Activity size={16} color={topoNodes > 0 ? 'var(--color-cyan)' : 'var(--text-muted)'} />
            )}
            <span className={labMode ? (networkOnline ? 'text-green' : 'text-amber') : (topoNodes > 0 ? 'text-cyan' : 'text-muted')}>
              {labMode
                ? (networkOnline
                    ? `ONLINE (${status?.nodes_running ?? 0}/${status?.total_nodes ?? 15})`
                    : 'OFFLINE')
                : `${topoNodes} hosts / ${topoEdges} edges`}
            </span>
          </div>
        </div>

        <div className="strip-item">
          <div className="strip-label">PASSIVE SENSOR</div>
          <div className="strip-value-row">
            <Radio size={16} color={sensorActive ? 'var(--color-cyan)' : 'var(--text-muted)'} className={sensorActive ? 'sensor-radiating' : ''} />
            <span className={sensorActive ? 'text-cyan' : 'text-muted'}>
              {sensorActive ? `SPAN (${iface})` : 'OFF'}
            </span>
          </div>
        </div>

        <div className="strip-item">
          <div className="strip-label">WORLD MODEL</div>
          <div className="strip-value-row">
            <Cpu size={16} color={isMlActive ? 'var(--color-purple)' : 'var(--text-muted)'} />
            {isMlActive && modelMeta ? (
              <span className="text-purple font-mono">
                {modelMeta.name} ({modelMeta.feature_count}D | L={modelMeta.history_steps} | {modelMeta.window_seconds}s | K={modelMeta.forecast_steps})
              </span>
            ) : (
              <span className="text-muted font-mono">
                STANDBY
              </span>
            )}
          </div>
        </div>

        {/* TGNE / network state for the host actually being scored this window */}
        <div className="strip-item">
          <div className="strip-label">TGNE STATE</div>
          <div className="strip-value-row font-mono">
            <Activity size={16} color={tgne ? 'var(--color-cyan)' : 'var(--text-muted)'} />
            {tgne ? (
              <span className="text-cyan">
                H {tgne.embedding_dim}d · |H| {Number(tgne.embedding_norm).toFixed(3)} · {tgne.graph_nodes}n/{tgne.graph_edges}e
              </span>
            ) : (
              <span className="text-muted">—</span>
            )}
          </div>
        </div>

        <div className="strip-item">
          <div className="strip-label">TOTAL LATENCY</div>
          <div className="strip-value-row font-mono">
            <Activity size={16} color={isMlActive ? 'var(--color-cyan)' : 'var(--text-muted)'} />
            <span className={isMlActive ? 'text-highlight' : 'text-muted'}>
              {isMlActive && latency.total_ms > 0 ? `${latency.total_ms.toFixed(1)} ms` : '—'}
            </span>
            {isMlActive && modelMeta?.window_seconds && latency.total_ms > 0 && (
              <span className="text-muted text-xs">
                (&lt; {((latency.total_ms / (modelMeta.window_seconds * 1000)) * 100).toFixed(1)}% budget)
              </span>
            )}
          </div>
        </div>

        <div className="strip-item">
          <div className="strip-label">SYSTEM TIME (UTC)</div>
          <div className="strip-value-row font-mono text-highlight">
            <Clock size={16} color="var(--text-muted)" />
            {currentTime.toISOString().substring(11, 19)}
            {!connected && <span className="text-amber text-xs"> · WS</span>}
          </div>
        </div>
      </div>
    </header>
  );
}
