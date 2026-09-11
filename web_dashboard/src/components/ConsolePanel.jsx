import React, { useState, useEffect, useRef } from 'react';
import { Terminal, Trash2, Download, Filter } from 'lucide-react';

export default function ConsolePanel({ logs, onClearLogs }) {
  const [filter, setFilter] = useState('ALL');
  const [autoScroll, setAutoScroll] = useState(true);
  const logContainerRef = useRef(null);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter(log => {
    if (filter === 'ALL') return true;
    if (filter === 'CONTROL') return log.type?.startsWith('command');
    if (filter === 'MODEL') return log.type === 'prediction' || log.line?.includes('Risk') || log.line?.includes('ALERT');
    if (filter === 'ATTACK') return log.type === 'attack_event' || log.line?.includes('ATTACK');
    if (filter === 'TELEMETRY') return log.line?.includes('Window #') || log.type === 'telemetry_window';
    return true;
  });

  const getLineClass = (log) => {
    const text = (log.line || '').toUpperCase();
    if (text.includes('[ERROR]') || text.includes('CRITICAL') || text.includes('ALERT') || text.includes('FAIL')) {
      return 'log-error';
    }
    if (text.includes('[ATTACKER]') || text.includes('WARNING') || text.includes('ELEVATED')) {
      return 'log-warning';
    }
    if (text.includes('[✓]') || text.includes('PASS') || text.includes('SUCCESS') || text.includes('ONLINE')) {
      return 'log-success';
    }
    if (text.includes('WINDOW #') || text.includes('PIPE:')) {
      return 'log-telemetry';
    }
    return 'log-info';
  };

  const exportLogs = () => {
    const content = logs.map(l => `[${l.timestamp || ''}] ${l.line || JSON.stringify(l)}`).join('\n');
    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `cyberworld_console_${new Date().toISOString().substring(0, 19)}.log`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="soc-card console-card">
      <div className="soc-card-header">
        <div className="soc-card-title">
          <Terminal size={16} color="var(--color-cyan)" />
          OPERATOR COMMAND &amp; TELEMETRY EVENT CONSOLE
        </div>

        <div className="console-actions">
          {/* Filter tabs */}
          <div className="console-filters">
            {['ALL', 'CONTROL', 'MODEL', 'ATTACK', 'TELEMETRY'].map((f) => (
              <button
                key={f}
                className={`filter-btn ${filter === f ? 'filter-btn-active' : ''}`}
                onClick={() => setFilter(f)}
              >
                {f}
              </button>
            ))}
          </div>

          <button 
            className="btn btn-subtle btn-xs" 
            onClick={() => setAutoScroll(!autoScroll)}
            title={autoScroll ? 'Pause auto-scroll' : 'Resume auto-scroll'}
          >
            {autoScroll ? 'AUTO-SCROLL: ON' : 'PAUSED'}
          </button>

          <button className="btn btn-subtle btn-xs" onClick={exportLogs} title="Export log file">
            <Download size={12} />
            EXPORT
          </button>

          <button className="btn btn-subtle btn-xs" onClick={onClearLogs} title="Clear terminal view">
            <Trash2 size={12} />
            CLEAR
          </button>
        </div>
      </div>

      <div className="console-body" ref={logContainerRef}>
        {filteredLogs.length === 0 ? (
          <div className="console-empty font-mono text-muted text-xs">
            [System initialized. Awaiting commands and passive telemetry frames...]
          </div>
        ) : (
          filteredLogs.map((log, idx) => (
            <div key={`log-${idx}`} className={`console-line font-mono ${getLineClass(log)}`}>
              <span className="log-ts text-muted">
                {log.timestamp ? log.timestamp.substring(11, 19) : ''}
              </span>
              <span className="log-msg">
                {log.line || (log.type === 'prediction' ? `[MODEL] Window #${log.state?.window_id} | Risk: ${log.prediction?.risk} | Stage: ${log.prediction?.predicted_stage}` : JSON.stringify(log))}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
