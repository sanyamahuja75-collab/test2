import React, { useState, useEffect, useRef, useCallback } from 'react';
import Header from './components/Header';
import ControlBar from './components/ControlBar';
import NetworkTopology from './components/NetworkTopology';
import ThreatTrajectory from './components/ThreatTrajectory';
import ForecastCards from './components/ForecastCards';
import MitreStageCard from './components/MitreStageCard';
import ExplainabilityPanel from './components/ExplainabilityPanel';
import PredictionStoryBanner from './components/PredictionStoryBanner';
import ConsolePanel from './components/ConsolePanel';

export default function App() {
  const [status, setStatus] = useState(null);
  const [site, setSite] = useState(null);
  const [topology, setTopology] = useState({ nodes: [], edges: [], stats: { nodes: 0, edges: 0, external_nodes: 0 } });
  const [connected, setConnected] = useState(false);
  const [activeCommand, setActiveCommand] = useState(null);
  const [latestPrediction, setLatestPrediction] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [attackEvent, setAttackEvent] = useState(null);
  const [logs, setLogs] = useState([]);

  const wsRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);

  const isMlActive = Boolean(status?.ml === 'running' || status?.ml_active || status?.ml_status === 'live');
  const isNetworkOnline = Boolean(status?.network === 'running' || status?.network_online);
  const isTelemetryActive = Boolean(status?.sensor === 'running' || status?.sensor_active || status?.telemetry_active);
  const mode = (isNetworkOnline || isTelemetryActive) ? 'LIVE' : 'STANDBY';

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
        if (!data.attack_active) {
          setAttackEvent(null);
        }
        if (data.active_command) setActiveCommand(data.active_command);
      }
    } catch (e) {
      console.warn('Could not fetch initial status:', e);
    }
  }, []);

  const fetchSite = useCallback(async () => {
    try {
      const res = await fetch('/api/site');
      if (res.ok) setSite(await res.json());
    } catch (e) {
      console.warn('Could not fetch site config:', e);
    }
  }, []);

  const fetchTopology = useCallback(async () => {
    try {
      const res = await fetch('/api/topology');
      if (res.ok) setTopology(await res.json());
    } catch (e) {
      console.warn('Could not fetch topology:', e);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    fetchSite();
    fetchTopology();

    const connectWebSocket = () => {
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const host = window.location.host;
      const wsUrl = `${protocol}//${host}/ws`;

      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnected(true);
        setLogs(prev => [...prev, {
          type: 'command_output',
          timestamp: new Date().toISOString(),
          line: '[✓] Connected to CyberWorld Real-Time Event Bus (WebSocket).'
        }]);
      };

      ws.onmessage = (evt) => {
        try {
          const event = JSON.parse(evt.data);

          if (event.type === 'prediction') {
            setLatestPrediction(event);
            setPredictionHistory(prev => [...prev.slice(-30), event]);
          } else if (event.type === 'topology_update') {
            setTopology(event);
          } else if (event.type === 'ml_reset') {
            setLatestPrediction(null);
            setPredictionHistory([]);
          } else if (event.type === 'system_status') {
            setStatus(event);
            if (event.active_command) setActiveCommand(event.active_command);
            if (!event.attack_active) {
              setAttackEvent(null);
            }
          } else if (event.type === 'command_started') {
            setActiveCommand(event.command);
            setLogs(prev => [...prev, event]);
          } else if (event.type === 'command_completed') {
            setActiveCommand(null);
            if (event.command === 'stop_attack' || event.command === 'reset_environment' || event.command === 'stop_network') {
              setAttackEvent(null);
            }
            if (event.command === 'reset_environment' || event.command === 'stop_telemetry') {
              fetchTopology();
            }
            setLogs(prev => [...prev, event]);
            fetchStatus();
          } else if (event.type === 'command_output') {
            setLogs(prev => [...prev, event]);
          } else if (event.type === 'attack_event') {
            if (event.stage === 'STOPPED' || event.details?.includes('TERMINATED')) {
              setAttackEvent(null);
            } else {
              setAttackEvent(event);
            }
            setLogs(prev => [...prev, {
              type: 'attack_event',
              timestamp: new Date(event.timestamp * 1000).toISOString(),
              line: `[ATTACK] ${event.details}`
            }]);
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        setConnected(false);
        reconnectTimeoutRef.current = setTimeout(connectWebSocket, 2000);
      };

      ws.onerror = () => {
        ws.close();
      };
    };

    connectWebSocket();

    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
    };
  }, [fetchStatus, fetchSite, fetchTopology]);

  const handleRunCommand = async (cmdName, payload = {}) => {
    try {
      setActiveCommand(cmdName);
      const res = await fetch(`/api/command/${cmdName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Command failed');
      }
    } catch (e) {
      setActiveCommand(null);
      setLogs(prev => [...prev, {
        type: 'command_output',
        timestamp: new Date().toISOString(),
        line: `[ERROR] Failed to execute ${cmdName}: ${e.message}`
      }]);
    }
  };

  const handleMitigate = async (action, target) => {
    try {
      const res = await fetch('/api/mitigate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, target })
      });
      if (res.ok) {
        const data = await res.json();
        setLogs(prev => [...prev, {
          type: 'command_output',
          timestamp: new Date().toISOString(),
          line: `[SOAR PLAYBOOK] ${data.message || action}`
        }]);
      }
    } catch (e) {
      setLogs(prev => [...prev, {
        type: 'command_output',
        timestamp: new Date().toISOString(),
        line: `[ERROR] Mitigation failed (${action}): ${e.message}`
      }]);
    }
  };

  const handleClearLogs = () => {
    setLogs([]);
  };

  return (
    <div className="dashboard-container">
      <Header
        status={status}
        latestPrediction={latestPrediction}
        mode={mode}
        connected={connected}
        isMlActive={isMlActive}
        topology={topology}
      />

      <ControlBar
        status={status}
        site={site}
        activeCommand={activeCommand}
        onRunCommand={handleRunCommand}
        isMlActive={isMlActive}
        mode={mode}
        onMitigate={handleMitigate}
      />

      <div className="grid-top">
        <NetworkTopology
          latestPrediction={latestPrediction}
          status={status}
          topology={topology}
          isMlActive={isMlActive}
        />

        <ThreatTrajectory
          history={predictionHistory}
          latestPrediction={latestPrediction}
          isMlActive={isMlActive}
        />
      </div>

      <div className="grid-middle">
        <ForecastCards
          latestPrediction={latestPrediction}
          isMlActive={isMlActive}
        />
        <MitreStageCard
          latestPrediction={latestPrediction}
          isMlActive={isMlActive}
        />
        <ExplainabilityPanel
          latestPrediction={latestPrediction}
          isMlActive={isMlActive}
        />
      </div>

      <PredictionStoryBanner
        latestPrediction={latestPrediction}
        attackEvent={attackEvent}
        isMlActive={isMlActive}
      />

      <ConsolePanel
        logs={logs}
        onClearLogs={handleClearLogs}
      />
    </div>
  );
}
