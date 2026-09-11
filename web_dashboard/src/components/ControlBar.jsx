import React from 'react';
import {
  Play, Square, RefreshCw, Activity,
  Radio, CheckCircle, Flame, Cpu,
  ShieldAlert, Lock, Key, Slash
} from 'lucide-react';

export default function ControlBar({
  status,
  activeCommand,
  onRunCommand,
  isMlActive,
  onMitigate,
  site,
}) {
  const isBusy = !!activeCommand;
  const labMode = Boolean(status?.lab_mode ?? site?.lab_mode);
  const isNetworkOnline = Boolean(status?.network === 'running' || status?.network_online);
  const isSensorActive = Boolean(status?.sensor === 'running' || status?.sensor_active);
  const isTrafficActive = Boolean(status?.normal_traffic === 'running' || status?.workloads_active);
  const isAttackActive = Boolean(status?.attack === 'running' || status?.attack_active);

  // Non-lab: sensor does not require Containerlab "network online"
  const canStartSensor = labMode ? isNetworkOnline : true;
  const soarOnline = labMode ? isNetworkOnline : (isSensorActive || isNetworkOnline);

  const assets = site?.assets_of_interest || [];
  const isolateTarget = assets[0]?.ip || null;

  return (
    <div className="soc-card control-bar-card">
      <div className="control-bar-inner">

        {labMode && (
          <div className="control-group">
            <div className="group-label">LAB MODE — CYBER-RANGE</div>
            <div className="btn-group">
              <button
                className="btn btn-primary"
                onClick={() => onRunCommand('start_network')}
                disabled={isBusy || isNetworkOnline}
                title="Deploy Containerlab enterprise range"
              >
                <Play size={14} />
                START NETWORK
              </button>

              <button
                className="btn"
                onClick={() => onRunCommand('healthcheck')}
                disabled={isBusy || !isNetworkOnline}
                title="Audit reachability and security boundaries"
              >
                <CheckCircle size={14} />
                HEALTH CHECK
              </button>

              <button
                className="btn btn-danger"
                onClick={() => onRunCommand('stop_network')}
                disabled={isBusy || !isNetworkOnline}
                title="Teardown enterprise topology"
              >
                <Square size={14} />
                STOP
              </button>

              <button
                className="btn"
                onClick={() => onRunCommand('reset_environment')}
                disabled={isBusy}
                title="Reset telemetry state and return to standby"
              >
                <RefreshCw size={14} />
                RESET
              </button>
            </div>
          </div>
        )}

        {!labMode && (
          <div className="control-group">
            <div className="group-label">SITE — LOCAL SPAN</div>
            <div className="btn-group">
              <button
                className="btn"
                onClick={() => onRunCommand('reset_environment')}
                disabled={isBusy}
                title="Clear topology cache, defenses, ML standby"
              >
                <RefreshCw size={14} />
                RESET
              </button>
              <span className="badge badge-standby" style={{ alignSelf: 'center' }}>
                {status?.site_id || site?.site_id || 'local'} · {status?.sensor_interface || site?.sensor?.interface || 'iface'}
              </span>
            </div>
          </div>
        )}

        <div className="control-group">
          <div className="group-label">STREAMING TELEMETRY (SPAN)</div>
          <div className="btn-group">
            <button
              className="btn btn-primary"
              onClick={() => onRunCommand('start_telemetry')}
              disabled={isBusy || isSensorActive || !canStartSensor}
              title="Start live passive packet sniffer on SPAN/mirror interface"
            >
              <Radio size={14} />
              START SENSOR
            </button>

            <button
              className="btn"
              onClick={() => onRunCommand('stop_telemetry')}
              disabled={isBusy || !isSensorActive}
              title="Stop live passive sniffer"
            >
              <Square size={14} />
              STOP SENSOR
            </button>

            {labMode && (
              <button
                className="btn"
                onClick={() => onRunCommand('verify_telemetry')}
                disabled={isBusy || !isNetworkOnline}
                title="Verify telemetry state builder on live SPAN mirror"
              >
                <Activity size={14} />
                VERIFY
              </button>
            )}
          </div>
        </div>

        <div className="control-group">
          <div className="group-label">ML WORLD MODEL (DUAL-BRANCH + DEEPOP)</div>
          <div className="btn-group">
            <button
              className="btn btn-primary"
              onClick={() => onRunCommand('start_ml')}
              disabled={isBusy || !isSensorActive || isMlActive}
              title="Enable Dual-Branch + DeepOP on live SPAN flows"
            >
              <Cpu size={14} />
              START ML
            </button>

            <button
              className="btn btn-danger"
              onClick={() => onRunCommand('stop_ml')}
              disabled={isBusy || !isMlActive}
              title="Disable ML inference (clears predictions)"
            >
              <Square size={14} />
              STOP ML
            </button>

            <span className={`badge ${isMlActive ? 'badge-live' : 'badge-standby'}`} style={{ alignSelf: 'center' }}>
              ML: {isMlActive ? 'LIVE' : 'STANDBY'}
            </span>
          </div>
        </div>

        {labMode && (
          <div className="control-group">
            <div className="group-label">TRAFFIC &amp; EXTERNAL</div>
            <div className="btn-group">
              <button
                className="btn"
                onClick={() => onRunCommand('start_normal_traffic')}
                disabled={isBusy || !isNetworkOnline || isTrafficActive}
                title="Generate background enterprise traffic"
              >
                <Activity size={14} />
                WORKLOADS
              </button>

              <button
                className="btn"
                onClick={() => onRunCommand('stop_normal_traffic')}
                disabled={isBusy || !isTrafficActive}
                title="Stop background workloads"
              >
                <Square size={14} />
                STOP WORKLOADS
              </button>

              <span className={`badge ${isAttackActive ? 'badge-live' : 'badge-standby'}`} style={{ alignSelf: 'center' }}>
                {isAttackActive ? 'EXTERNAL ARMED' : 'EXTERNAL IDLE'}
              </span>

              <button
                className="btn btn-amber"
                onClick={() => onRunCommand('start_attack')}
                disabled={isBusy || isAttackActive}
                title="Arm dashboard for outside→enterprise traffic"
              >
                <Flame size={14} />
                ARM EXTERNAL
              </button>

              <button
                className="btn btn-danger"
                onClick={() => onRunCommand('stop_attack')}
                disabled={isBusy || !isAttackActive}
                title="Disarm external-attack monitoring"
              >
                <Square size={14} />
                DISARM
              </button>
            </div>
          </div>
        )}

        {!labMode && (
          <div className="control-group">
            <div className="group-label">EXTERNAL MONITORING</div>
            <div className="btn-group">
              <span className={`badge ${isAttackActive ? 'badge-live' : 'badge-standby'}`} style={{ alignSelf: 'center' }}>
                {isAttackActive ? 'EXTERNAL ARMED' : 'EXTERNAL IDLE'}
              </span>
              <button
                className="btn btn-amber"
                onClick={() => onRunCommand('start_attack')}
                disabled={isBusy || isAttackActive}
                title="Arm for outside traffic observation"
              >
                <Flame size={14} />
                ARM EXTERNAL
              </button>
              <button
                className="btn btn-danger"
                onClick={() => onRunCommand('stop_attack')}
                disabled={isBusy || !isAttackActive}
              >
                <Square size={14} />
                DISARM
              </button>
            </div>
          </div>
        )}

        <div className="control-group">
          <div className="group-label">DEFENSIVE SOAR (RECORDED)</div>
          <div className="btn-group">
            <button
              className="btn btn-soar"
              onClick={() => onMitigate && onMitigate('ISOLATE_HOST', isolateTarget)}
              disabled={isBusy || !soarOnline || !isolateTarget}
              title={isolateTarget ? `Isolate ${isolateTarget}` : 'Configure assets_of_interest in site YAML'}
            >
              <Slash size={14} />
              ISOLATE HOST
            </button>

            <button
              className="btn btn-soar"
              onClick={() => {
                const ip = window.prompt('Block IP (external host observed on topology):', '');
                if (ip) onMitigate && onMitigate('BLOCK_IP', ip.trim());
              }}
              disabled={isBusy || !soarOnline}
              title="Block an observed IP (no hardcoded attacker)"
            >
              <ShieldAlert size={14} />
              BLOCK IP
            </button>

            <button
              className="btn btn-soar"
              onClick={() => onMitigate && onMitigate('BLOCK_PORT', 80)}
              disabled={isBusy || !soarOnline}
              title="Record block for service port"
            >
              <Lock size={14} />
              BLOCK PORT
            </button>

            <button
              className="btn btn-soar"
              onClick={() => onMitigate && onMitigate('REVOKE_CREDENTIALS', 'compromised_admin')}
              disabled={isBusy || !soarOnline}
            >
              <Key size={14} />
              REVOKE CREDS
            </button>

            <button
              className="btn"
              onClick={() => onMitigate && onMitigate('CLEAR_DEFENSES', null)}
              disabled={isBusy || !soarOnline}
            >
              <RefreshCw size={14} />
              CLEAR
            </button>
          </div>
        </div>

      </div>

      {activeCommand && (
        <div className="active-command-indicator">
          <RefreshCw size={14} className="spin text-cyan" />
          <span>Executing allowlisted operation: <strong className="font-mono text-cyan">{activeCommand}</strong>...</span>
        </div>
      )}
    </div>
  );
}
