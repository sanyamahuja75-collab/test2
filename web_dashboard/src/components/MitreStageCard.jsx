import React from 'react';
import { Target, CheckCircle2, ShieldAlert } from 'lucide-react';

const TACTICAL_PROGRESSION = [
  { id: 'RECONNAISSANCE', label: 'Reconnaissance', tactic: 'TA0043' },
  { id: 'INITIAL_ACCESS', label: 'Initial Access', tactic: 'TA0001' },
  { id: 'EXECUTION_PERSISTENCE', label: 'Execution', tactic: 'TA0002' },
  { id: 'LATERAL_MOVEMENT', label: 'Lateral Movement', tactic: 'TA0008' },
  { id: 'COMMAND_AND_CONTROL', label: 'Command & Control', tactic: 'TA0011' },
  { id: 'IMPACT_EXFILTRATION', label: 'Exfiltration', tactic: 'TA0010' }
];

export default function MitreStageCard({ latestPrediction, isMlActive }) {
  const pred = latestPrediction?.prediction;
  const isAvailable = Boolean(isMlActive && pred && pred.predicted_stage);

  if (!isAvailable) {
    return (
      <div className="soc-card mitre-card">
        <div className="soc-card-header">
          <div className="soc-card-title">
            <Target size={16} color="var(--text-muted)" />
            PREDICTED MITRE ATT&amp;CK STAGE
          </div>
          <span className="badge badge-standby">STANDBY</span>
        </div>
        <div className="mitre-card-body" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '140px' }}>
          <div className="font-mono text-xs text-muted" style={{ textAlign: 'center' }}>
            [ML STANDBY: MITRE stage classification inactive]
          </div>
        </div>
      </div>
    );
  }

  const currentStage = pred.predicted_stage;
  const tactic = pred.mitre_tactic || '';
  const technique = pred.mitre_technique || currentStage;
  const tacticId = pred.mitre_tactic_id || (currentStage === 'BENIGN' ? 'TA0000' : '');
  const description = pred.mitre_description || '';
  const isAttackStage = currentStage !== 'BENIGN' && currentStage !== 'Benign' && currentStage !== 'UNKNOWN';
  // Branch A produced every value on this card, in the same inference pass.
  const branchA = latestPrediction?.stages?.branch_a;

  // Derive progression dynamically if stage_probabilities are provided by runtime
  const displayStages = pred.stage_probabilities 
    ? Object.keys(pred.stage_probabilities)
        .filter(s => s !== 'BENIGN' && s !== 'ATTACK_OTHER')
        .map(s => {
          const match = TACTICAL_PROGRESSION.find(t => t.id === s);
          return {
            id: s,
            label: match ? match.label : s.replace(/_/g, ' ').toLowerCase().replace(/\b\w/g, c => c.toUpperCase()),
            tactic: match ? match.tactic : (s === currentStage && tacticId ? tacticId : 'STAGE'),
            prob: pred.stage_probabilities[s]
          };
        })
    : TACTICAL_PROGRESSION;

  return (
    <div className="soc-card mitre-card">
      <div className="soc-card-header">
        <div className="soc-card-title">
          <Target size={16} color={isAttackStage ? 'var(--color-red)' : 'var(--color-green)'} />
          PREDICTED MITRE ATT&amp;CK STAGE
        </div>
        <span className={`badge ${isAttackStage ? 'badge-critical' : 'badge-nominal'}`}>
          {currentStage}
        </span>
      </div>

      <div className="mitre-card-body">
        {/* Provenance: current risk / technique / confidence / target host from Branch A */}
        {branchA && (
          <div className="stage-provenance font-mono text-xs text-muted">
            <span>BRANCH&nbsp;A · in {branchA.input_dim}d × {branchA.history_steps}</span>
            <span>RISK {Number(branchA.risk).toFixed(4)}</span>
            <span>CONF {Math.round((branchA.confidence ?? 0) * 100)}%</span>
            <span>HOST {branchA.target_host || '—'}</span>
          </div>
        )}

        {/* Main Stage Banner */}
        <div className={`mitre-highlight-banner ${isAttackStage ? 'banner-alert' : 'banner-nominal'}`}>
          <div className="mitre-stage-title">
            {(tactic ? `${tactic} · ` : '') + currentStage.replace(/_/g, ' ')}
          </div>
          <div className="mitre-technique-sub font-mono">
            {tacticId && <span className="text-highlight font-bold">{tacticId}: </span>}
            {technique}
          </div>
          {description && (
            <div className="mitre-desc-text text-muted text-xs">
              {description}
            </div>
          )}
        </div>

        {/* Tactical Progression Stepper */}
        <div className="mitre-stepper-title text-muted text-xs font-mono">
          MITRE ATT&amp;CK TACTICAL MATRIX ALIGNMENT
        </div>
        <div className="mitre-stepper-grid">
          {displayStages.map((t) => {
            // Branch A emits a technique id (e.g. "T1046"); the stepper is a tactic
            // matrix, so align on the MITRE tactic id / tactic name the backend sent
            // rather than comparing a technique id to a tactic label (which never matched).
            const isActive =
              currentStage === t.id ||
              (tacticId && tacticId === t.tactic) ||
              (tactic && t.label && tactic.toLowerCase().replace(/[^a-z]/g, '') ===
                t.label.toLowerCase().replace(/[^a-z]/g, ''));
            return (
              <div 
                key={t.id} 
                className={`mitre-step-item ${isActive ? 'step-active' : ''}`}
              >
                <div className="step-dot" />
                <div className="step-info">
                  <div className="step-label">{t.label}</div>
                  <div className="step-id font-mono text-xs">{t.tactic}</div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
