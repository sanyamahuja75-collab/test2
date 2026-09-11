import React from 'react';
import { ShieldCheck, Flame, TrendingUp, AlertTriangle, Crosshair, Zap, Activity } from 'lucide-react';

export default function PredictionStoryBanner({ latestPrediction, attackEvent, isMlActive }) {
  const isLivePrediction = Boolean(isMlActive && latestPrediction?.prediction);
  const isAlert = isLivePrediction && (latestPrediction?.prediction?.alert ?? false);
  const isAttackActive = latestPrediction?.attack_active ?? false;
  const earlyWarning = isLivePrediction ? latestPrediction?.early_warning : null;
  const latency = isLivePrediction ? (latestPrediction?.latency || { telemetry_ms: 0, inference_ms: 0, total_ms: 0 }) : { telemetry_ms: 0, inference_ms: 0, total_ms: 0 };
  const leadTime = isLivePrediction ? earlyWarning?.lead_time_seconds : null;
  const threshold = isLivePrediction ? (latestPrediction?.model?.threshold ?? null) : null;
  const windowSec = isLivePrediction ? (latestPrediction?.model?.window_seconds ?? null) : null;
  const windowBudgetMs = windowSec ? windowSec * 1000.0 : null;
  const spareCapacity = (windowBudgetMs && latency.total_ms > 0) 
    ? Math.max(0, 100.0 - (latency.total_ms / windowBudgetMs * 100.0)) 
    : null;

  // Determine current active narrative stage dynamically
  let activeStep = 1;
  if (!isLivePrediction) {
    activeStep = 1;
  } else if (leadTime != null) {
    activeStep = 6;
  } else if (earlyWarning?.actual_milestone_timestamp) {
    activeStep = 5;
  } else if (isAlert) {
    activeStep = 4;
  } else if (latestPrediction?.prediction?.risk != null && threshold != null && latestPrediction.prediction.risk >= threshold * 0.75) {
    activeStep = 3;
  } else if (isAttackActive) {
    activeStep = 2;
  }

  // No invented milestone text: this is DeepOP's first forecast step or nothing.
  // (It was previously defaulted to the hardcoded string "Lateral Pivot Probing",
  // which read as a model prediction on screen but came from the frontend.)
  const milestoneDesc =
    earlyWarning?.target_milestone_desc ||
    latestPrediction?.stages?.deepop?.technique_trajectory?.[0] ||
    null;
  const attackDesc = attackEvent?.details || latestPrediction?.attack_phase || 'Controlled Campaign Active';

  const steps = [
    { num: 1, title: 'NORMAL ENTERPRISE BASELINE', desc: 'Passive SPAN capture of benign enterprise traffic' },
    { num: 2, title: 'ATTACK CAMPAIGN LAUNCHED', desc: isAttackActive ? attackDesc : 'Adversarial campaign standby' },
    { num: 3, title: 'TRAJECTORY CHANGE DETECTED', desc: 'Flow compression & state vector dynamics shift' },
    { num: 4, title: 'PREDICTIVE ALERT TRIGGERED', desc: earlyWarning?.alert_timestamp ? `T_alert = ${new Date(earlyWarning.alert_timestamp * 1000).toISOString().substring(11, 19)}` : (isAlert ? 'Operating threshold exceeded' : 'Monitoring trajectory...') },
    { num: 5, title: 'ACTUAL TARGET MILESTONE', desc: earlyWarning?.actual_milestone_timestamp ? `T_actual = ${new Date(earlyWarning.actual_milestone_timestamp * 1000).toISOString().substring(11, 19)}` : (milestoneDesc ? `DeepOP next: ${milestoneDesc}` : 'Awaiting physical milestone') },
    { num: 6, title: 'DEFENDER EARLY WARNING', desc: leadTime != null ? `+${leadTime.toFixed(2)}s ADVANCE NOTICE` : 'Measured Lead Time' }
  ];

  return (
    <div className="soc-card storyboard-card">
      <div className="soc-card-header">
        <div className="soc-card-title">
          <Zap size={16} color="var(--color-green)" />
          PREDICTIVE DEFENSE VERIFICATION STORYBOARD
        </div>
        {leadTime != null && (
          <div className="lead-time-highlight-pill font-mono">
            ★ MEASURED EARLY WARNING: <strong>+{leadTime.toFixed(2)} SECONDS</strong> (T_alert &lt; T_actual)
          </div>
        )}
      </div>

      <div className="storyboard-body">
        {/* Step Progression Bar */}
        <div className="storyboard-steps-grid">
          {steps.map((step) => {
            const isCompleted = activeStep > step.num;
            const isCurrent = activeStep === step.num;

            return (
              <div 
                key={`story-step-${step.num}`} 
                className={`story-step-card ${isCurrent ? 'step-current' : ''} ${isCompleted ? 'step-done' : ''}`}
              >
                <div className="story-step-badge font-mono">
                  {isCompleted ? '✓' : `0${step.num}`}
                </div>
                <div className="story-step-content">
                  <div className="story-step-title">{step.title}</div>
                  <div className="story-step-desc font-mono text-xs text-muted">{step.desc}</div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Latency & Headroom Status Row */}
        <div className="storyboard-telemetry-row font-mono text-xs">
          <div className="lat-item">
            <span className="text-muted">STATE EXTRACTION:</span>
            <strong className="text-cyan">{isLivePrediction && latency.telemetry_ms > 0 ? `${latency.telemetry_ms.toFixed(1)} ms` : '—'}</strong>
          </div>
          <div className="lat-item">
            <span className="text-muted">WORLD MODEL INFERENCE:</span>
            <strong className="text-purple">{isLivePrediction && latency.inference_ms > 0 ? `${latency.inference_ms.toFixed(1)} ms` : '—'}</strong>
          </div>
          <div className="lat-item">
            <span className="text-muted">TOTAL PROCESSING LATENCY:</span>
            <strong className="text-highlight">{isLivePrediction && latency.total_ms > 0 ? `${latency.total_ms.toFixed(1)} ms` : '—'}</strong>
          </div>
          <div className="lat-item">
            <span className="text-muted">WINDOW BUDGET:</span>
            <strong className="text-muted">{isLivePrediction && windowBudgetMs != null ? `${windowBudgetMs.toFixed(1)} ms` : '—'}</strong>
          </div>
          <div className="lat-item">
            <span className="text-muted">PROCESSING HEADROOM:</span>
            <strong className={isLivePrediction && spareCapacity != null ? 'text-green' : 'text-muted'}>
              {isLivePrediction && spareCapacity != null ? `> ${spareCapacity.toFixed(1)}% SPARE CAPACITY` : '—'}
            </strong>
          </div>
        </div>
      </div>
    </div>
  );
}
