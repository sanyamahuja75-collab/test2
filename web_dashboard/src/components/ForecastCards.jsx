import React from 'react';
import { Clock, TrendingUp, TrendingDown, Minus } from 'lucide-react';

// Compact label for a DeepOP joint token like "CredentialAccess.T1110".
function techniqueLabel(stage) {
  if (!stage) return null;
  const [coarse, tech] = String(stage).split('.');
  if (coarse === 'Benign') return 'BENIGN';
  return tech ? `${tech}` : coarse.toUpperCase();
}

export default function ForecastCards({ latestPrediction, isMlActive }) {
  const forecast = isMlActive ? (latestPrediction?.forecast || []) : [];
  const threshold = isMlActive ? (latestPrediction?.model?.threshold ?? null) : null;
  const currentRisk = isMlActive ? (latestPrediction?.prediction?.risk ?? null) : null;
  const stages = isMlActive ? latestPrediction?.stages : null;
  const totalHorizonSec = (isMlActive && forecast.length > 0)
    ? forecast[forecast.length - 1].horizon_seconds
    : (isMlActive && latestPrediction?.model?.window_seconds && latestPrediction?.model?.forecast_steps
        ? latestPrediction.model.window_seconds * latestPrediction.model.forecast_steps
        : null);

  return (
    <div className="soc-card forecast-card-container">
      <div className="soc-card-header">
        <div className="soc-card-title">
          <Clock size={16} color="var(--color-purple)" />
          MULTI-STEP FORWARD FORECAST
        </div>
        <span className="font-mono text-xs text-muted">HORIZON: {totalHorizonSec != null ? `${totalHorizonSec}s` : '—'}</span>
      </div>

      {/* Provenance: risk comes from Branch B, technique from DeepOP — same pass. */}
      {stages && (
        <div className="stage-provenance font-mono text-xs text-muted">
          <span>BRANCH&nbsp;B · future risk · latent {stages.branch_b?.latent_dim}d × {stages.branch_b?.forecast_steps}</span>
          <span>DEEPOP · future technique · vocab {stages.deepop?.vocab_size}</span>
        </div>
      )}

      <div className="forecast-cards-grid">
        {forecast.length === 0 ? (
          <div className="forecast-empty-state text-muted font-mono text-xs">
            {isMlActive 
              ? '[WARMING UP: Accumulating historical sequence states...]'
              : '[ML STANDBY: Multi-step forward forecast inactive]'}
          </div>
        ) : (
          forecast.map((step, idx) => {
            const prevRisk = idx === 0 ? (currentRisk ?? step.risk) : forecast[idx - 1].risk;
            const diff = step.risk - prevRisk;
            const isEscalating = diff > 0.01;
            const isDeescalating = diff < -0.01;
            const isAlert = threshold != null && step.risk >= threshold;
            const riskPct = Math.round(step.risk * 100);

            let colorClass = 'text-green';
            let barColor = 'var(--color-green)';
            if (isAlert) {
              colorClass = 'text-red';
              barColor = 'var(--color-red)';
            } else if (step.risk >= 0.5) {
              colorClass = 'text-amber';
              barColor = 'var(--color-amber)';
            }

            return (
              <div key={`fc-step-${idx}`} className={`forecast-step-card ${isAlert ? 'border-alert' : ''}`}>
                <div className="fc-card-top">
                  <span className="fc-horizon-badge">T+{step.horizon_seconds}s</span>
                  {isEscalating && <TrendingUp size={14} className="text-red" />}
                  {isDeescalating && <TrendingDown size={14} className="text-green" />}
                  {!isEscalating && !isDeescalating && <Minus size={14} className="text-muted" />}
                </div>

                <div className="fc-card-metric font-mono">
                  <span className={`fc-risk-val ${colorClass}`}>{step.risk.toFixed(4)}</span>
                  <span className="fc-pct-sub">{riskPct}%</span>
                </div>

                <div className="fc-bar-track">
                  <div
                    className="fc-bar-fill"
                    style={{
                      width: `${Math.min(100, Math.max(0, riskPct))}%`,
                      backgroundColor: barColor
                    }}
                  />
                </div>

                {/* DeepOP: predicted future technique + decoder confidence for this step */}
                <div className="fc-technique font-mono">
                  <span className={`fc-tech-name ${techniqueLabel(step.predicted_stage) === 'BENIGN' ? 'text-muted' : colorClass}`}>
                    {techniqueLabel(step.predicted_stage) || '—'}
                  </span>
                  {step.technique_confidence != null && (
                    <span className="fc-tech-conf text-muted" title="DeepOP decoder confidence for this token">
                      {Math.round(step.technique_confidence * 100)}%
                    </span>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
