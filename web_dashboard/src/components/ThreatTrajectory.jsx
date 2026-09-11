import React, { useMemo, useState } from 'react';
import { TrendingUp, AlertTriangle, ShieldCheck, Zap, Activity } from 'lucide-react';

export default function ThreatTrajectory({ 
  history, 
  latestPrediction,
  isMlActive
}) {
  const [hoveredPoint, setHoveredPoint] = useState(null);

  const isLivePrediction = Boolean(isMlActive && latestPrediction?.prediction);
  const threshold = isLivePrediction ? (latestPrediction?.model?.threshold ?? null) : null;
  const currentRisk = isLivePrediction ? (latestPrediction.prediction.risk ?? null) : null;
  const alertLevel = isLivePrediction ? (latestPrediction.prediction.alert_level || 'NOMINAL') : 'STANDBY';
  const isAlert = isLivePrediction ? (latestPrediction.prediction.alert ?? false) : false;
  const forecast = isLivePrediction ? (latestPrediction?.forecast || []) : [];
  const earlyWarning = isLivePrediction ? latestPrediction?.early_warning : null;

  // Coordinate space: 620 width x 220 height
  const W = 620;
  const H = 220;
  const padL = 45;
  const padR = 85;
  const padT = 25;
  const padB = 30;

  const chartW = W - padL - padR;
  const chartH = H - padT - padB;

  // Prepare points from history (limit to last 25 windows)
  const windowSlice = useMemo(() => {
    if (!isLivePrediction) return [];
    const records = (history || []).filter(r => r?.prediction != null).slice(-25);
    return records;
  }, [history, isLivePrediction]);

  const numPoints = Math.max(12, windowSlice.length + forecast.length);

  // Map coordinates
  const getX = (index) => padL + (index / (numPoints - 1)) * chartW;
  const getY = (val) => padT + (1.0 - Math.max(0, Math.min(1.0, val))) * chartH;

  // Threshold Y coordinate
  const thresholdY = threshold != null ? getY(threshold) : null;

  // 1. Process Historical Points (Both Actual & Model's Prior Prediction for each event)
  const histPoints = useMemo(() => {
    return windowSlice.map((rec, i) => {
      const actualRisk = rec.prediction?.risk ?? 0.0;
      // Prior prediction made in advance for this specific window:
      let predictedRisk = null;
      if (i > 0 && windowSlice[i - 1]?.forecast?.[0]?.risk != null) {
        // Direct 1-step advance forecast that the model produced at step i-1 for this exact event:
        predictedRisk = windowSlice[i - 1].forecast[0].risk;
      } else if (rec.prediction?.predicted_risk_prior != null) {
        predictedRisk = rec.prediction.predicted_risk_prior;
      } else {
        predictedRisk = actualRisk;
      }

      const delta = actualRisk - predictedRisk;
      const x = getX(i);

      return {
        idx: i,
        x,
        yActual: getY(actualRisk),
        yPredicted: getY(predictedRisk),
        actualRisk,
        predictedRisk,
        delta,
        isAlert: rec.prediction?.alert ?? false,
        wallClock: rec.wall_clock,
        windowId: rec.state?.window_id ?? i + 1,
        stage: rec.prediction?.predicted_stage || 'BENIGN',
        technique: rec.prediction?.mitre_technique || 'Benign'
      };
    });
  }, [windowSlice, numPoints]);

  const lastHistPt = histPoints[histPoints.length - 1] || (currentRisk != null ? {
    x: padL,
    yActual: getY(currentRisk),
    yPredicted: getY(currentRisk),
    actualRisk: currentRisk,
    predictedRisk: currentRisk,
    delta: 0.0,
    windowId: 1
  } : null);

  // Latest metrics
  const latestPriorPredicted = lastHistPt?.predictedRisk ?? currentRisk;
  const latestDelta = lastHistPt ? (lastHistPt.actualRisk - lastHistPt.predictedRisk) : 0.0;

  // 2. Build Paths:
  // (A) Actual Observed Telemetry Path (Cyan) - ends at lastHistPt
  const actualPathD = histPoints.reduce((acc, pt, i) => {
    return i === 0 ? `M ${pt.x} ${pt.yActual}` : `${acc} L ${pt.x} ${pt.yActual}`;
  }, '');

  // (B) Future Forecast Points
  const forecastPoints = lastHistPt ? forecast.map((fc, i) => {
    const idx = (histPoints.length - 1) + (i + 1);
    return {
      idx,
      x: getX(idx),
      y: getY(fc.risk),
      risk: fc.risk,
      horizonSec: fc.horizon_seconds,
      predictedStage: fc.predicted_stage
    };
  }) : [];

  // (C) Full Model Forecast Curve (Violet) - spans across past predictions AND into future forecast!
  const allPredictedPoints = [
    ...histPoints.map(p => ({ x: p.x, y: p.yPredicted })),
    ...forecastPoints.map(p => ({ x: p.x, y: p.y }))
  ];

  const modelForecastPathD = allPredictedPoints.reduce((acc, pt, i) => {
    return i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
  }, '');

  // Future segment path for purple cone fill
  const futureForecastPathD = (lastHistPt && forecastPoints.length > 0) ? [
    { x: lastHistPt.x, y: lastHistPt.yPredicted },
    ...forecastPoints
  ].reduce((acc, pt, i) => {
    return i === 0 ? `M ${pt.x} ${pt.y}` : `${acc} L ${pt.x} ${pt.y}`;
  }, '') : '';

  // Alert marker point
  const firstAlertIdx = windowSlice.findIndex(r => r.prediction?.alert);
  const alertPt = firstAlertIdx !== -1 ? histPoints[firstAlertIdx] : null;

  return (
    <div className="soc-card trajectory-card">
      <div className="soc-card-header">
        <div className="soc-card-title">
          <TrendingUp size={18} color="var(--color-cyan)" />
          THREAT TRAJECTORY &amp; MULTI-STEP FORECAST HORIZON
        </div>
        <div className="trajectory-header-metrics font-mono">
          <span className="text-muted">THRESHOLD: <strong className="text-amber">{threshold != null ? threshold.toFixed(4) : '—'}</strong></span>
          <span className="text-muted">ACTUAL: <strong className={currentRisk != null ? (currentRisk >= (threshold ?? 1.0) ? 'text-red' : (currentRisk > 0.5 ? 'text-amber' : 'text-cyan')) : 'text-muted'}>{currentRisk != null ? currentRisk.toFixed(4) : '—'}</strong></span>
          <span className="text-muted">PREDICTED: <strong className="text-purple">{latestPriorPredicted != null ? latestPriorPredicted.toFixed(4) : '—'}</strong></span>
          {latestDelta != null && (
            <span className="text-muted">
              Δ: <strong className={Math.abs(latestDelta) < 0.15 ? 'text-green' : 'text-amber'}>
                {latestDelta > 0 ? `+${latestDelta.toFixed(4)}` : latestDelta.toFixed(4)}
              </strong>
            </span>
          )}
          <span className={`badge badge-${alertLevel.toLowerCase()}`}>{alertLevel}</span>
        </div>
      </div>

      {/* Trajectory Legend Bar */}
      <div className="trajectory-legend-bar font-mono">
        <div className="trajectory-legend-item">
          <span className="legend-line-actual" />
          <span className="text-cyan">ACTUAL OBSERVED (TELEMETRY)</span>
        </div>
        <div className="trajectory-legend-item">
          <span className="legend-line-predicted" />
          <span className="text-purple">MODEL PREDICTED (WORLD MODEL &amp; DEEPOP)</span>
        </div>
      </div>

      <div className="trajectory-svg-container">
        <svg viewBox={`0 0 ${W} ${H}`} className="trajectory-svg">
          <defs>
            <linearGradient id="histAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="var(--color-cyan)" stopOpacity="0.22" />
              <stop offset="100%" stopColor="var(--color-cyan)" stopOpacity="0.0" />
            </linearGradient>
            <linearGradient id="forecastAreaGrad" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="var(--color-purple)" stopOpacity="0.28" />
              <stop offset="100%" stopColor="var(--color-purple)" stopOpacity="0.0" />
            </linearGradient>
          </defs>

          {/* Grid lines */}
          {[0.0, 0.25, 0.5, 0.75, 1.0].map(level => {
            const y = getY(level);
            return (
              <g key={`grid-${level}`}>
                <line x1={padL} y1={y} x2={W - padR + 20} y2={y} stroke="rgba(255, 255, 255, 0.05)" strokeDasharray="3 3" />
                <text x={padL - 8} y={y + 4} textAnchor="end" className="chart-axis-label">{level.toFixed(2)}</text>
              </g>
            );
          })}

          {/* Calibrated Operating Threshold Line */}
          {thresholdY != null && (
            <g>
              <line 
                x1={padL} 
                y1={thresholdY} 
                x2={W - padR + 20} 
                y2={thresholdY} 
                stroke="var(--color-amber)" 
                strokeWidth="1.5" 
                strokeDasharray="5 4" 
                opacity="0.85"
              />
              <text x={W - padR + 25} y={thresholdY + 4} fill="var(--color-amber)" className="chart-tag-label font-mono">
                θ* = {threshold.toFixed(4)}
              </text>
            </g>
          )}

          {/* Standby / Empty indicator */}
          {(!isLivePrediction || histPoints.length === 0) && (
            <text 
              x={(padL + W - padR) / 2} 
              y={padT + chartH / 2} 
              textAnchor="middle" 
              fill="var(--text-muted)" 
              fontSize="12" 
              fontFamily="monospace"
            >
              {isMlActive ? '[AWAITING PREDICTION STREAM]' : '[ML STANDBY — Threat trajectory inactive]'}
            </text>
          )}

          {/* 1. Historical Actual Telemetry Area & Line */}
          {isLivePrediction && actualPathD && lastHistPt && (
            <>
              <path 
                d={`${actualPathD} L ${lastHistPt.x} ${getY(0)} L ${padL} ${getY(0)} Z`} 
                fill="url(#histAreaGrad)" 
              />
              <path 
                d={actualPathD} 
                fill="none" 
                stroke="var(--color-cyan)" 
                strokeWidth="2.5" 
              />
            </>
          )}

          {/* 2. Forecast Cone / Area Fill under future segment */}
          {futureForecastPathD && forecastPoints.length > 0 && lastHistPt && (
            <path 
              d={`${futureForecastPathD} L ${forecastPoints[forecastPoints.length - 1].x} ${getY(0)} L ${lastHistPt.x} ${getY(0)} Z`} 
              fill="url(#forecastAreaGrad)" 
            />
          )}

          {/* 3. Model Predicted Trajectory (Spanning both Historical Predictions & Future Forecast) */}
          {isLivePrediction && modelForecastPathD && (
            <path 
              d={modelForecastPathD} 
              fill="none" 
              stroke="var(--color-purple)" 
              strokeWidth="2.2" 
              strokeDasharray="5 3" 
            />
          )}

          {/* 4. Vertical Stems Connecting Actual vs Predicted for each historical event */}
          {histPoints.map((pt, i) => {
            if (Math.abs(pt.delta) < 0.008) return null;
            return (
              <line 
                key={`stem-${i}`}
                x1={pt.x}
                y1={pt.yActual}
                x2={pt.x}
                y2={pt.yPredicted}
                stroke="rgba(179, 102, 255, 0.45)"
                strokeDasharray="2 2"
                strokeWidth="1"
              />
            );
          })}

          {/* 5. Historical Actual Data Dots */}
          {histPoints.map((pt, i) => (
            <circle 
              key={`actual-dot-${i}`} 
              cx={pt.x} 
              cy={pt.yActual} 
              r={pt.isAlert ? 4 : 2.5} 
              fill={pt.isAlert ? 'var(--color-red)' : 'var(--color-cyan)'} 
            />
          ))}

          {/* 6. Historical Predicted Data Dots (What the model had predicted for this event) */}
          {histPoints.map((pt, i) => (
            <circle 
              key={`pred-dot-${i}`} 
              cx={pt.x} 
              cy={pt.yPredicted} 
              r="2.5" 
              fill="var(--color-purple)" 
              stroke="#ffffff" 
              strokeWidth="0.8" 
            />
          ))}

          {/* 7. Future Forecast Dots */}
          {forecastPoints.map((pt, i) => (
            <circle 
              key={`fc-dot-${i}`} 
              cx={pt.x} 
              cy={pt.y} 
              r="3.5" 
              fill="var(--color-purple)" 
              stroke="#fff" 
              strokeWidth="1"
            />
          ))}

          {/* 8. Invisible hitboxes for interactive hover inspection */}
          {histPoints.map((pt, i) => (
            <rect
              key={`hover-hitbox-${i}`}
              x={pt.x - 10}
              y={padT}
              width="20"
              height={chartH}
              fill="transparent"
              style={{ cursor: 'pointer' }}
              onMouseEnter={() => setHoveredPoint(pt)}
              onMouseLeave={() => setHoveredPoint(null)}
            />
          ))}

          {/* 9. Alert Pin Callout */}
          {alertPt && (
            <g className="alert-callout-group">
              <line x1={alertPt.x} y1={alertPt.yActual} x2={alertPt.x} y2={padT - 5} stroke="var(--color-red)" strokeWidth="1.5" />
              <circle cx={alertPt.x} cy={alertPt.yActual} r="5" fill="var(--color-red)" />
              <rect x={alertPt.x - 45} y={padT - 22} width="90" height="18" rx="4" fill="rgba(255, 45, 85, 0.9)" />
              <text x={alertPt.x} y={padT - 10} textAnchor="middle" fill="#fff" className="callout-text">MODEL ALERT</text>
            </g>
          )}

          {/* 10. Actual Attack Milestone Marker */}
          {earlyWarning?.actual_milestone_timestamp && (
            <g className="actual-milestone-group">
              <line x1={W - padR - 20} y1={padT} x2={W - padR - 20} y2={getY(0)} stroke="var(--color-amber)" strokeWidth="1.5" strokeDasharray="3 3" />
              <rect x={W - padR - 65} y={padT - 22} width="90" height="18" rx="4" fill="rgba(255, 170, 0, 0.9)" />
              <text x={W - padR - 20} y={padT - 10} textAnchor="middle" fill="#000" className="callout-text font-bold">ACTUAL PIVOT</text>
            </g>
          )}

          {/* 11. Early Warning Lead Time Delta Bracket */}
          {earlyWarning?.lead_time_seconds && alertPt && (
            <g className="lead-time-bracket">
              <line x1={alertPt.x} y1={H - padB + 14} x2={W - padR - 20} y2={H - padB + 14} stroke="var(--color-green)" strokeWidth="2" />
              <text x={(alertPt.x + W - padR - 20) / 2} y={H - padB + 26} textAnchor="middle" fill="var(--color-green)" className="font-mono text-xs font-bold">
                ▲ EARLY WARNING: +{earlyWarning.lead_time_seconds}s ADVANCE NOTICE
              </text>
            </g>
          )}

          {/* 12. Interactive HUD Tooltip on Hover */}
          {hoveredPoint && (
            <g className="trajectory-hud-tooltip font-mono">
              <rect 
                x={Math.min(W - 190, Math.max(padL, hoveredPoint.x - 85))} 
                y={padT + 8} 
                width="170" 
                height="62" 
                rx="5" 
                fill="rgba(10, 15, 26, 0.94)" 
                stroke="var(--color-cyan)" 
                strokeWidth="1" 
              />
              <text x={Math.min(W - 190, Math.max(padL, hoveredPoint.x - 85)) + 10} y={padT + 22} fill="#fff" fontSize="9" fontWeight="bold">
                WINDOW #{hoveredPoint.windowId} ({hoveredPoint.wallClock || 'LIVE'})
              </text>
              <text x={Math.min(W - 190, Math.max(padL, hoveredPoint.x - 85)) + 10} y={padT + 36} fill="var(--color-cyan)" fontSize="9">
                ACTUAL RISK: {hoveredPoint.actualRisk.toFixed(4)}
              </text>
              <text x={Math.min(W - 190, Math.max(padL, hoveredPoint.x - 85)) + 10} y={padT + 49} fill="var(--color-purple)" fontSize="9">
                PREDICTED:   {hoveredPoint.predictedRisk.toFixed(4)}
              </text>
              <text x={Math.min(W - 190, Math.max(padL, hoveredPoint.x - 85)) + 10} y={padT + 62} fill={Math.abs(hoveredPoint.delta) < 0.15 ? 'var(--color-green)' : 'var(--color-amber)'} fontSize="8">
                VARIANCE Δ:  {(hoveredPoint.delta > 0 ? '+' : '') + hoveredPoint.delta.toFixed(4)}
              </text>
            </g>
          )}
        </svg>
      </div>
    </div>
  );
}
