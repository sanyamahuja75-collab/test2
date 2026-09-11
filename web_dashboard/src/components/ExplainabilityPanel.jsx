import React, { useState } from 'react';
import { HelpCircle, ChevronDown, ChevronUp, Layers, Info } from 'lucide-react';

export default function ExplainabilityPanel({ latestPrediction, isMlActive }) {
  const [showDetails, setShowDetails] = useState(false);
  const expl = latestPrediction?.explainability;

  const isAvailable = Boolean(isMlActive && expl?.available);
  const groups = isAvailable ? (expl?.groups || []) : [];
  const topFeatures = isAvailable ? (expl?.top_features || []) : [];
  const method = isAvailable ? (expl?.method || null) : null;

  return (
    <div className="soc-card explainability-card">
      <div className="soc-card-header">
        <div className="soc-card-title">
          <HelpCircle size={16} color="var(--color-cyan)" />
          FEATURE ATTRIBUTION &amp; EXPLAINABILITY
        </div>
        <span className="font-mono text-xs text-muted">
          {isAvailable ? 'ACTIVE' : 'STANDBY'}
        </span>
      </div>

      <div className="explainability-body">
        {!isAvailable ? (
          <div className="expl-empty-state">
            <Info size={24} color="var(--text-muted)" />
            <p className="text-muted font-mono text-xs">
              {isMlActive 
                ? "Explainability attribution is calculated when causal sequence inference is active."
                : "[ML STANDBY: Feature attribution inactive]"}
            </p>
          </div>
        ) : (
          <>
            {method && (
              <div className="expl-method-banner font-mono text-xs">
                <span>METHOD:</span> <strong className="text-cyan">{method}</strong>
              </div>
            )}

            {/* Feature Group Contribution Bars */}
            <div className="expl-groups-list">
              {groups.map((grp) => {
                let barColor = 'var(--color-cyan)';
                if (grp.percentage > 25) barColor = 'var(--color-red)';
                else if (grp.percentage > 15) barColor = 'var(--color-amber)';

                return (
                  <div key={grp.name} className="expl-group-row">
                    <div className="expl-group-labels">
                      <span className="expl-group-name">{grp.name}</span>
                      <span className="expl-group-pct font-mono">{grp.percentage.toFixed(1)}%</span>
                    </div>
                    <div className="expl-bar-track">
                      <div 
                        className="expl-bar-fill" 
                        style={{ 
                          width: `${Math.min(100, Math.max(0, grp.percentage))}%`,
                          backgroundColor: barColor
                        }} 
                      />
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Top Individual Drivers Drilldown */}
            <div className="expl-drilldown-toggle">
              <button 
                className="btn btn-subtle text-xs" 
                onClick={() => setShowDetails(!showDetails)}
              >
                <Layers size={13} />
                {showDetails ? 'Hide Top Feature Drivers' : `View Top Individual Features (${topFeatures.length})`}
                {showDetails ? <ChevronUp size={13} /> : <ChevronDown size={13} />}
              </button>
            </div>

            {showDetails && topFeatures.length > 0 && (
              <div className="expl-top-features-panel font-mono text-xs">
                {topFeatures.map((tf, i) => (
                  <div key={tf.feature} className="top-feature-item">
                    <span className="text-muted">#{i + 1}</span>
                    <span className="text-highlight font-bold">{tf.feature}</span>
                    <span className="text-cyan">({tf.group})</span>
                    <span className="text-amber ml-auto">Score: {tf.score.toFixed(3)}</span>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
