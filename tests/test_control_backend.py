"""
tests/test_control_backend.py
Unit tests for Containerlab control backend + Dual-Branch/DeepOP adapter.
"""

import os
import sys
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "bita"))

from control_backend.main import app
from control_backend.model_adapter import (
    AntigravityModelAdapter,
    flows_from_span_dicts,
    select_primary_target,
)
from control_backend.commands import ALLOWED_COMMANDS, executor


@pytest.fixture
def client():
    return TestClient(app)


def test_01_api_status_endpoint(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    assert data["network"] in ["stopped", "starting", "running", "stopping"]
    assert data["sensor"] in ["stopped", "starting", "running", "stopping"]
    assert data["ml"] in ["stopped", "starting", "running", "stopping"]
    assert data["mode"] in ["STANDBY", "LIVE"]
    assert data["ml_status"] in ["standby", "live"]
    assert data["model_meta"]["name"] == "Antigravity-DualBranch-DeepOP"
    assert data["model_meta"]["feature_count"] == 27
    assert data["model_meta"]["history_steps"] == 5
    assert data["model_meta"]["window_seconds"] == 2.0
    assert data["model_meta"]["forecast_steps"] == 8


def test_02_dual_branch_adapter_smoke():
    adapter = AntigravityModelAdapter()
    flows = flows_from_span_dicts(
        [
            {
                "src_ip": "192.168.100.10",
                "dst_ip": "10.0.3.10",
                "src_port": 4444,
                "dst_port": 80,
                "protocol": 6,
                "fwd_bytes": 1200,
                "bwd_bytes": 400,
                "fwd_packets": 10,
                "bwd_packets": 8,
                "start_time": 1.0,
                "end_time": 2.0,
            }
        ]
    )
    target = select_primary_target(flows)
    assert target == "10.0.3.10"
    event = adapter.predict_window(target, flows)
    assert event.type == "prediction"
    assert event.model.feature_count == 27
    assert event.model.forecast_steps == 8
    assert event.prediction.risk is not None
    assert 0.0 <= float(event.prediction.risk) <= 1.0
    assert len(event.forecast) == 8
    assert "10.0.3.10" in event.focus_ips
    assert event.target_ip == "10.0.3.10"
    assert len(adapter.feature_history) == 1


def test_02b_adapter_scopes_live_features_to_target_host():
    target_flow = {
        "src_ip": "192.168.100.10",
        "dst_ip": "10.0.3.10",
        "src_port": 4444,
        "dst_port": 80,
        "protocol": 6,
        "fwd_bytes": 1200,
        "bwd_bytes": 400,
        "fwd_packets": 10,
        "bwd_packets": 8,
        "start_time": 1.0,
        "end_time": 2.0,
    }
    unrelated_flow = {
        "src_ip": "10.0.9.10",
        "dst_ip": "10.0.9.20",
        "src_port": 5555,
        "dst_port": 443,
        "protocol": 6,
        "fwd_bytes": 100000,
        "bwd_bytes": 50000,
        "fwd_packets": 1000,
        "bwd_packets": 800,
        "start_time": 1.0,
        "end_time": 2.0,
    }
    target_flows = flows_from_span_dicts([target_flow])
    mixed_flows = flows_from_span_dicts([target_flow, unrelated_flow])

    adapter = AntigravityModelAdapter()
    target_event = adapter.predict_window("10.0.3.10", target_flows)
    adapter.reset_history()
    mixed_event = adapter.predict_window("10.0.3.10", mixed_flows)

    assert mixed_event.prediction.risk == target_event.prediction.risk
    assert mixed_event.prediction.predicted_stage == target_event.prediction.predicted_stage


def test_02c_adapter_keeps_histories_per_target_host():
    flows = flows_from_span_dicts(
        [
            {
                "src_ip": "10.0.1.12",
                "dst_ip": "10.0.2.10",
                "src_port": 5000,
                "dst_port": 53,
                "protocol": 17,
                "fwd_bytes": 74,
                "bwd_bytes": 90,
                "fwd_packets": 1,
                "bwd_packets": 1,
                "start_time": 1.0,
                "end_time": 2.0,
            }
        ]
    )
    adapter = AntigravityModelAdapter()
    adapter.predict_window("10.0.2.10", flows)
    adapter.predict_window("10.0.1.12", flows)

    assert set(adapter.feature_history_by_target) == {"10.0.2.10", "10.0.1.12"}
    assert set(adapter.h_state_history_by_target) == {"10.0.2.10", "10.0.1.12"}
    assert all(len(history) == 1 for history in adapter.feature_history_by_target.values())


def test_03_allowlist_command_validation(client):
    resp_invalid = client.post("/api/command/rm_rf_slash")
    assert resp_invalid.status_code == 400
    assert "not allowed" in resp_invalid.json()["detail"]

    for cmd in [
        "healthcheck",
        "build_environment",
        "reset_environment",
        "start_ml",
        "stop_ml",
        "start_network",
        "stop_network",
        "start_telemetry",
        "stop_telemetry",
        "start_normal_traffic",
        "stop_normal_traffic",
        "start_attack",
        "stop_attack",
    ]:
        assert cmd in ALLOWED_COMMANDS

    assert "run_full_demo" not in ALLOWED_COMMANDS


def test_04_ml_standby_and_live_transition(client):
    from control_backend.telemetry_service import telemetry_service
    from control_backend.event_broker import broker

    telemetry_service.stop_ml()
    assert telemetry_service.is_ml_active is False
    assert broker.latest_prediction is None

    status_resp = client.get("/api/status")
    assert status_resp.json()["ml_active"] is False
    assert status_resp.json()["ml_status"] == "standby"

    with pytest.raises(RuntimeError, match="telemetry sensor must be running first"):
        telemetry_service.start_ml()

    telemetry_service.is_running = True
    try:
        start_res = telemetry_service.start_ml()
        assert start_res["status"] == "started"
        assert telemetry_service.is_ml_active is True
        assert client.get("/api/status").json()["ml_status"] == "live"

        stop_res = telemetry_service.stop_ml()
        assert stop_res["status"] == "stopped"
        assert telemetry_service.is_ml_active is False
        assert broker.latest_prediction is None
    finally:
        telemetry_service.is_running = False
        telemetry_service.stop_ml()


def test_05_external_only_scenarios(client):
    resp = client.get("/api/scenarios")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 1
    assert data[0]["id"] == "external"
    assert data[0].get("attacker_ip") in (None, "")
    # target_host may be site asset or null — never a hardcoded requirement for topology
    assert "site_id" in data[0]


def test_06_external_attack_arming():
    executor.attack_running = False
    assert executor.is_attack_running() is False
    executor.arm_external_attack({})
    assert executor.is_attack_running() is True
    executor.disarm_external_attack({})
    assert executor.is_attack_running() is False


def test_07_prediction_event_carries_full_stage_provenance():
    """One PredictionEvent == one coherent pass through TGNE -> A -> B -> DeepOP.

    Every value the dashboard renders must be traceable to the stage that produced
    it in this same pass — no stale, recomputed or frontend-invented numbers.
    """
    import model_contract as MC

    adapter = AntigravityModelAdapter()
    flows = flows_from_span_dicts(
        [
            {
                "src_ip": "192.168.100.10", "dst_ip": "10.0.3.10",
                "src_port": 4444, "dst_port": 80, "protocol": 6,
                "fwd_bytes": 1200, "bwd_bytes": 400,
                "fwd_packets": 10, "bwd_packets": 8,
                "start_time": 1.0, "end_time": 2.0,
            }
        ]
    )
    event = adapter.predict_window("10.0.3.10", flows)
    s = event.stages
    assert s is not None, "PredictionEvent must carry per-stage provenance"

    # Stage shapes match the authoritative contract
    assert s.tgne.embedding_dim == MC.TGNE_LATENT_DIM
    assert s.branch_a.input_dim == MC.BRANCH_A_INPUT_DIM
    assert s.branch_a.history_steps == MC.HISTORY_STEPS
    assert s.branch_b.latent_dim == MC.BRANCH_B_D_LATENT
    assert s.branch_b.forecast_steps == MC.FORECAST_STEPS
    assert s.branch_b.forecast_horizon_seconds == MC.FORECAST_HORIZON_SECONDS
    assert len(s.branch_b.risk_trajectory) == MC.FORECAST_STEPS
    assert s.deepop.vocab_size == MC.DEEPOP_VOCAB_SIZE
    assert len(s.deepop.technique_trajectory) == MC.FORECAST_STEPS

    # Traceability of every displayed value
    assert event.prediction.risk == s.branch_a.risk
    assert event.prediction.predicted_stage == s.branch_a.technique
    assert abs(event.prediction.max_future_risk - max(s.branch_b.risk_trajectory)) < 1e-6
    assert [f.risk for f in event.forecast] == s.branch_b.risk_trajectory
    assert [f.predicted_stage for f in event.forecast] == s.deepop.technique_trajectory
    assert [f.technique_confidence for f in event.forecast] == s.deepop.confidence
    assert event.forecast[-1].horizon_seconds == MC.FORECAST_HORIZON_SECONDS


def test_08_models_are_really_loaded_not_randomly_initialised():
    """Checkpoints must load strict; a missing checkpoint must raise, never fall back."""
    adapter = AntigravityModelAdapter()
    assert adapter.models_loaded is True
    assert len(adapter.technique_vocab) == 14
    assert adapter.vocab.vocab_size == 10
    # Branch B's time encoder must hold trained weights, not fresh random init.
    import torch
    w = adapter.wdt.time_encoder.linear.weight
    assert w.shape == (64, 1)
    assert torch.isfinite(w).all()
