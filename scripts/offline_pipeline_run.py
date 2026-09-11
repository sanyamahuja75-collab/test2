#!/usr/bin/env python3
"""
scripts/offline_pipeline_run.py

Executes ONE COMPLETE INFERENCE CYCLE of the production chain

    PCAP -> 2s window -> flow table -> graph -> TGNE/BiTA
         -> Branch A -> Branch B -> DeepOP -> canonical prediction event

against the REAL production checkpoints, using the repository's own telemetry,
graph-construction and feature code, and a numpy reference implementation of the
four models (scripts/numpy_reference_runtime.py).

Why it exists: it runs with numpy only, so the whole chain can be executed and
inspected on a machine without PyTorch — CI, a fresh laptop, a locked-down box.
When torch IS available, scripts/smoke_pipeline.py runs the identical chain
through the actual runtime (control_backend.model_adapter) and is the
authoritative check; this script is the portable equivalent.

Usage:
    python scripts/offline_pipeline_run.py [--pcap captures/live.pcap] [--windows 12]
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from typing import Dict, List

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bita"))

import model_contract as MC                                            # noqa: E402
from scripts import numpy_reference_runtime as nrt                     # noqa: E402
from telemetry.capture.sniffer import StreamingPacketSniffer           # noqa: E402
from telemetry.flow.flow_table import LiveFlowTable                    # noqa: E402
from data_unification.unified_schema import UnifiedFlowRecord, CoarseCategory   # noqa: E402
from data_unification.flow_to_temporal_event import FlowToTemporalEventAdapter  # noqa: E402
from data_unification.host_attributes import compute_host_temporal_attributes   # noqa: E402
from branch_a_gnn_lstm.technique_vocab import TECHNIQUE_VOCAB          # noqa: E402
from deepop_decoder.joint_vocab import get_joint_vocab, consolidate_network_technique  # noqa: E402
# control_backend.schema needs pydantic (no torch). When pydantic is present the
# event built below is validated against the REAL wire schema; when it is not,
# the same dict is still built and checked structurally.
try:
    from control_backend.schema import PredictionEvent  # noqa: E402
    HAVE_SCHEMA = True
except ImportError:  # pragma: no cover - bare environment
    PredictionEvent = None
    HAVE_SCHEMA = False

TECHNIQUE_TO_TACTIC = {
    "Benign": "Benign", "T1046": "Recon", "T1595": "Recon", "T1110": "CredentialAccess",
    "T1190": "InitialAccess", "T1189": "InitialAccess", "T1071": "C2", "T1071.001": "C2",
    "T1568.001": "C2", "T1204": "Execution", "T1005": "Collection", "T1498": "Impact",
    "T1498.001": "Impact", "T1020": "Exfiltration",
}


# ---------------------------------------------------------------------------
# Telemetry: PCAP -> packets (uses the repo's own frame parser)
# ---------------------------------------------------------------------------

def read_pcap(path: str):
    with open(path, "rb") as fh:
        hdr = fh.read(24)
        if len(hdr) < 24:
            return
        magic = struct.unpack("=I", hdr[:4])[0]
        endian = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
        nano = magic in (0xA1B23C4D, 0x4D3CB2A1)
        while True:
            rec = fh.read(16)
            if len(rec) < 16:
                return
            sec, frac, incl, _orig = struct.unpack(endian + "IIII", rec)
            data = fh.read(incl)
            if len(data) < incl:
                return
            ts = sec + (frac / 1e9 if nano else frac / 1e6)
            pkt = StreamingPacketSniffer.parse_frame(data, ts)
            if pkt:
                yield pkt


def pcap_windows(path: str, window_sec: float):
    """Group packets into fixed 2s windows and export 5-tuple flow snapshots,
    exactly the way telemetry/state/state_builder.py does on the live path."""
    table = LiveFlowTable()
    win_start = None
    windows = []
    n_pkts = 0
    for pkt in read_pcap(path):
        ts = pkt["timestamp"]
        if win_start is None:
            win_start = ts
        while ts - win_start >= window_sec:
            snapshot = table.snapshot_flows(max_flows=256)
            table.extract_window_features(window_sec)   # mirrors live pruning/reset
            windows.append({
                "window_start": win_start,
                "window_end": win_start + window_sec,
                "packet_count": n_pkts,
                "flows": snapshot,
            })
            win_start += window_sec
            n_pkts = 0
        table.process_packet(pkt)
        n_pkts += 1
    if win_start is not None:
        snapshot = table.snapshot_flows(max_flows=256)
        windows.append({
            "window_start": win_start,
            "window_end": win_start + window_sec,
            "packet_count": n_pkts,
            "flows": snapshot,
        })
    return windows


def to_records(raw_flows) -> List[UnifiedFlowRecord]:
    out = []
    for f in raw_flows:
        out.append(UnifiedFlowRecord(
            src_ip=str(f["src_ip"]), dst_ip=str(f["dst_ip"]),
            src_port=int(f["src_port"]), dst_port=int(f["dst_port"]),
            protocol=int(f["protocol"]),
            start_time=float(f["start_time"]), end_time=float(f["end_time"]),
            fwd_bytes=int(f["fwd_bytes"]), bwd_bytes=int(f["bwd_bytes"]),
            fwd_packets=int(f["fwd_packets"]), bwd_packets=int(f["bwd_packets"]),
            raw_label="UNLABELED", raw_label_source="PCAP_REPLAY", is_attack=False,
            coarse_category=CoarseCategory.UNKNOWN.value, attck_technique_ids=[],
            metadata={"source": "offline_pipeline_run"},
        ))
    return out


def pick_target(records) -> str:
    activity: Dict[str, int] = {}
    for r in records:
        tot = r.fwd_bytes + r.bwd_bytes + r.fwd_packets + r.bwd_packets
        activity[r.src_ip] = activity.get(r.src_ip, 0) + tot
        activity[r.dst_ip] = activity.get(r.dst_ip, 0) + tot
    return max(activity.items(), key=lambda kv: kv[1])[0] if activity else ""


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", default=os.path.join(ROOT, "captures", "live.pcap"))
    ap.add_argument("--windows", type=int, default=12, help="max windows to process")
    ap.add_argument("--json-out", default=None, help="write the last prediction event here")
    args = ap.parse_args()

    print("=" * 78)
    print("CYBERWORLD OFFLINE END-TO-END PIPELINE RUN (numpy reference, real checkpoints)")
    print("=" * 78)
    print(MC.summary())
    MC.validate_temporal_contract_file()

    # ---- load the real checkpoints ------------------------------------------
    tgne_sd = nrt.load_state(MC.TGNE_CHECKPOINT)
    ba_sd = nrt.load_state(MC.BRANCH_A_CHECKPOINT)[MC.BRANCH_A_STATE_KEY]
    bb = nrt.load_state(MC.BRANCH_B_CHECKPOINT)
    dp = nrt.load_state(MC.DEEPOP_CHECKPOINT)

    vocab = get_joint_vocab()
    if vocab.vocab_size != MC.DEEPOP_VOCAB_SIZE:
        raise MC.ContractViolation(
            f"joint vocab={vocab.vocab_size} contract={MC.DEEPOP_VOCAB_SIZE}")

    tgne = nrt.TGNE(tgne_sd, MC.TGNE_LATENT_DIM, MC.TGNE_N_HEADS)
    branch_a = nrt.BranchA(ba_sd, MC.BRANCH_A_HIDDEN_DIM, MC.BRANCH_A_NUM_LAYERS)
    branch_b = nrt.BranchB(bb[MC.BRANCH_B_WDT_KEY], bb[MC.BRANCH_B_RISK_KEY],
                           MC.BRANCH_B_D_MODEL, MC.BRANCH_B_N_HEADS, MC.BRANCH_B_N_LAYERS)
    deepop = nrt.DeepOP(dp[MC.DEEPOP_STATE_KEY], vocab.tokens, MC.DEEPOP_D_MODEL,
                        MC.DEEPOP_N_HEADS, MC.DEEPOP_N_LAYERS, MC.DEEPOP_WINDOW_SIZES,
                        MC.DEEPOP_D_LATENT)
    print(f"[TGNE] loaded   input_dim(edge)={MC.TGNE_EDGE_FEAT_DIM} latent_dim={MC.TGNE_LATENT_DIM} "
          f"categories={MC.TGNE_NUM_CATEGORIES}")
    print(f"[BRANCH_A] loaded input_dim={MC.BRANCH_A_INPUT_DIM} hidden={MC.BRANCH_A_HIDDEN_DIM} "
          f"techniques={MC.BRANCH_A_NUM_TECHNIQUES}")
    print(f"[BRANCH_B] loaded d_latent={MC.BRANCH_B_D_LATENT} d_model={MC.BRANCH_B_D_MODEL} "
          f"layers={MC.BRANCH_B_N_LAYERS} K={MC.FORECAST_STEPS}")
    print(f"[DEEPOP] loaded  d_model={MC.DEEPOP_D_MODEL} heads={MC.DEEPOP_N_HEADS} "
          f"vocab={vocab.vocab_size} -> {vocab.tokens}")

    # ---- telemetry -----------------------------------------------------------
    if not os.path.exists(args.pcap):
        print(f"[-] capture not found: {args.pcap}")
        return 2
    windows = pcap_windows(args.pcap, MC.WINDOW_SECONDS)
    windows = [w for w in windows if w["flows"]][: args.windows]
    print(f"\n[TELEMETRY] {os.path.relpath(args.pcap, ROOT)} -> {len(windows)} non-empty "
          f"{MC.WINDOW_SECONDS}s windows")

    feature_history: List[np.ndarray] = []
    latent_history: List[np.ndarray] = []
    last_event = None
    benign_idx = vocab.encode("Benign", None)

    for w in windows:
        t0 = time.perf_counter()
        records = to_records(w["flows"])
        target = pick_target(records)
        host_flows = [r for r in records if r.src_ip == target or r.dst_ip == target]

        # ---- STAGE 1: graph + TGNE ------------------------------------------
        adapter = FlowToTemporalEventAdapter(window_size_sec=MC.WINDOW_SECONDS)
        stream = adapter.process_records(host_flows, sort_by_time=True)
        MC.assert_last_dim("edge_features", stream.edge_features.shape, MC.TGNE_EDGE_FEAT_DIM)

        n_nodes = stream.n_nodes + 10
        adj = [[] for _ in range(n_nodes)]
        for i in range(len(stream.sources)):
            u, v, t = int(stream.sources[i]), int(stream.destinations[i]), float(stream.timestamps[i])
            adj[u].append((v, i, t))
            adj[v].append((u, i, t))
        nf = nrt.NeighborFinder(adj)

        host_ids = np.array([adapter.ip_to_id.get(target, 0)], dtype=np.int64)
        H_t = tgne.host_embeddings(host_ids, w["window_end"], nf, stream.edge_features,
                                   MC.TGNE_NODE_FEAT_DIM, MC.TGNE_N_NEIGHBORS)
        MC.assert_shape("TGNE H_t", H_t.shape, (1, MC.TGNE_LATENT_DIM))
        h_emb = H_t[0]

        attrs = compute_host_temporal_attributes(target, host_flows, MC.WINDOW_SECONDS)
        MC.assert_last_dim("host_attrs", attrs.shape, MC.HOST_TEMPORAL_ATTR_DIM)

        feat = np.concatenate([h_emb, attrs]).astype(np.float32)
        MC.assert_last_dim("branch_a_feature_vector", feat.shape, MC.BRANCH_A_INPUT_DIM)

        feature_history.append(feat)
        latent_history.append(h_emb.astype(np.float32))
        feature_history = feature_history[-MC.HISTORY_STEPS:]
        latent_history = latent_history[-MC.HISTORY_STEPS:]

        fh = list(feature_history)
        lh = list(latent_history)
        while len(fh) < MC.HISTORY_STEPS:
            fh.insert(0, np.zeros_like(fh[0]))
            lh.insert(0, np.zeros_like(lh[0]))

        # ---- STAGE 2: Branch A ----------------------------------------------
        x = np.stack(fh)[None, ...]
        MC.assert_shape("branch_a_input", x.shape, (1, MC.HISTORY_STEPS, MC.BRANCH_A_INPUT_DIM))
        a_out = branch_a(x)
        MC.assert_shape("branch_a_technique_logits", a_out["technique_logits"].shape,
                        (1, MC.BRANCH_A_NUM_TECHNIQUES))
        probs = nrt.softmax(a_out["technique_logits"], axis=-1)[0]
        tech_idx = int(probs.argmax())
        technique = TECHNIQUE_VOCAB[tech_idx]
        risk = float(a_out["risk_score"][0])
        gradation = int(a_out["gradation_logits"][0].argmax())

        # ---- STAGE 3: Branch B ----------------------------------------------
        h_seq = np.stack(lh)[None, ...]
        MC.assert_shape("branch_b_input", h_seq.shape, (1, MC.HISTORY_STEPS, MC.BRANCH_B_D_LATENT))
        h_future = branch_b.rollout(h_seq, K=MC.FORECAST_STEPS, delta_t=MC.WINDOW_SECONDS)
        MC.assert_shape("branch_b_future_latent", h_future.shape,
                        (1, MC.FORECAST_STEPS, MC.BRANCH_B_D_LATENT))
        fut_risks = branch_b.step_risks(h_future)[0]
        MC.assert_shape("branch_b_step_risks", (1,) + fut_risks.shape, (1, MC.FORECAST_STEPS))

        # ---- STAGE 4: DeepOP -------------------------------------------------
        MC.assert_last_dim("deepop_memory_input", h_future.shape, MC.DEEPOP_D_LATENT)
        coarse, ctech = consolidate_network_technique(
            TECHNIQUE_TO_TACTIC.get(technique, "Benign"), technique)
        obs_token = vocab.encode(coarse, ctech)
        tokens, names, conf = deepop.forecast_sequence(
            h_future, max_steps=MC.FORECAST_STEPS,
            observed_token=np.array([obs_token]), benign_idx=benign_idx)
        MC.assert_shape("deepop_tokens", tokens.shape, (1, MC.FORECAST_STEPS))

        inf_ms = (time.perf_counter() - t0) * 1000.0
        max_future = float(np.max(fut_risks))

        print()
        print(f"--- WINDOW @ {w['window_end']:.2f}  host={target}  "
              f"({len(host_flows)} flows / {w['packet_count']} pkts) ---")
        print(f"[TELEMETRY] window ready        flows={len(host_flows)} "
              f"nodes={len(set([r.src_ip for r in host_flows] + [r.dst_ip for r in host_flows]))} "
              f"edges={len(set((r.src_ip, r.dst_ip) for r in host_flows))}")
        print(f"[TGNE] embedding generated     shape={H_t.shape} |H|={np.linalg.norm(h_emb):.4f}")
        print(f"[BRANCH_A] risk={risk:.4f}")
        print(f"[BRANCH_A] technique={technique} (p={float(probs[tech_idx]):.4f}) gradation={gradation}")
        print(f"[BRANCH_B] future_latent_shape={h_future.shape} "
              f"risk_trajectory={[round(float(r), 4) for r in fut_risks]}")
        print(f"[DEEPOP] observed_token={coarse}.{ctech} -> future_technique={names[0]}")
        print(f"[DEEPOP] confidence={[round(c, 4) for c in conf[0]]}")
        print(f"[INFERENCE] prediction event created in {inf_ms:.1f} ms")

        # Canonical prediction event — exactly the payload control_backend emits.
        last_event = {
            "type": "prediction",
            "mode": "LIVE",
            "timestamp": str(w["window_end"]),
            "wall_clock": time.strftime("%H:%M:%S", time.localtime(w["window_end"])),
            "model": {
                "name": MC.MODEL_NAME, "version": MC.MODEL_VERSION,
                "feature_count": MC.BRANCH_A_INPUT_DIM, "history_steps": MC.HISTORY_STEPS,
                "window_seconds": MC.WINDOW_SECONDS, "forecast_steps": MC.FORECAST_STEPS,
                "checkpoint": MC.CHECKPOINT_LABEL, "threshold": MC.ALERT_THRESHOLD,
            },
            "state": {
                "window_id": int(w["window_end"]) % 100000,
                "sequence_ready": len(latent_history) >= MC.HISTORY_STEPS,
                "packet_count": int(w["packet_count"]), "active_flows": len(host_flows),
                "pipeline_latency_ms": 0.0, "buffer_length": len(latent_history),
            },
            "prediction": {
                "risk": round(risk, 4), "max_future_risk": round(max_future, 4),
                "hazard_score": round(max_future, 4),
                "malicious_confidence": round(risk, 4),
                "precursor_confidence": round(max(0.0, max_future - risk), 4),
                "alert": bool(risk >= MC.ALERT_THRESHOLD or max_future >= MC.ALERT_THRESHOLD),
                "alert_level": ("CRITICAL" if max(risk, max_future) >= MC.ALERT_THRESHOLD + 0.2
                                else "ELEVATED" if max(risk, max_future) >= MC.ALERT_THRESHOLD
                                else "WARNING" if max(risk, max_future) >= MC.ALERT_THRESHOLD * 0.75
                                else "NOMINAL"),
                "threshold": MC.ALERT_THRESHOLD, "predicted_stage": technique,
                "stage_probabilities": {
                    TECHNIQUE_VOCAB[i]: round(float(probs[i]), 4)
                    for i in range(len(TECHNIQUE_VOCAB)) if float(probs[i]) > 0.01
                },
            },
            "forecast": [
                {"horizon_seconds": (i + 1) * MC.WINDOW_SECONDS,
                 "risk": round(float(fut_risks[i]), 4),
                 "confidence": None,
                 "predicted_stage": names[0][i],
                 "technique_confidence": round(conf[0][i], 4)}
                for i in range(MC.FORECAST_STEPS)
            ],
            "stages": {
                "tgne": {
                    "model": "TGNE-TA (ExtendedTGN / BiTA)",
                    "embedding_dim": int(h_emb.shape[-1]),
                    "embedding_norm": round(float(np.linalg.norm(h_emb)), 4),
                    "graph_nodes": len({ip for r in host_flows for ip in (r.src_ip, r.dst_ip)}),
                    "graph_edges": len({(r.src_ip, r.dst_ip) for r in host_flows}),
                    "n_neighbors": MC.TGNE_N_NEIGHBORS,
                },
                "branch_a": {
                    "model": "MultiTaskLSTM",
                    "input_dim": MC.BRANCH_A_INPUT_DIM, "history_steps": MC.HISTORY_STEPS,
                    "risk": round(risk, 4), "technique": technique,
                    "confidence": round(float(probs[tech_idx]), 4), "gradation": gradation,
                    "target_host": target,
                },
                "branch_b": {
                    "model": "HostWorldDynamicsTransformer",
                    "latent_dim": MC.BRANCH_B_D_LATENT,
                    "forecast_steps": MC.FORECAST_STEPS,
                    "forecast_horizon_seconds": MC.FORECAST_HORIZON_SECONDS,
                    "risk_trajectory": [round(float(r), 4) for r in fut_risks],
                    "latent_trajectory_norms": [round(float(v), 4)
                                                for v in np.linalg.norm(h_future[0], axis=-1)],
                    "max_future_risk": round(max_future, 4),
                },
                "deepop": {
                    "model": "DeepOPForecastDecoder (CWA)",
                    "latent_dim": MC.DEEPOP_D_LATENT, "vocab_size": vocab.vocab_size,
                    "observed_token": f"{coarse}.{ctech}",
                    "technique_trajectory": list(names[0]),
                    "confidence": [round(c, 4) for c in conf[0]],
                },
            },
            "explainability": {"available": False, "method": "n/a (offline reference run)",
                               "groups": [], "top_features": []},
            "latency": {"telemetry_ms": 0.0, "inference_ms": round(inf_ms, 2),
                        "total_ms": round(inf_ms, 2)},
            "early_warning": None,
            "attack_active": False,
            "attack_phase": None,
            "target_ip": target,
            "focus_ips": sorted({ip for r in host_flows for ip in (r.src_ip, r.dst_ip)})[:12],
            "focus_edges": [],
        }

    if last_event is None:
        print("\n[-] No non-empty windows in the capture.")
        return 3

    wire = last_event
    if HAVE_SCHEMA:
        # Validate the event against the real backend wire contract.
        validated = PredictionEvent.model_validate(wire)
        wire = validated.model_dump()
        print("\n[SCHEMA] event validates against control_backend.schema.PredictionEvent")
    else:
        print("\n[SCHEMA] pydantic not installed — structural checks only "
              "(install pydantic to validate against control_backend.schema)")

    print("\n" + "=" * 78)
    print("CANONICAL PREDICTION EVENT (last window) — one coherent pass through the chain")
    print("=" * 78)
    print(json.dumps(wire, indent=2, default=str)[:2600])

    # ---- backend event -> WebSocket payload -> frontend expected schema -------
    checks = [
        ("type == prediction", wire["type"] == "prediction"),
        ("model.threshold", "threshold" in wire["model"]),
        ("model.window_seconds == 2.0", wire["model"]["window_seconds"] == MC.WINDOW_SECONDS),
        ("model.forecast_steps == 8", wire["model"]["forecast_steps"] == MC.FORECAST_STEPS),
        ("state.window_id", "window_id" in wire["state"]),
        ("prediction.risk", isinstance(wire["prediction"]["risk"], float)),
        ("prediction.max_future_risk", isinstance(wire["prediction"]["max_future_risk"], float)),
        ("prediction.alert / alert_level", isinstance(wire["prediction"]["alert"], bool)
            and isinstance(wire["prediction"]["alert_level"], str)),
        ("prediction.predicted_stage", bool(wire["prediction"]["predicted_stage"])),
        ("forecast length == 8", len(wire["forecast"]) == MC.FORECAST_STEPS),
        ("forecast horizon == 16s",
            wire["forecast"][-1]["horizon_seconds"] == MC.FORECAST_HORIZON_SECONDS),
        ("forecast[].predicted_stage (DeepOP)", all(f["predicted_stage"] for f in wire["forecast"])),
        ("forecast[].technique_confidence (DeepOP)",
            all(f["technique_confidence"] is not None for f in wire["forecast"])),
        ("stages.tgne.embedding_dim == 12", wire["stages"]["tgne"]["embedding_dim"] == MC.TGNE_LATENT_DIM),
        ("stages.branch_a.input_dim == 27", wire["stages"]["branch_a"]["input_dim"] == MC.BRANCH_A_INPUT_DIM),
        ("stages.branch_b.risk_trajectory == 8",
            len(wire["stages"]["branch_b"]["risk_trajectory"]) == MC.FORECAST_STEPS),
        ("stages.deepop.vocab_size == 10", wire["stages"]["deepop"]["vocab_size"] == MC.DEEPOP_VOCAB_SIZE),
        ("stages.deepop.technique_trajectory == 8",
            len(wire["stages"]["deepop"]["technique_trajectory"]) == MC.FORECAST_STEPS),
    ]
    traceability = [
        ("prediction.risk  <-  Branch A",
            wire["prediction"]["risk"] == wire["stages"]["branch_a"]["risk"]),
        ("prediction.predicted_stage  <-  Branch A",
            wire["prediction"]["predicted_stage"] == wire["stages"]["branch_a"]["technique"]),
        ("prediction.max_future_risk  <-  max(Branch B trajectory)",
            abs(wire["prediction"]["max_future_risk"]
                - max(wire["stages"]["branch_b"]["risk_trajectory"])) < 1e-6),
        ("forecast[].risk  <-  Branch B trajectory",
            [f["risk"] for f in wire["forecast"]] == wire["stages"]["branch_b"]["risk_trajectory"]),
        ("forecast[].predicted_stage  <-  DeepOP trajectory",
            [f["predicted_stage"] for f in wire["forecast"]]
            == wire["stages"]["deepop"]["technique_trajectory"]),
    ]

    print("\n[DASHBOARD] backend event -> WebSocket payload -> frontend expected schema")
    for name, ok in checks:
        print(f"    {'OK  ' if ok else 'FAIL'} {name}")
    print("\n[TRACEABILITY] every dashboard value traced back to its producing stage")
    for name, ok in traceability:
        print(f"    {'OK  ' if ok else 'FAIL'} {name}")

    failed = [n for n, ok in checks + traceability if not ok]
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(wire, fh, indent=2, default=str)
        print(f"\n[+] written to {args.json_out}")
    if failed:
        print(f"\n[FAIL] {failed}")
        return 1
    print("\n[OK] Full chain executed end-to-end on real checkpoints; event contract verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
