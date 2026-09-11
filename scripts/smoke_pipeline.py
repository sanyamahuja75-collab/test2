#!/usr/bin/env python3
"""
scripts/smoke_pipeline.py

THE end-to-end smoke test. Drives the ACTUAL production runtime
(control_backend.model_adapter.AntigravityModelAdapter) over real telemetry and
prints one complete inference cycle through

    Telemetry -> 2s window -> Graph -> TGNE/BiTA -> Branch A -> Branch B
              -> DeepOP -> canonical prediction event -> WebSocket payload

with the real production checkpoints. Requires torch.

    python scripts/smoke_pipeline.py                          # replay captures/live.pcap
    python scripts/smoke_pipeline.py --pcap /path/to/x.pcap
    python scripts/smoke_pipeline.py --state-stream /tmp/cyberworld_live_stream.jsonl

Exit 0 means: every checkpoint loaded strict, every interface assertion held,
a real PredictionEvent was produced, and its JSON matches the schema the React
dashboard consumes.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import struct
import sys
from typing import Any, Dict, List

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "bita"))

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
logging.getLogger("antigravity.model_adapter").setLevel(logging.DEBUG)

import model_contract as MC                                        # noqa: E402


def windows_from_pcap(path: str) -> List[Dict[str, Any]]:
    """Replay a capture through the repo's own parser + flow table into 2s windows."""
    from telemetry.capture.sniffer import StreamingPacketSniffer
    from telemetry.flow.flow_table import LiveFlowTable

    def packets():
        with open(path, "rb") as fh:
            hdr = fh.read(24)
            magic = struct.unpack("=I", hdr[:4])[0]
            endian = "<" if magic in (0xA1B2C3D4, 0xA1B23C4D) else ">"
            nanos = magic in (0xA1B23C4D, 0x4D3CB2A1)
            while True:
                rec = fh.read(16)
                if len(rec) < 16:
                    return
                sec, frac, incl, _ = struct.unpack(endian + "IIII", rec)
                data = fh.read(incl)
                if len(data) < incl:
                    return
                pkt = StreamingPacketSniffer.parse_frame(
                    data, sec + (frac / 1e9 if nanos else frac / 1e6)
                )
                if pkt:
                    yield pkt

    table = LiveFlowTable()
    out: List[Dict[str, Any]] = []
    win_start = None
    n = 0
    for pkt in packets():
        ts = pkt["timestamp"]
        if win_start is None:
            win_start = ts
        while ts - win_start >= MC.WINDOW_SECONDS:
            snap = table.snapshot_flows(max_flows=256)
            table.extract_window_features(MC.WINDOW_SECONDS)
            out.append({"window_id": len(out) + 1, "window_end": win_start + MC.WINDOW_SECONDS,
                        "packet_count": n, "flows": snap, "active_flows": len(snap)})
            win_start += MC.WINDOW_SECONDS
            n = 0
        table.process_packet(pkt)
        n += 1
    if win_start is not None:
        snap = table.snapshot_flows(max_flows=256)
        out.append({"window_id": len(out) + 1, "window_end": win_start + MC.WINDOW_SECONDS,
                    "packet_count": n, "flows": snap, "active_flows": len(snap)})
    return [w for w in out if w["flows"]]


def windows_from_stream(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("flows"):
                out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pcap", default=os.path.join(ROOT, "captures", "live.pcap"))
    ap.add_argument("--state-stream", default=None,
                    help="Use a recorded telemetry JSONL instead of a PCAP")
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("CYBERWORLD MVP SMOKE TEST — real checkpoints, real runtime, real telemetry")
    print("=" * 78)
    print(MC.summary())
    MC.validate_temporal_contract_file()

    try:
        import torch  # noqa: F401
    except ImportError:
        print("\n[!] PyTorch is not installed — this smoke test drives the real torch runtime.")
        print("    Install it (pip install torch) or run the torch-free equivalents:")
        print("        python scripts/offline_contract_check.py")
        print("        python scripts/offline_pipeline_run.py")
        return 4

    # --- STAGE 0: load the production adapter (loads all four checkpoints strict)
    from control_backend.model_adapter import (
        AntigravityModelAdapter, flows_from_span_dicts, select_primary_target,
    )

    adapter = AntigravityModelAdapter()
    assert adapter.models_loaded, "adapter reported models_loaded=False"

    # --- telemetry
    if args.state_stream:
        windows = windows_from_stream(args.state_stream)
        source = args.state_stream
    else:
        if not os.path.exists(args.pcap):
            print(f"[-] capture not found: {args.pcap}")
            return 2
        windows = windows_from_pcap(args.pcap)
        source = args.pcap
    windows = windows[: args.windows]
    print(f"\n[TELEMETRY] {os.path.relpath(source, ROOT)} -> {len(windows)} non-empty "
          f"{MC.WINDOW_SECONDS}s windows")
    if not windows:
        print("[-] no windows with flows; nothing to infer on")
        return 3

    event = None
    for w in windows:
        flows = flows_from_span_dicts(w["flows"])
        try:
            target = select_primary_target(flows)
        except Exception:
            target = max(
                {ip: 1 for r in flows for ip in (r.src_ip, r.dst_ip)},
                key=lambda ip: sum(1 for r in flows if ip in (r.src_ip, r.dst_ip)),
            )
        print(f"\n[TELEMETRY] window #{w['window_id']} ready — flows={len(flows)} "
              f"packets={w.get('packet_count', 0)} target={target}")
        event = adapter.predict_window(
            target_ip=target,
            flows=flows,
            window_id=int(w.get("window_id", 0)),
            packet_count=int(w.get("packet_count", 0)),
            active_flows=int(w.get("active_flows", len(flows))),
        )
        s = event.stages
        print(f"[TGNE]      embedding_dim={s.tgne.embedding_dim} |H|={s.tgne.embedding_norm} "
              f"graph={s.tgne.graph_nodes} nodes / {s.tgne.graph_edges} edges")
        print(f"[BRANCH_A]  risk={s.branch_a.risk} technique={s.branch_a.technique} "
              f"confidence={s.branch_a.confidence} host={s.branch_a.target_host}")
        print(f"[BRANCH_B]  future_latent K={s.branch_b.forecast_steps} x {s.branch_b.latent_dim} "
              f"horizon={s.branch_b.forecast_horizon_seconds}s "
              f"risk_trajectory={s.branch_b.risk_trajectory}")
        print(f"[DEEPOP]    future_technique={s.deepop.technique_trajectory}")
        print(f"[DEEPOP]    confidence={s.deepop.confidence}")
        print(f"[INFERENCE] prediction event created  alert={event.prediction.alert} "
              f"level={event.prediction.alert_level} latency={event.latency.inference_ms}ms")

    # --- WebSocket payload == what the dashboard parses
    payload = event.model_dump() if hasattr(event, "model_dump") else event.dict()
    wire = json.loads(json.dumps(payload, default=str))
    print(f"\n[WEBSOCKET] prediction sent — {len(json.dumps(wire))} bytes, type={wire['type']}")

    # --- frontend contract check (fields read by web_dashboard/src/**)
    required = [
        ("type", wire.get("type") == "prediction"),
        ("model.threshold", "threshold" in wire["model"]),
        ("model.window_seconds", "window_seconds" in wire["model"]),
        ("model.forecast_steps", "forecast_steps" in wire["model"]),
        ("state.window_id", "window_id" in wire["state"]),
        ("prediction.risk", isinstance(wire["prediction"]["risk"], (int, float))),
        ("prediction.max_future_risk", isinstance(wire["prediction"]["max_future_risk"], (int, float))),
        ("prediction.alert", isinstance(wire["prediction"]["alert"], bool)),
        ("prediction.alert_level", isinstance(wire["prediction"]["alert_level"], str)),
        ("prediction.predicted_stage", isinstance(wire["prediction"]["predicted_stage"], str)),
        ("forecast[] length == K", len(wire["forecast"]) == MC.FORECAST_STEPS),
        ("forecast[].horizon_seconds", all("horizon_seconds" in f for f in wire["forecast"])),
        ("forecast[].risk", all("risk" in f for f in wire["forecast"])),
        ("forecast[].predicted_stage", all(f.get("predicted_stage") for f in wire["forecast"])),
        ("forecast[].technique_confidence",
            all(f.get("technique_confidence") is not None for f in wire["forecast"])),
        ("forecast horizon == 16s", wire["forecast"][-1]["horizon_seconds"] == MC.FORECAST_HORIZON_SECONDS),
        ("stages.tgne", wire["stages"]["tgne"]["embedding_dim"] == MC.TGNE_LATENT_DIM),
        ("stages.branch_a", wire["stages"]["branch_a"]["input_dim"] == MC.BRANCH_A_INPUT_DIM),
        ("stages.branch_b", len(wire["stages"]["branch_b"]["risk_trajectory"]) == MC.FORECAST_STEPS),
        ("stages.deepop", len(wire["stages"]["deepop"]["technique_trajectory"]) == MC.FORECAST_STEPS),
        ("stages.deepop.vocab", wire["stages"]["deepop"]["vocab_size"] == MC.DEEPOP_VOCAB_SIZE),
        ("explainability.available", wire["explainability"]["available"] is True),
        ("focus_ips", isinstance(wire["focus_ips"], list)),
    ]
    print("\n[DASHBOARD] backend event -> WebSocket payload -> frontend expected schema")
    bad = [name for name, ok in required if not ok]
    for name, ok in required:
        print(f"    {'OK ' if ok else 'FAIL'}  {name}")

    # Consistency: dashboard-visible values must come from this same pass.
    s = event.stages
    consistency = [
        ("prediction.risk == stages.branch_a.risk", event.prediction.risk == s.branch_a.risk),
        ("prediction.predicted_stage == stages.branch_a.technique",
         event.prediction.predicted_stage == s.branch_a.technique),
        ("prediction.max_future_risk == max(stages.branch_b.risk_trajectory)",
         abs(event.prediction.max_future_risk - max(s.branch_b.risk_trajectory)) < 1e-6),
        ("forecast[].risk == stages.branch_b.risk_trajectory",
         [f.risk for f in event.forecast] == s.branch_b.risk_trajectory),
        ("forecast[].predicted_stage == stages.deepop.technique_trajectory",
         [f.predicted_stage for f in event.forecast] == s.deepop.technique_trajectory),
        ("forecast[].technique_confidence == stages.deepop.confidence",
         [f.technique_confidence for f in event.forecast] == s.deepop.confidence),
    ]
    print("\n[TRACEABILITY] every displayed value traced back to its producing stage")
    for name, ok in consistency:
        print(f"    {'OK ' if ok else 'FAIL'}  {name}")
        if not ok:
            bad.append(name)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(wire, fh, indent=2, default=str)
        print(f"\n[+] event written to {args.json_out}")

    if bad:
        print(f"\n[FAIL] {len(bad)} contract/traceability check(s) failed: {bad}")
        return 1
    print("\n[OK] TGNE -> Branch A -> Branch B -> DeepOP -> event -> WebSocket -> dashboard schema")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
