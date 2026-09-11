#!/usr/bin/env python3
"""
telemetry/run_telemetry.py
Live SPAN capture → 2s windows + 5-tuple flow snapshots (JSONL).

ML inference runs in control_backend (Dual-Branch + DeepOP), not here.
Use --no-inference (default via run_telemetry.sh) for the SOC path.
"""

import argparse
import json
import os
import queue
import signal
import struct
import sys
import threading
import time

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from telemetry.capture.sniffer import StreamingPacketSniffer
from telemetry.state.state_builder import LiveStateBuilder


def replay_pcap(path: str, callback, speed: float = 1.0, loop: bool = False, stop=lambda: False):
    """Replay a PCAP file through the SAME frame parser the live sniffer uses.

    Packets are paced against their recorded inter-arrival times (scaled by
    `speed`) so the downstream 2.0s windowing sees a realistic cadence. This is
    real captured telemetry, not synthetic data — it is the demo path for
    machines with no mirror NIC or no CAP_NET_RAW.
    """
    while True:
        with open(path, "rb") as fh:
            hdr = fh.read(24)
            if len(hdr) < 24:
                raise ValueError(f"{path} is not a valid PCAP file (short header)")
            magic = struct.unpack("=I", hdr[:4])[0]
            if magic in (0xA1B2C3D4, 0xA1B23C4D):
                endian = "<" if sys.byteorder == "little" else ">"
            elif magic in (0xD4C3B2A1, 0x4D3CB2A1):
                endian = ">" if sys.byteorder == "little" else "<"
            else:
                raise ValueError(f"{path} is not a PCAP file (magic=0x{magic:08X})")
            nanos = magic in (0xA1B23C4D, 0x4D3CB2A1)

            prev_ts = None
            while not stop():
                rec = fh.read(16)
                if len(rec) < 16:
                    break
                sec, frac, incl_len, _orig_len = struct.unpack(endian + "IIII", rec)
                data = fh.read(incl_len)
                if len(data) < incl_len:
                    break
                ts = sec + (frac / 1e9 if nanos else frac / 1e6)
                if prev_ts is not None and speed > 0:
                    delay = (ts - prev_ts) / speed
                    if 0 < delay < 5.0:
                        time.sleep(delay)
                prev_ts = ts
                pkt = StreamingPacketSniffer.parse_frame(data, time.time())
                if pkt:
                    callback(pkt)
        if not loop or stop():
            return


class AsyncStateRecorder:
    """Asynchronously writes state records to disk off the critical live path."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.queue: queue.Queue = queue.Queue(maxsize=1000)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def record(self, state_dict: dict):
        if not self.queue.full():
            self.queue.put_nowait(state_dict)

    def _worker(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "a", encoding="utf-8") as f:
            while self.running or not self.queue.empty():
                try:
                    item = self.queue.get(timeout=0.5)
                    clean_item = {}
                    for k, v in item.items():
                        if hasattr(v, "tolist"):
                            clean_item[k] = v.tolist()
                        else:
                            clean_item[k] = v
                    f.write(json.dumps(clean_item) + "\n")
                    f.flush()
                except queue.Empty:
                    continue
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


class AsyncPcapRecorder:
    """Asynchronously writes captured frames to standard PCAP format."""

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.queue: queue.Queue = queue.Queue(maxsize=10000)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def record(self, ts: float, raw_data: bytes):
        if not self.queue.full():
            self.queue.put_nowait((ts, raw_data))

    def _worker(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.output_path)), exist_ok=True)
        with open(self.output_path, "wb") as f:
            f.write(struct.pack("=IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1))
            f.flush()
            while self.running or not self.queue.empty():
                try:
                    ts, raw_data = self.queue.get(timeout=0.5)
                    sec = int(ts)
                    usec = int((ts - sec) * 1_000_000)
                    pkt_len = len(raw_data)
                    f.write(struct.pack("=IIII", sec, usec, pkt_len, pkt_len))
                    f.write(raw_data)
                    f.flush()
                except queue.Empty:
                    continue
                except Exception:
                    pass

    def stop(self):
        self.running = False
        self.thread.join(timeout=2.0)


def _emit_window(state: dict, recorder: AsyncStateRecorder | None) -> str:
    n_flows = len(state.get("flows") or [])
    if recorder:
        slim = {
            k: v
            for k, v in state.items()
            if k not in ("raw_state", "scaled_state")
        }
        recorder.record({**slim, "prediction": None})
    return f" | Flows: {n_flows} | [SPAN CAPTURE → Dual-Branch/DeepOP]"


def main():
    parser = argparse.ArgumentParser(
        description="CyberWorld Live SPAN Telemetry (capture for Dual-Branch/DeepOP)"
    )
    parser.add_argument("--interface", default="eth1", help="Observation interface")
    parser.add_argument(
        "--replay", "--pcap", dest="replay", default=None,
        help="Replay a PCAP capture instead of sniffing a NIC (no root required)",
    )
    parser.add_argument("--replay-speed", type=float, default=1.0,
                        help="Replay speed multiplier (1.0 = original pacing)")
    parser.add_argument("--replay-loop", action="store_true",
                        help="Loop the capture forever (continuous demo feed)")
    parser.add_argument("--dim", type=int, default=72, choices=[70, 72, 73])
    parser.add_argument("--record-state", default=None, help="JSONL state stream path")
    parser.add_argument("--record-pcap", default=None, help="Optional raw PCAP path")
    parser.add_argument("--max-windows", type=int, default=None)
    # Kept for CLI compatibility with older scripts; inference is never run here.
    parser.add_argument("--checkpoint", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--control-file", default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--no-inference",
        action="store_true",
        default=True,
        help="Capture only (default). ML runs in control_backend.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("CYBERWORLD LIVE SPAN TELEMETRY (CAPTURE ONLY)")
    print(f"  Source:               {args.replay or ('NIC ' + args.interface)}")
    print(f"  Window Resolution:    2.0 seconds")
    print(f"  Output:               5-tuple flows + window metadata → Dual-Branch/DeepOP")
    print(f"  Hot Path Storage:     Zero-Disk (Pure In-Memory)")
    print("=" * 80)
    sys.stdout.flush()

    state_builder = LiveStateBuilder(
        window_sec=2.0,
        history_len=15,
        dim_mode=args.dim,
    )

    recorder = AsyncStateRecorder(args.record_state) if args.record_state else None
    pcap_recorder = AsyncPcapRecorder(args.record_pcap) if args.record_pcap else None

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[*] Gracefully shutting down telemetry pipeline...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    def live_packet_callback(pkt):
        if pcap_recorder and "raw" in pkt:
            pcap_recorder.record(pkt["timestamp"], pkt["raw"])
        state_builder.ingest_packet(pkt)

    sniffer = None
    if args.replay:
        if not os.path.exists(args.replay):
            print(f"[!] Replay capture not found: {args.replay}", file=sys.stderr)
            sys.exit(1)
        print(f"[*] PCAP REPLAY MODE: {args.replay} "
              f"(speed={args.replay_speed}x loop={args.replay_loop})")
        sys.stdout.flush()
        sniffer_thread = threading.Thread(
            target=replay_pcap,
            args=(args.replay, live_packet_callback, args.replay_speed, args.replay_loop),
            kwargs={"stop": lambda: not running},
            daemon=True,
        )
    else:
        sniffer = StreamingPacketSniffer(interface=args.interface)
        sniffer_thread = threading.Thread(
            target=sniffer.capture_loop,
            args=(live_packet_callback,),
            daemon=True,
        )
    sniffer_thread.start()

    windows_processed = 0
    try:
        while running:
            time.sleep(0.05)
            if state_builder.is_window_ready():
                now = time.time()
                state = state_builder.close_window(close_ts=now)
                windows_processed += 1
                pred_str = _emit_window(state, recorder)

                timestr = time.strftime("%H:%M:%S", time.localtime(state["window_end"]))
                print(
                    f"[{timestr}] Window #{state['window_id']:04d} | "
                    f"Packets: {state['packet_count']:4d} | "
                    f"Pipe Latency: {state['pipeline_latency_ms']:.2f}ms"
                    f"{pred_str}"
                )
                sys.stdout.flush()

                if args.max_windows and windows_processed >= args.max_windows:
                    print(f"[*] Reached target {args.max_windows} windows. Exiting.")
                    break
    finally:
        running = False
        if sniffer is not None:
            sniffer.stop()
        if recorder:
            recorder.stop()
        if pcap_recorder:
            pcap_recorder.stop()


if __name__ == "__main__":
    main()
