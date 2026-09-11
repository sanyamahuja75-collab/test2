"""
Per-host temporal attribute extraction (the 15 scalars concatenated to the TGNE
latent to form Branch A's 27-dimensional input).

Kept torch-free and dependency-light so it can be unit-tested and reused by the
offline verification tooling without loading the full training stack. The live
HostTrajectoryExtractor delegates here, so there is exactly one implementation
of this preprocessing step.

Index map (must stay aligned with explainability.unified_explanation.FEATURE_NAMES[12:]):
     0: flow_count            1: fwd_bytes           2: bwd_bytes
     3: total_bytes           4: fwd_packets         5: bwd_packets
     6: total_packets         7: unique_peers        8: unique_dst_ports
     9: tcp_ratio            10: udp_ratio          11: avg_flow_duration
    12: byte_rate            13: packet_rate        14: active_conn_density
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

N_HOST_TEMPORAL_ATTRS = 15

HOST_ATTR_NAMES: List[str] = [
    "flow_count",
    "fwd_bytes",
    "bwd_bytes",
    "total_bytes",
    "fwd_packets",
    "bwd_packets",
    "total_packets",
    "unique_peers",
    "unique_dst_ports",
    "tcp_ratio",
    "udp_ratio",
    "avg_flow_duration",
    "byte_rate",
    "packet_rate",
    "active_conn_density",
]


def compute_host_temporal_attributes(
    host_ip: str,
    window_records: List[Any],
    window_duration: float,
    n_attrs: int = N_HOST_TEMPORAL_ATTRS,
) -> np.ndarray:
    """Computes the 15 normalized per-host temporal scalars for one window.

    `window_records` are UnifiedFlowRecord-like objects (duck-typed: src_ip,
    dst_ip, dst_port, protocol, duration, fwd/bwd bytes and packets).
    All outputs are clamped to [0, 1] — this is the exact normalization the
    Branch A checkpoint was trained with.
    """
    attrs = np.zeros(n_attrs, dtype=np.float32)
    n = len(window_records)
    if n == 0:
        return attrs

    fwd_b = sum(r.fwd_bytes for r in window_records)
    bwd_b = sum(r.bwd_bytes for r in window_records)
    tot_b = fwd_b + bwd_b
    fwd_p = sum(r.fwd_packets for r in window_records)
    bwd_p = sum(r.bwd_packets for r in window_records)
    tot_p = fwd_p + bwd_p

    peers = set()
    ports = set()
    tcp_count = 0
    udp_count = 0
    tot_dur = 0.0

    for r in window_records:
        peer = r.dst_ip if r.src_ip == host_ip else r.src_ip
        peers.add(peer)
        ports.add(r.dst_port)
        if r.protocol == 6:
            tcp_count += 1
        elif r.protocol == 17:
            udp_count += 1
        tot_dur += r.duration

    dur = max(1.0, window_duration)
    attrs[0] = min(1.0, np.log1p(n) / 10.0)
    attrs[1] = min(1.0, np.log1p(fwd_b) / 20.0)
    attrs[2] = min(1.0, np.log1p(bwd_b) / 20.0)
    attrs[3] = min(1.0, np.log1p(tot_b) / 20.0)
    attrs[4] = min(1.0, np.log1p(fwd_p) / 10.0)
    attrs[5] = min(1.0, np.log1p(bwd_p) / 10.0)
    attrs[6] = min(1.0, np.log1p(tot_p) / 10.0)
    attrs[7] = min(1.0, np.log1p(len(peers)) / 5.0)
    attrs[8] = min(1.0, np.log1p(len(ports)) / 5.0)
    attrs[9] = float(tcp_count) / n
    attrs[10] = float(udp_count) / n
    attrs[11] = min(1.0, (tot_dur / n) / 300.0)
    attrs[12] = min(1.0, np.log1p(tot_b / dur) / 15.0)
    attrs[13] = min(1.0, np.log1p(tot_p / dur) / 10.0)
    attrs[14] = min(1.0, float(len(peers)) / max(1, n))

    return attrs
