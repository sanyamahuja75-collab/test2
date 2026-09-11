"""
model_contract.py — THE single authoritative model/temporal contract for CyberWorld.

Every dimension in this file was derived from the ACTUAL production checkpoints
(`saved_models/**`, `bita/saved_models/**`) by inspecting the serialized tensor
shapes, not from documentation or manifests. Where a manifest disagreed with a
checkpoint, the checkpoint won and the manifest was corrected.

Runtime modules import from here instead of hardcoding constants.

Derivation (tensor shapes in the shipped checkpoints):

  TGNE-TA   bita/saved_models/bita_bigru_transformer-warden_alerts.pth
      time_encoder.w.weight                                 (12, 1)   -> time dim 12
      embedding_module.attention_models.0.merger.fc1.weight (12, 36)  -> 24 query + 12 src
      ...multi_head_target.q_proj_weight                    (24, 24)  -> query dim 24
      ...multi_head_target.k_proj_weight                    (24, 36)  -> key dim 36
      category_predictor.weight                             (81, 12)  -> latent 12, 81 categories

  Branch A  saved_models/branch_a/branch_a_lstm.pt   (key: model_state_dict)
      lstm.weight_ih_l0        (256, 27)  -> input 27 (12 TGNE latent + 15 host attrs), hidden 64
      lstm.weight_ih_l1        (256, 64)  -> 2 layers
      attention.proj.weight    (32, 64)   -> attn dim = hidden // 2
      risk_head.3.weight       (1, 32)    -> 1 risk output
      technique_head.3.weight  (14, 64)   -> 14 techniques
      gradation_head.3.weight  (4, 32)    -> 4 gradations

  Branch B  saved_models/branch_b/host_wdt.pt        (keys: wdt_state_dict, risk_head_state_dict)
      in_proj.weight                       (64, 12)  -> d_latent 12 -> d_model 64
      time_encoder.linear.weight           (64, 1)   -> Linear(1, d_model) + cos  (NOT freq/sin-cos)
      transformer.layers.{0,1,2}...                  -> 3 encoder layers (manifest said 2: WRONG)
      ...self_attn.in_proj_weight          (192, 64) -> 3 * d_model
      ...linear1.weight                    (128, 64) -> dim_feedforward 128
      out_head.3.weight                    (12, 128) -> delta in latent space
      risk_head mlp.0.weight               (32, 12)  -> hidden 32

  DeepOP    saved_models/deepop/cwa_forecast_decoder.pt (key: decoder_state_dict)
      pos_embed                (1, 16, 72) -> max_seq_len 16, d_model 72
      token_embed.weight       (10, 72)    -> vocab 10   (manifest said 24: WRONG)
      prototypes               (10, 12)    -> vocab 10, d_latent 12
      direct_head.weight       (10, 12)
      fc_out.weight            (10, 72)
      layers.{0,1}...                      -> 2 decoder layers
      layers.*.ffn.0.weight    (144, 72)   -> dim_feedforward 144
      checkpoint["vocab_size"] == 10

  The DeepOP vocabulary of 10 is exactly what deepop_decoder.joint_vocab builds:
      3 special tokens (<PAD>, <BOS>, <EOS>) + 7 network-observable macro techniques
      (Benign.None, C2.T1071, CredentialAccess.T1110, Exfiltration.T1005,
       Impact.T1498, InitialAccess.T1190, Recon.T1595) = 10.

  DeepOP head count: CausalWindowAttention asserts n_heads % len(window_sizes) == 0.
  With the trained window_sizes [2, 4, 8] (3 scales) and d_model 72, n_heads must be
  divisible by 3 and divide 72 -> 6. The manifest's "nhead: 4" is arithmetically
  impossible for this module and was corrected.
"""

from __future__ import annotations

import json
import os
from typing import List

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))


def _p(*parts: str) -> str:
    return os.path.join(REPO_ROOT, *parts)


# ---------------------------------------------------------------------------
# Temporal contract (mirrors config/temporal_contract.json :: live_micro)
# ---------------------------------------------------------------------------
TEMPORAL_MODE = "live_micro"
WINDOW_SECONDS: float = 2.0          # 2-second telemetry window
HISTORY_STEPS: int = 5               # 5-window causal history  (10 s of context)
FORECAST_STEPS: int = 8              # 8-window forecast        (16 s horizon)
FORECAST_HORIZON_SECONDS: float = WINDOW_SECONDS * FORECAST_STEPS   # 16.0

# ---------------------------------------------------------------------------
# TGNE / BiTA
# ---------------------------------------------------------------------------
TGNE_LATENT_DIM: int = 12
TGNE_NODE_FEAT_DIM: int = 12
TGNE_EDGE_FEAT_DIM: int = 12
TGNE_TIME_DIM: int = 12
TGNE_N_LAYERS: int = 1
TGNE_N_HEADS: int = 2
TGNE_NUM_CATEGORIES: int = 81
TGNE_N_NEIGHBORS: int = 10
TGNE_USE_MEMORY: bool = False
TGNE_CHECKPOINT: str = _p("bita", "saved_models", "bita_bigru_transformer-warden_alerts.pth")
TGNE_CONFIG: str = _p("bita", "saved_models", "bita_config.json")

# ---------------------------------------------------------------------------
# Branch A — MultiTaskLSTM (current risk + current technique)
# ---------------------------------------------------------------------------
HOST_TEMPORAL_ATTR_DIM: int = 15
BRANCH_A_INPUT_DIM: int = TGNE_LATENT_DIM + HOST_TEMPORAL_ATTR_DIM       # 27
BRANCH_A_HIDDEN_DIM: int = 64
BRANCH_A_NUM_LAYERS: int = 2
BRANCH_A_NUM_TECHNIQUES: int = 14
BRANCH_A_NUM_GRADATIONS: int = 4
BRANCH_A_CHECKPOINT: str = _p("saved_models", "branch_a", "branch_a_lstm.pt")
BRANCH_A_STATE_KEY: str = "model_state_dict"

# ---------------------------------------------------------------------------
# Branch B — HostWorldDynamicsTransformer (future latent trajectory + future risk)
# ---------------------------------------------------------------------------
BRANCH_B_D_LATENT: int = TGNE_LATENT_DIM                                 # 12
BRANCH_B_D_MODEL: int = 64
BRANCH_B_N_HEADS: int = 4
BRANCH_B_N_LAYERS: int = 3          # checkpoint has transformer.layers.0/1/2
BRANCH_B_FEEDFORWARD: int = 128
BRANCH_B_RISK_HIDDEN: int = 32
BRANCH_B_CHECKPOINT: str = _p("saved_models", "branch_b", "host_wdt.pt")
BRANCH_B_WDT_KEY: str = "wdt_state_dict"
BRANCH_B_RISK_KEY: str = "risk_head_state_dict"

# ---------------------------------------------------------------------------
# DeepOP — CWA forecast decoder (future technique trajectory)
# ---------------------------------------------------------------------------
DEEPOP_D_LATENT: int = TGNE_LATENT_DIM                                   # 12
DEEPOP_D_MODEL: int = 72
DEEPOP_N_HEADS: int = 6
DEEPOP_N_LAYERS: int = 2
DEEPOP_FEEDFORWARD: int = 144
DEEPOP_MAX_SEQ_LEN: int = 16
DEEPOP_WINDOW_SIZES: List[int] = [2, 4, 8]
DEEPOP_VOCAB_SIZE: int = 10
DEEPOP_CHECKPOINT: str = _p("saved_models", "deepop", "cwa_forecast_decoder.pt")
DEEPOP_STATE_KEY: str = "decoder_state_dict"

# ---------------------------------------------------------------------------
# Operating point
# ---------------------------------------------------------------------------
ALERT_THRESHOLD: float = 0.65

MODEL_NAME: str = "Antigravity-DualBranch-DeepOP"
MODEL_VERSION: str = "3.3-SOC"
CHECKPOINT_LABEL: str = (
    "tgne bita_bigru_transformer-warden_alerts.pth + branch_a_lstm.pt "
    "+ host_wdt.pt + cwa_forecast_decoder.pt"
)


# ---------------------------------------------------------------------------
# Interface documentation — the exact tensor contract between the four models.
# Consumed by scripts/offline_contract_check.py and scripts/smoke_pipeline.py.
# ---------------------------------------------------------------------------
INTERFACES = [
    {
        "edge": "Telemetry -> Graph",
        "input": f"list[UnifiedFlowRecord] over a {WINDOW_SECONDS}s window",
        "output": f"TemporalEventStream(edge_features=[E, {TGNE_EDGE_FEAT_DIM}] float32)",
        "meaning": "12 normalized flow features per observed 5-tuple edge",
    },
    {
        "edge": "Graph -> TGNE/BiTA",
        "input": f"node_ids[int], timestamps[float], edge_features [E, {TGNE_EDGE_FEAT_DIM}]",
        "output": f"H_t [N_hosts, {TGNE_LATENT_DIM}] float32",
        "meaning": "per-host temporal graph embedding at window_end",
    },
    {
        "edge": "TGNE -> Branch A",
        "input": (
            f"concat(H_t[target] [{TGNE_LATENT_DIM}], host_temporal_attrs [{HOST_TEMPORAL_ATTR_DIM}]) "
            f"stacked over {HISTORY_STEPS} windows -> [1, {HISTORY_STEPS}, {BRANCH_A_INPUT_DIM}]"
        ),
        "output": (
            f"risk_score [1] in [0,1]; technique_logits [1, {BRANCH_A_NUM_TECHNIQUES}]; "
            f"gradation_logits [1, {BRANCH_A_NUM_GRADATIONS}]"
        ),
        "meaning": "current risk + current MITRE technique for the scored host",
    },
    {
        "edge": "TGNE -> Branch B",
        "input": f"H_t history [1, {HISTORY_STEPS}, {BRANCH_B_D_LATENT}]",
        "output": f"H_future [1, {FORECAST_STEPS}, {BRANCH_B_D_LATENT}] + step_risks [1, {FORECAST_STEPS}]",
        "meaning": f"future latent trajectory over {FORECAST_HORIZON_SECONDS}s + per-step risk",
    },
    {
        "edge": "Branch B -> DeepOP",
        "input": (
            f"H_future [1, {FORECAST_STEPS}, {DEEPOP_D_LATENT}] as cross-attention memory "
            f"+ observed token id in [0, {DEEPOP_VOCAB_SIZE})"
        ),
        "output": f"token ids [1, {FORECAST_STEPS}] in [0, {DEEPOP_VOCAB_SIZE}) + per-step probabilities",
        "meaning": "future ATT&CK technique trajectory over the same horizon",
    },
]


# ---------------------------------------------------------------------------
# Assertions — fail loudly, never silently reshape.
# ---------------------------------------------------------------------------
class ContractViolation(AssertionError):
    """Raised when a runtime tensor does not match the authoritative contract."""


def assert_last_dim(name: str, shape, expected: int) -> None:
    shape = tuple(int(s) for s in shape)
    if not shape or shape[-1] != expected:
        raise ContractViolation(
            f"[CONTRACT] {name}: expected last dimension {expected}, got shape {shape}. "
            "Do not reshape/pad/truncate to make this pass — fix the producing stage."
        )


def assert_shape(name: str, shape, expected) -> None:
    shape = tuple(int(s) for s in shape)
    expected = tuple(expected)
    if len(shape) != len(expected) or any(
        e is not None and int(e) != s for s, e in zip(shape, expected)
    ):
        raise ContractViolation(
            f"[CONTRACT] {name}: expected shape {expected}, got {shape}. "
            "Do not reshape/pad/truncate to make this pass — fix the producing stage."
        )


def validate_temporal_contract_file(path: str | None = None) -> dict:
    """Cross-check this module against config/temporal_contract.json (live_micro)."""
    path = path or _p("config", "temporal_contract.json")
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    live = doc["timescales"]["live_micro"]
    mismatches = []
    if float(live["window_size_sec"]) != WINDOW_SECONDS:
        mismatches.append(("window_size_sec", live["window_size_sec"], WINDOW_SECONDS))
    if int(live["history_steps"]) != HISTORY_STEPS:
        mismatches.append(("history_steps", live["history_steps"], HISTORY_STEPS))
    if int(live["forecast_horizon_steps"]) != FORECAST_STEPS:
        mismatches.append(("forecast_horizon_steps", live["forecast_horizon_steps"], FORECAST_STEPS))
    if float(live["total_forecast_duration_sec"]) != FORECAST_HORIZON_SECONDS:
        mismatches.append(
            ("total_forecast_duration_sec", live["total_forecast_duration_sec"], FORECAST_HORIZON_SECONDS)
        )
    if mismatches:
        raise ContractViolation(
            "config/temporal_contract.json disagrees with model_contract.py: "
            + "; ".join(f"{k}: file={a} contract={b}" for k, a, b in mismatches)
        )
    return live


def summary() -> str:
    return (
        f"window={WINDOW_SECONDS}s history={HISTORY_STEPS} forecast={FORECAST_STEPS} "
        f"({FORECAST_HORIZON_SECONDS}s) | TGNE latent={TGNE_LATENT_DIM} | "
        f"BranchA in={BRANCH_A_INPUT_DIM} tech={BRANCH_A_NUM_TECHNIQUES} | "
        f"BranchB d_model={BRANCH_B_D_MODEL} layers={BRANCH_B_N_LAYERS} | "
        f"DeepOP d_model={DEEPOP_D_MODEL} heads={DEEPOP_N_HEADS} vocab={DEEPOP_VOCAB_SIZE}"
    )
