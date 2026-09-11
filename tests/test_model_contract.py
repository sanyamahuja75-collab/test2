"""
Contract regression tests — torch-free, so they run in CI before the heavy deps
are installed. They lock in the three inconsistencies that were found between the
shipped checkpoints and the repository's configuration:

  1. Branch B's time encoder parameter names must match the checkpoint
     (`time_encoder.linear.*`), otherwise strict loading silently dropped them.
  2. Branch B has 3 transformer encoder layers, not the 2 the manifest claimed.
  3. DeepOP's vocabulary is 10 tokens (checkpoint + joint_vocab), not the 24 the
     manifest claimed.

Run with:  python -m pytest tests/test_model_contract.py
      or:  python tests/test_model_contract.py
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import model_contract as MC  # noqa: E402
from scripts.offline_contract_check import load_checkpoint  # noqa: E402
from branch_a_gnn_lstm.technique_vocab import TECHNIQUE_VOCAB  # noqa: E402
from deepop_decoder.joint_vocab import get_joint_vocab  # noqa: E402
from data_unification.host_attributes import (  # noqa: E402
    N_HOST_TEMPORAL_ATTRS,
    HOST_ATTR_NAMES,
)


def test_temporal_contract_matches_config_file():
    live = MC.validate_temporal_contract_file()
    assert live["window_size_sec"] == 2.0
    assert live["history_steps"] == 5
    assert live["forecast_horizon_steps"] == 8
    assert live["total_forecast_duration_sec"] == 16.0


def test_branch_a_feature_composition():
    assert MC.TGNE_LATENT_DIM + MC.HOST_TEMPORAL_ATTR_DIM == MC.BRANCH_A_INPUT_DIM == 27
    assert N_HOST_TEMPORAL_ATTRS == MC.HOST_TEMPORAL_ATTR_DIM
    assert len(HOST_ATTR_NAMES) == MC.HOST_TEMPORAL_ATTR_DIM
    assert len(TECHNIQUE_VOCAB) == MC.BRANCH_A_NUM_TECHNIQUES


def test_branch_a_checkpoint_shapes():
    sd = load_checkpoint(MC.BRANCH_A_CHECKPOINT)[MC.BRANCH_A_STATE_KEY]
    assert sd["lstm.weight_ih_l0"].shape == (4 * MC.BRANCH_A_HIDDEN_DIM, MC.BRANCH_A_INPUT_DIM)
    assert sd["technique_head.3.weight"].shape[0] == len(TECHNIQUE_VOCAB)
    assert sd["gradation_head.3.weight"].shape[0] == MC.BRANCH_A_NUM_GRADATIONS


def test_branch_b_has_three_encoder_layers():
    sd = load_checkpoint(MC.BRANCH_B_CHECKPOINT)[MC.BRANCH_B_WDT_KEY]
    layers = {int(k.split(".")[2]) for k in sd if k.startswith("transformer.layers.")}
    assert layers == set(range(MC.BRANCH_B_N_LAYERS)) == {0, 1, 2}


def test_branch_b_time_encoder_parameter_names_match_checkpoint():
    """Regression: the model class must expose `time_encoder.linear.*`.

    A refactor renamed this to `freq_linear`/`proj`; combined with a
    load_state_dict override forcing strict=False, the trained time encoder was
    silently discarded and Branch B ran with random weights in that submodule.
    """
    sd = load_checkpoint(MC.BRANCH_B_CHECKPOINT)[MC.BRANCH_B_WDT_KEY]
    assert sd["time_encoder.linear.weight"].shape == (MC.BRANCH_B_D_MODEL, 1)
    assert sd["time_encoder.linear.bias"].shape == (MC.BRANCH_B_D_MODEL,)

    src = open(os.path.join(ROOT, "branch_b_world_model", "rollout_encoder_decoder.py"),
               encoding="utf-8").read()
    assert "self.linear = nn.Linear(1, d_model)" in src, \
        "ContinuousTimeEncoding must use the checkpoint's `linear` projection"
    assert "self.freq_linear" not in src, "stale freq_linear parameterisation is back"
    assert "super().load_state_dict(state_dict, strict=False" not in src, \
        "Branch B must not force strict=False on load"


def test_deepop_vocabulary_is_single_sourced():
    dp = load_checkpoint(MC.DEEPOP_CHECKPOINT)
    sd = dp[MC.DEEPOP_STATE_KEY]
    vocab = get_joint_vocab()
    assert int(dp["vocab_size"]) == MC.DEEPOP_VOCAB_SIZE == 10
    assert sd["fc_out.weight"].shape[0] == MC.DEEPOP_VOCAB_SIZE
    assert sd["token_embed.weight"].shape[0] == MC.DEEPOP_VOCAB_SIZE
    assert sd["prototypes"].shape == (MC.DEEPOP_VOCAB_SIZE, MC.DEEPOP_D_LATENT)
    assert vocab.vocab_size == MC.DEEPOP_VOCAB_SIZE


def test_deepop_head_count_is_arithmetically_possible():
    # CausalWindowAttention asserts d_model % n_heads == 0 and n_heads % n_scales == 0
    assert MC.DEEPOP_D_MODEL % MC.DEEPOP_N_HEADS == 0
    assert MC.DEEPOP_N_HEADS % len(MC.DEEPOP_WINDOW_SIZES) == 0


def test_manifests_agree_with_checkpoints():
    bb = json.load(open(os.path.join(ROOT, "saved_models", "branch_b", "branch_b.manifest.json")))
    dp = json.load(open(os.path.join(ROOT, "saved_models", "deepop", "deepop.manifest.json")))
    ba = json.load(open(os.path.join(ROOT, "saved_models", "branch_a", "branch_a.manifest.json")))
    assert bb["architecture"]["num_encoder_layers"] == MC.BRANCH_B_N_LAYERS
    assert dp["architecture"]["vocab_size"] == MC.DEEPOP_VOCAB_SIZE
    assert dp["architecture"]["nhead"] == MC.DEEPOP_N_HEADS
    assert ba["architecture"]["input_dim"] == MC.BRANCH_A_INPUT_DIM


def test_tgne_latent_feeds_branch_a_and_branch_b():
    sd = load_checkpoint(MC.TGNE_CHECKPOINT)
    assert sd["category_predictor.weight"].shape == (MC.TGNE_NUM_CATEGORIES, MC.TGNE_LATENT_DIM)
    bb = load_checkpoint(MC.BRANCH_B_CHECKPOINT)[MC.BRANCH_B_WDT_KEY]
    # Branch B consumes the TGNE latent directly
    assert bb["in_proj.weight"].shape[1] == MC.TGNE_LATENT_DIM
    # DeepOP cross-attends the same latent space
    dp = load_checkpoint(MC.DEEPOP_CHECKPOINT)[MC.DEEPOP_STATE_KEY]
    assert dp["future_proj.weight"].shape[1] == MC.TGNE_LATENT_DIM


def test_host_attributes_are_normalised():
    class _Rec:
        src_ip, dst_ip, dst_port, protocol = "10.0.0.1", "10.0.0.2", 443, 6
        fwd_bytes = bwd_bytes = 10_000_000
        fwd_packets = bwd_packets = 500_000
        duration = 1.5

    from data_unification.host_attributes import compute_host_temporal_attributes
    attrs = compute_host_temporal_attributes("10.0.0.1", [_Rec()] * 50, MC.WINDOW_SECONDS)
    assert attrs.shape == (MC.HOST_TEMPORAL_ATTR_DIM,)
    assert np.all(attrs >= 0.0) and np.all(attrs <= 1.0)
    assert compute_host_temporal_attributes("x", [], MC.WINDOW_SECONDS).shape == (15,)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS  {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL  {name}: {exc}")
    print(f"\n{'ALL PASS' if not failures else str(failures) + ' FAILED'}")
    sys.exit(1 if failures else 0)
