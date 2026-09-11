#!/usr/bin/env python3
"""
Pre-flight verification of the four production checkpoints.

Checks that every checkpoint exists, loads, exposes the state-dict key the runtime
reads, and carries tensors whose shapes match model_contract.py. Previously this
script listed expected key names that do not exist in the shipped DeepOP
checkpoint (`token_embedding.weight`, `decoder_layers.0...` instead of
`token_embed.weight`, `layers.0...`) and never actually compared them, so it
reported success regardless. It now really checks.

    python scripts/ensure_checkpoints.py

Exit 0 = every checkpoint is present and matches the contract.

Torch is used when available; otherwise this falls back to the torch-free reader
in scripts/offline_contract_check.py so it can run before dependencies are set up.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model_contract as MC  # noqa: E402


def _load(path):
    try:
        import torch
        return torch.load(path, map_location="cpu", weights_only=False), "torch"
    except ImportError:
        from scripts.offline_contract_check import load_checkpoint
        return load_checkpoint(path), "numpy"


REQUIRED = [
    ("TGNE-TA (ExtendedTGN / BiTA)", MC.TGNE_CHECKPOINT, None, {
        "time_encoder.w.weight": (MC.TGNE_TIME_DIM, 1),
        "category_predictor.weight": (MC.TGNE_NUM_CATEGORIES, MC.TGNE_LATENT_DIM),
        "embedding_module.attention_models.0.multi_head_target.q_proj_weight": (24, 24),
    }),
    ("Branch A (MultiTaskLSTM)", MC.BRANCH_A_CHECKPOINT, MC.BRANCH_A_STATE_KEY, {
        "lstm.weight_ih_l0": (4 * MC.BRANCH_A_HIDDEN_DIM, MC.BRANCH_A_INPUT_DIM),
        "risk_head.3.weight": (1, 32),
        "technique_head.3.weight": (MC.BRANCH_A_NUM_TECHNIQUES, MC.BRANCH_A_HIDDEN_DIM),
    }),
    ("Branch B (HostWorldDynamicsTransformer)", MC.BRANCH_B_CHECKPOINT, MC.BRANCH_B_WDT_KEY, {
        "in_proj.weight": (MC.BRANCH_B_D_MODEL, MC.BRANCH_B_D_LATENT),
        "time_encoder.linear.weight": (MC.BRANCH_B_D_MODEL, 1),
        f"transformer.layers.{MC.BRANCH_B_N_LAYERS - 1}.self_attn.in_proj_weight":
            (3 * MC.BRANCH_B_D_MODEL, MC.BRANCH_B_D_MODEL),
        "out_head.3.weight": (MC.BRANCH_B_D_LATENT, MC.BRANCH_B_FEEDFORWARD),
    }),
    ("Branch B risk head (InfiltrationRiskHead)", MC.BRANCH_B_CHECKPOINT, MC.BRANCH_B_RISK_KEY, {
        "mlp.0.weight": (MC.BRANCH_B_RISK_HIDDEN, MC.BRANCH_B_D_LATENT),
    }),
    ("DeepOP CWA Forecasting Decoder", MC.DEEPOP_CHECKPOINT, MC.DEEPOP_STATE_KEY, {
        "token_embed.weight": (MC.DEEPOP_VOCAB_SIZE, MC.DEEPOP_D_MODEL),
        "pos_embed": (1, MC.DEEPOP_MAX_SEQ_LEN, MC.DEEPOP_D_MODEL),
        f"layers.{MC.DEEPOP_N_LAYERS - 1}.cross_attn.in_proj_weight":
            (3 * MC.DEEPOP_D_MODEL, MC.DEEPOP_D_MODEL),
        "fc_out.weight": (MC.DEEPOP_VOCAB_SIZE, MC.DEEPOP_D_MODEL),
    }),
]


def verify_checkpoints() -> bool:
    print("=" * 74)
    print("CHECKPOINT PRE-FLIGHT — shapes verified against model_contract.py")
    print("=" * 74)
    print(MC.summary())
    print()

    ok_all = True
    for name, path, key, expected in REQUIRED:
        if not os.path.exists(path):
            print(f"[-] MISSING   {name}\n              {path}")
            ok_all = False
            continue
        size_kb = os.path.getsize(path) / 1024.0
        try:
            blob, backend = _load(path)
        except Exception as e:
            print(f"[!] CORRUPT   {name}: {e}")
            ok_all = False
            continue

        sd = blob[key] if key else blob
        if key and (not isinstance(blob, dict) or key not in blob):
            print(f"[-] BAD KEY   {name}: checkpoint has no '{key}' "
                  f"(keys: {list(blob) if isinstance(blob, dict) else type(blob)})")
            ok_all = False
            continue

        problems = []
        for tname, want in expected.items():
            if tname not in sd:
                problems.append(f"{tname}: MISSING")
                continue
            got = tuple(int(x) for x in sd[tname].shape)
            if got != tuple(want):
                problems.append(f"{tname}: expected {tuple(want)} got {got}")

        n_tensors = len(sd)
        n_params = sum(int(getattr(v, "numel", lambda: v.size)()) for v in sd.values()
                       if hasattr(v, "shape"))
        if problems:
            ok_all = False
            print(f"[-] MISMATCH  {name:<42} ({backend})")
            for p in problems:
                print(f"              {p}")
        else:
            print(f"[+] VERIFIED  {name:<42} | {size_kb:>7.1f} KB | "
                  f"{n_tensors:>3} tensors | {n_params:>8,} weights")

    print()
    if ok_all:
        print("[+] All core checkpoints present and shape-consistent with the contract.")
    else:
        print("[-] One or more checkpoints are missing or inconsistent. Retrain/restore:")
        print("    - Branch A: python branch_a_gnn_lstm/train_branch_a.py")
        print("    - Branch B: python branch_b_world_model/train_branch_b.py")
        print("    - DeepOP:   python deepop_decoder/train_cwa_decoder.py")
        print("    Backups are under saved_models/backups/.")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if verify_checkpoints() else 1)
