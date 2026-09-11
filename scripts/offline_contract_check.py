#!/usr/bin/env python3
"""
scripts/offline_contract_check.py

Verifies that every production checkpoint on disk matches model_contract.py —
WITHOUT importing torch. It reads the .pt/.pth zip archives directly and
reconstructs tensor shapes from the pickled state dicts, so it runs anywhere
(CI, a laptop with no CUDA, a box where torch isn't installed yet).

This is the fast pre-flight gate:

    python scripts/offline_contract_check.py

Exit code 0 = every interface in the TGNE -> Branch A -> Branch B -> DeepOP
chain is internally consistent. Non-zero = a real mismatch, printed with the
expected and actual shapes.

For the full numeric end-to-end run (real inference through all four models)
use scripts/smoke_pipeline.py, which requires torch.
"""

from __future__ import annotations

import io
import json
import os
import pickle
import sys
import zipfile

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import model_contract as MC  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal torch-free .pt reader (shapes only; no torch import)
# ---------------------------------------------------------------------------

_DTYPES = {
    "FloatStorage": np.float32,
    "DoubleStorage": np.float64,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
    "BFloat16Storage": np.float32,
}


class _StorageType:
    def __init__(self, name: str):
        self.name = name
        self.dtype = _DTYPES.get(name, np.float32)


class _Stub:
    def __init__(self, name):
        self.name = name

    def __call__(self, *a, **k):
        return None


def _rebuild(storage, offset, size, stride, *rest):
    size = tuple(int(s) for s in size)
    if not size:
        return np.asarray(storage[offset])
    n = int(np.prod(size))
    return np.asarray(storage[offset: offset + n]).reshape(size)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zf, prefix):
        super().__init__(fh, encoding="latin1")
        self.zf = zf
        self.prefix = prefix

    def find_class(self, mod, name):
        if mod.startswith("torch") and name.endswith("Storage"):
            return _StorageType(name)
        if mod == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor_v3"):
            return _rebuild
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, rg, hooks: data
        try:
            import importlib

            return getattr(importlib.import_module(mod), name)
        except Exception:
            return _Stub(f"{mod}.{name}")

    def persistent_load(self, pid):
        assert pid[0] == "storage", pid
        storage_type, key = pid[1], pid[2]
        name = self.prefix + "data/" + str(key)
        if name not in self.zf.namelist():
            cands = [n for n in self.zf.namelist() if n.endswith("data/" + str(key))]
            if not cands:
                raise KeyError(f"storage {key} not found in archive")
            name = cands[0]
        return np.frombuffer(self.zf.read(name), dtype=storage_type.dtype)


def load_checkpoint(path: str):
    """Returns the checkpoint object with tensors as numpy arrays. No torch needed."""
    with zipfile.ZipFile(path) as zf:
        data_pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
        prefix = data_pkl[: -len("data.pkl")]
        return _Unpickler(io.BytesIO(zf.read(data_pkl)), zf, prefix).load()


# ---------------------------------------------------------------------------
# Expected tensor shapes, derived from model_contract
# ---------------------------------------------------------------------------

def _expected():
    C = MC
    branch_a = {
        "lstm.weight_ih_l0": (4 * C.BRANCH_A_HIDDEN_DIM, C.BRANCH_A_INPUT_DIM),
        "lstm.weight_hh_l0": (4 * C.BRANCH_A_HIDDEN_DIM, C.BRANCH_A_HIDDEN_DIM),
        "lstm.weight_ih_l1": (4 * C.BRANCH_A_HIDDEN_DIM, C.BRANCH_A_HIDDEN_DIM),
        "attention.proj.weight": (C.BRANCH_A_HIDDEN_DIM // 2, C.BRANCH_A_HIDDEN_DIM),
        "risk_head.3.weight": (1, 32),
        "technique_head.3.weight": (C.BRANCH_A_NUM_TECHNIQUES, C.BRANCH_A_HIDDEN_DIM),
        "gradation_head.3.weight": (C.BRANCH_A_NUM_GRADATIONS, 32),
    }
    branch_b_wdt = {
        "in_proj.weight": (C.BRANCH_B_D_MODEL, C.BRANCH_B_D_LATENT),
        "time_encoder.linear.weight": (C.BRANCH_B_D_MODEL, 1),
        "time_encoder.linear.bias": (C.BRANCH_B_D_MODEL,),
        "out_head.3.weight": (C.BRANCH_B_D_LATENT, C.BRANCH_B_FEEDFORWARD),
    }
    for layer in range(C.BRANCH_B_N_LAYERS):
        branch_b_wdt[f"transformer.layers.{layer}.self_attn.in_proj_weight"] = (
            3 * C.BRANCH_B_D_MODEL,
            C.BRANCH_B_D_MODEL,
        )
        branch_b_wdt[f"transformer.layers.{layer}.linear1.weight"] = (
            C.BRANCH_B_FEEDFORWARD,
            C.BRANCH_B_D_MODEL,
        )
    branch_b_risk = {
        "mlp.0.weight": (C.BRANCH_B_RISK_HIDDEN, C.BRANCH_B_D_LATENT),
        "mlp.3.weight": (1, C.BRANCH_B_RISK_HIDDEN),
    }
    deepop = {
        "pos_embed": (1, C.DEEPOP_MAX_SEQ_LEN, C.DEEPOP_D_MODEL),
        "token_embed.weight": (C.DEEPOP_VOCAB_SIZE, C.DEEPOP_D_MODEL),
        "future_proj.weight": (C.DEEPOP_D_MODEL, C.DEEPOP_D_LATENT),
        "prototypes": (C.DEEPOP_VOCAB_SIZE, C.DEEPOP_D_LATENT),
        "direct_head.weight": (C.DEEPOP_VOCAB_SIZE, C.DEEPOP_D_LATENT),
        "fc_out.weight": (C.DEEPOP_VOCAB_SIZE, C.DEEPOP_D_MODEL),
    }
    for layer in range(C.DEEPOP_N_LAYERS):
        deepop[f"layers.{layer}.cwa_self_attn.q_proj.weight"] = (C.DEEPOP_D_MODEL, C.DEEPOP_D_MODEL)
        deepop[f"layers.{layer}.cross_attn.in_proj_weight"] = (3 * C.DEEPOP_D_MODEL, C.DEEPOP_D_MODEL)
        deepop[f"layers.{layer}.ffn.0.weight"] = (C.DEEPOP_FEEDFORWARD, C.DEEPOP_D_MODEL)
    tgne = {
        "time_encoder.w.weight": (C.TGNE_TIME_DIM, 1),
        "embedding_module.attention_models.0.merger.fc1.weight": (
            C.TGNE_LATENT_DIM,
            (C.TGNE_LATENT_DIM + C.TGNE_TIME_DIM) + C.TGNE_NODE_FEAT_DIM,
        ),
        "embedding_module.attention_models.0.multi_head_target.q_proj_weight": (
            C.TGNE_LATENT_DIM + C.TGNE_TIME_DIM,
            C.TGNE_LATENT_DIM + C.TGNE_TIME_DIM,
        ),
        "embedding_module.attention_models.0.multi_head_target.k_proj_weight": (
            C.TGNE_LATENT_DIM + C.TGNE_TIME_DIM,
            C.TGNE_NODE_FEAT_DIM + C.TGNE_EDGE_FEAT_DIM + C.TGNE_TIME_DIM,
        ),
        "category_predictor.weight": (C.TGNE_NUM_CATEGORIES, C.TGNE_LATENT_DIM),
    }
    return [
        ("TGNE / BiTA", MC.TGNE_CHECKPOINT, None, tgne),
        ("Branch A", MC.BRANCH_A_CHECKPOINT, MC.BRANCH_A_STATE_KEY, branch_a),
        ("Branch B (WDT)", MC.BRANCH_B_CHECKPOINT, MC.BRANCH_B_WDT_KEY, branch_b_wdt),
        ("Branch B (risk head)", MC.BRANCH_B_CHECKPOINT, MC.BRANCH_B_RISK_KEY, branch_b_risk),
        ("DeepOP", MC.DEEPOP_CHECKPOINT, MC.DEEPOP_STATE_KEY, deepop),
    ]


def main() -> int:
    failures: list[str] = []

    print("=" * 78)
    print("CYBERWORLD MODEL CONTRACT — OFFLINE CHECKPOINT AUDIT (no torch required)")
    print("=" * 78)
    print(MC.summary())
    print()

    try:
        live = MC.validate_temporal_contract_file()
        print(
            f"[+] temporal contract  window={live['window_size_sec']}s "
            f"history={live['history_steps']} forecast={live['forecast_horizon_steps']} "
            f"({live['total_forecast_duration_sec']}s) — matches model_contract"
        )
    except Exception as e:
        failures.append(str(e))
        print(f"[-] temporal contract  {e}")

    for stage, path, key, expected in _expected():
        if not os.path.exists(path):
            failures.append(f"{stage}: checkpoint missing at {path}")
            print(f"[-] {stage:<22} MISSING {path}")
            continue
        try:
            blob = load_checkpoint(path)
        except Exception as e:
            failures.append(f"{stage}: unreadable checkpoint ({e})")
            print(f"[-] {stage:<22} UNREADABLE {path}: {e}")
            continue

        sd = blob[key] if key else blob
        if not isinstance(sd, dict):
            failures.append(f"{stage}: '{key}' is not a state dict")
            print(f"[-] {stage:<22} '{key}' is not a state dict")
            continue

        stage_fail = []
        for name, want in expected.items():
            if name not in sd:
                stage_fail.append(f"{name}: MISSING (checkpoint has {len(sd)} tensors)")
                continue
            got = tuple(int(x) for x in np.asarray(sd[name]).shape)
            if got != tuple(want):
                stage_fail.append(f"{name}: expected {tuple(want)} got {got}")

        n_params = sum(int(np.asarray(v).size) for v in sd.values() if hasattr(v, "shape"))
        if stage_fail:
            failures.extend(f"{stage}: {m}" for m in stage_fail)
            print(f"[-] {stage:<22} {len(stage_fail)} mismatch(es)")
            for m in stage_fail:
                print(f"      {m}")
        else:
            print(
                f"[+] {stage:<22} {len(sd):>3} tensors, {n_params:>8,} params — "
                f"all {len(expected)} contract shapes match"
            )

    # Extra cross-checks that manifests no longer disagree with the checkpoints.
    print()
    for label, manifest_path, checks in (
        (
            "branch_b.manifest.json",
            os.path.join(MC.REPO_ROOT, "saved_models", "branch_b", "branch_b.manifest.json"),
            [("architecture.num_encoder_layers", MC.BRANCH_B_N_LAYERS),
             ("architecture.d_latent", MC.BRANCH_B_D_LATENT),
             ("architecture.d_model", MC.BRANCH_B_D_MODEL)],
        ),
        (
            "deepop.manifest.json",
            os.path.join(MC.REPO_ROOT, "saved_models", "deepop", "deepop.manifest.json"),
            [("architecture.vocab_size", MC.DEEPOP_VOCAB_SIZE),
             ("architecture.nhead", MC.DEEPOP_N_HEADS),
             ("architecture.d_model", MC.DEEPOP_D_MODEL),
             ("architecture.num_decoder_layers", MC.DEEPOP_N_LAYERS)],
        ),
        (
            "branch_a.manifest.json",
            os.path.join(MC.REPO_ROOT, "saved_models", "branch_a", "branch_a.manifest.json"),
            [("architecture.input_dim", MC.BRANCH_A_INPUT_DIM),
             ("architecture.hidden_dim", MC.BRANCH_A_HIDDEN_DIM),
             ("feature_schema.tgne_latent_dim", MC.TGNE_LATENT_DIM)],
        ),
    ):
        doc = json.load(open(manifest_path, encoding="utf-8"))
        bad = []
        for dotted, want in checks:
            node = doc
            for part in dotted.split("."):
                node = node.get(part, {}) if isinstance(node, dict) else {}
            if node != want:
                bad.append(f"{dotted}: manifest={node} contract={want}")
        if bad:
            failures.extend(f"{label}: {m}" for m in bad)
            print(f"[-] {label:<24} {'; '.join(bad)}")
        else:
            print(f"[+] {label:<24} agrees with model_contract and the checkpoint")

    # DeepOP vocabulary must be the one the decoder head was trained with.
    print()
    try:
        dp = load_checkpoint(MC.DEEPOP_CHECKPOINT)
        ckpt_vocab = int(dp.get("vocab_size", -1))
        head_vocab = int(np.asarray(dp[MC.DEEPOP_STATE_KEY]["fc_out.weight"]).shape[0])
        if not (ckpt_vocab == head_vocab == MC.DEEPOP_VOCAB_SIZE):
            failures.append(
                f"DeepOP vocabulary disagreement: checkpoint meta={ckpt_vocab} "
                f"head={head_vocab} contract={MC.DEEPOP_VOCAB_SIZE}"
            )
            print(
                f"[-] deepop vocabulary     meta={ckpt_vocab} head={head_vocab} "
                f"contract={MC.DEEPOP_VOCAB_SIZE}"
            )
        else:
            print(f"[+] deepop vocabulary     one authoritative size: {head_vocab}")
    except Exception as e:
        failures.append(f"DeepOP vocabulary check failed: {e}")
        print(f"[-] deepop vocabulary     {e}")

    print()
    print("-" * 78)
    for iface in MC.INTERFACES:
        print(f"  {iface['edge']}")
        print(f"      in  : {iface['input']}")
        print(f"      out : {iface['output']}")
    print("-" * 78)

    if failures:
        print(f"\n[FAIL] {len(failures)} contract violation(s).")
        return 1
    print("\n[OK] All checkpoints, manifests and the temporal contract are consistent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
