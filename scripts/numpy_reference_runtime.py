"""
scripts/numpy_reference_runtime.py

A dependency-free (numpy only) REFERENCE implementation of the four production
models, used to execute the complete
    TGNE -> Branch A -> Branch B -> DeepOP
chain against the real checkpoints on machines where PyTorch is not installed.

This is a VERIFICATION tool, not the runtime. The product path is
control_backend/model_adapter.py running the actual torch modules; this file
exists so the chain can be proven numerically end-to-end (shapes, weights,
decoding, vocabulary) in a bare environment, and so CI can catch checkpoint /
architecture drift without a torch install.

Every layer here mirrors the exact torch module it replaces:
  * nn.Linear                      -> linear()
  * nn.LayerNorm                   -> layer_norm()
  * nn.LSTM (batch_first)          -> lstm()
  * nn.MultiheadAttention          -> mha() / mha_split()
  * nn.TransformerEncoderLayer     -> encoder_layer()  (norm_first=True, gelu)
  * TimeEncode / ContinuousTimeEncoding -> cos(Linear(1, d))
"""

from __future__ import annotations

import io
import math
import pickle
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# torch-free checkpoint loading
# ---------------------------------------------------------------------------

_DTYPES = {
    "FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES.get(name, np.float32)


def _rebuild(storage, offset, size, stride, *rest):
    size = tuple(int(s) for s in size)
    if not size:
        return np.asarray(storage[offset])
    n = int(np.prod(size))
    return np.array(storage[offset: offset + n]).reshape(size)


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zf, prefix):
        super().__init__(fh, encoding="latin1")
        self.zf, self.prefix = zf, prefix

    def find_class(self, mod, name):
        if mod.startswith("torch") and name.endswith("Storage"):
            return _StorageType(name)
        if mod == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor_v3"):
            return _rebuild
        if mod == "torch._utils" and name == "_rebuild_parameter":
            return lambda data, rg, hooks: data
        import importlib
        try:
            return getattr(importlib.import_module(mod), name)
        except Exception:
            return lambda *a, **k: None

    def persistent_load(self, pid):
        storage_type, key = pid[1], pid[2]
        name = self.prefix + "data/" + str(key)
        if name not in self.zf.namelist():
            name = [n for n in self.zf.namelist() if n.endswith("data/" + str(key))][0]
        return np.frombuffer(self.zf.read(name), dtype=storage_type.dtype)


def load_state(path: str):
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
        return _Unpickler(io.BytesIO(zf.read(pkl)), zf, pkl[: -len("data.pkl")]).load()


# ---------------------------------------------------------------------------
# primitive layers
# ---------------------------------------------------------------------------

def linear(x, w, b=None):
    y = x @ np.asarray(w).T
    return y + np.asarray(b) if b is not None else y


def relu(x):
    return np.maximum(x, 0.0)


def gelu(x):
    # torch's exact (erf) GELU
    from math import sqrt
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / sqrt(2.0)))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(e.sum(axis=axis, keepdims=True), 1e-30, None)


def layer_norm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * np.asarray(w) + np.asarray(b)


def l2_normalize(x, axis=-1, eps=1e-12):
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.clip(n, eps, None)


def lstm(x, sd, prefix="lstm.", num_layers=2):
    """nn.LSTM(batch_first=True) forward. x: [B, T, D] -> [B, T, H]. Gates: i, f, g, o."""
    out = x
    for layer in range(num_layers):
        w_ih = np.asarray(sd[f"{prefix}weight_ih_l{layer}"])
        w_hh = np.asarray(sd[f"{prefix}weight_hh_l{layer}"])
        b_ih = np.asarray(sd[f"{prefix}bias_ih_l{layer}"])
        b_hh = np.asarray(sd[f"{prefix}bias_hh_l{layer}"])
        H = w_hh.shape[1]
        B, T, _ = out.shape
        h = np.zeros((B, H), dtype=np.float32)
        c = np.zeros((B, H), dtype=np.float32)
        seq = []
        for t in range(T):
            g = out[:, t, :] @ w_ih.T + b_ih + h @ w_hh.T + b_hh
            i, f, gg, o = np.split(g, 4, axis=-1)
            i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
            gg = np.tanh(gg)
            c = f * c + i * gg
            h = o * np.tanh(c)
            seq.append(h.copy())
        out = np.stack(seq, axis=1).astype(np.float32)
    return out


def _attend(q, k, v, n_heads, additive_mask=None, key_padding_mask=None):
    """q: [B, Nq, E], k/v: [B, Nk, E] (already projected). Returns [B, Nq, E]."""
    B, Nq, E = q.shape
    Nk = k.shape[1]
    hd = E // n_heads
    q = q.reshape(B, Nq, n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, Nk, n_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, Nk, n_heads, hd).transpose(0, 2, 1, 3)
    scores = (q * (hd ** -0.5)) @ k.transpose(0, 1, 3, 2)     # [B, H, Nq, Nk]
    if additive_mask is not None:
        scores = scores + additive_mask
    if key_padding_mask is not None:
        scores = np.where(key_padding_mask[:, None, None, :], -np.inf, scores)
    attn = softmax(scores, axis=-1)
    attn = np.nan_to_num(attn, nan=0.0)
    out = attn @ v                                            # [B, H, Nq, hd]
    return out.transpose(0, 2, 1, 3).reshape(B, Nq, E)


def mha(q_in, kv_in, sd, prefix, n_heads, additive_mask=None, key_padding_mask=None):
    """nn.MultiheadAttention with PACKED in_proj_weight (self/cross, same embed dim)."""
    W = np.asarray(sd[prefix + "in_proj_weight"])
    b = np.asarray(sd[prefix + "in_proj_bias"])
    E = q_in.shape[-1]
    q = linear(q_in, W[:E], b[:E])
    k = linear(kv_in, W[E: 2 * E], b[E: 2 * E])
    v = linear(kv_in, W[2 * E:], b[2 * E:])
    out = _attend(q, k, v, n_heads, additive_mask, key_padding_mask)
    return linear(out, sd[prefix + "out_proj.weight"], sd[prefix + "out_proj.bias"])


def mha_split(q_in, kv_in, sd, prefix, n_heads, key_padding_mask=None):
    """nn.MultiheadAttention with SEPARATE q/k/v projections (kdim != embed_dim)."""
    Wq = np.asarray(sd[prefix + "q_proj_weight"])
    Wk = np.asarray(sd[prefix + "k_proj_weight"])
    Wv = np.asarray(sd[prefix + "v_proj_weight"])
    b = np.asarray(sd[prefix + "in_proj_bias"])
    E = Wq.shape[0]
    q = linear(q_in, Wq, b[:E])
    k = linear(kv_in, Wk, b[E: 2 * E])
    v = linear(kv_in, Wv, b[2 * E:])
    out = _attend(q, k, v, n_heads, None, key_padding_mask)
    return linear(out, sd[prefix + "out_proj.weight"], sd[prefix + "out_proj.bias"])


def encoder_layer(x, sd, prefix, n_heads, additive_mask=None):
    """nn.TransformerEncoderLayer(norm_first=True, activation='gelu'), eval mode."""
    h = layer_norm(x, sd[prefix + "norm1.weight"], sd[prefix + "norm1.bias"])
    x = x + mha(h, h, sd, prefix + "self_attn.", n_heads, additive_mask=additive_mask)
    h = layer_norm(x, sd[prefix + "norm2.weight"], sd[prefix + "norm2.bias"])
    h = linear(gelu(linear(h, sd[prefix + "linear1.weight"], sd[prefix + "linear1.bias"])),
               sd[prefix + "linear2.weight"], sd[prefix + "linear2.bias"])
    return x + h


# ---------------------------------------------------------------------------
# TGNE / BiTA  (graph_attention, n_layers=1, use_memory=False)
# ---------------------------------------------------------------------------

class NeighborFinder:
    """Mirror of bita/utils/utils.py NeighborFinder (uniform=False path)."""

    def __init__(self, adj_list):
        self.nb, self.ei, self.et = [], [], []
        for neighbors in adj_list:
            s = sorted(neighbors, key=lambda x: x[2])
            self.nb.append(np.array([x[0] for x in s], dtype=np.int64))
            self.ei.append(np.array([x[1] for x in s], dtype=np.int64))
            self.et.append(np.array([x[2] for x in s], dtype=np.float64))

    def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=10):
        N = len(source_nodes)
        neighbors = np.zeros((N, n_neighbors), dtype=np.int64)
        edge_times = np.zeros((N, n_neighbors), dtype=np.float64)
        edge_idxs = np.zeros((N, n_neighbors), dtype=np.int64)
        for i, (node, ts) in enumerate(zip(source_nodes, timestamps)):
            cut = np.searchsorted(self.et[node], ts)
            nb, ei, et = self.nb[node][:cut], self.ei[node][:cut], self.et[node][:cut]
            if len(nb) > 0:
                nb, ei, et = nb[-n_neighbors:], ei[-n_neighbors:], et[-n_neighbors:]
                neighbors[i, n_neighbors - len(nb):] = nb
                edge_idxs[i, n_neighbors - len(ei):] = ei
                edge_times[i, n_neighbors - len(et):] = et
        return neighbors, edge_idxs, edge_times


class TGNE:
    """ExtendedTGN.get_host_embeddings for n_layers=1, use_memory=False."""

    def __init__(self, state: Dict[str, np.ndarray], latent_dim=12, n_heads=2):
        self.sd = state
        self.d = latent_dim
        self.n_heads = n_heads

    def _time_encode(self, t):
        # TimeEncode: cos(Linear(1, d)(t))
        w = np.asarray(self.sd["embedding_module.time_encoder.w.weight"])   # (d, 1)
        b = np.asarray(self.sd["embedding_module.time_encoder.w.bias"])     # (d,)
        return np.cos(t[..., None] * w[:, 0] + b)

    def host_embeddings(self, node_ids, timestamp, neighbor_finder, edge_features,
                        n_node_features, n_neighbors=10):
        sd = self.sd
        N = len(node_ids)
        if N == 0:
            return np.zeros((0, self.d), dtype=np.float32)

        ts = np.full(N, float(timestamp))
        # node features are zeros in the live path (no static host attributes)
        src_feat = np.zeros((N, n_node_features), dtype=np.float32)
        src_time = self._time_encode(np.zeros((N, 1)))                        # [N, 1, d]

        neighbors, edge_idxs, edge_times = neighbor_finder.get_temporal_neighbor(
            node_ids, ts, n_neighbors=n_neighbors
        )
        nb_feat = np.zeros((N, n_neighbors, n_node_features), dtype=np.float32)
        edge_deltas = ts[:, None] - edge_times
        edge_time_emb = self._time_encode(edge_deltas)                        # [N, K, d]
        edge_feat = np.asarray(edge_features)[edge_idxs]                      # [N, K, 12]
        mask = neighbors == 0                                                 # True = padding

        query = np.concatenate([src_feat[:, None, :], src_time], axis=2)      # [N, 1, 24]
        key = np.concatenate([nb_feat, edge_feat, edge_time_emb], axis=2)     # [N, K, 36]

        invalid = mask.all(axis=1)
        kpm = mask.copy()
        kpm[invalid, 0] = False

        p = "embedding_module.attention_models.0."
        attn_out = mha_split(query, key, sd, p + "multi_head_target.", self.n_heads,
                             key_padding_mask=kpm)                            # [N, 1, 24]
        attn_out = attn_out[:, 0, :]
        attn_out = np.where(invalid[:, None], 0.0, attn_out)

        merged = np.concatenate([attn_out, src_feat], axis=1)                 # [N, 36]
        h = relu(linear(merged, sd[p + "merger.fc1.weight"], sd[p + "merger.fc1.bias"]))
        return linear(h, sd[p + "merger.fc2.weight"], sd[p + "merger.fc2.bias"]).astype(np.float32)


# ---------------------------------------------------------------------------
# Branch A
# ---------------------------------------------------------------------------

class BranchA:
    def __init__(self, state, hidden_dim=64, num_layers=2):
        self.sd = state
        self.H = hidden_dim
        self.L = num_layers

    def __call__(self, x):                       # x: [B, T, 27]
        sd = self.sd
        out = lstm(x, sd, "lstm.", self.L)       # [B, T, H]
        energy = np.tanh(linear(out, sd["attention.proj.weight"], sd["attention.proj.bias"]))
        scores = linear(energy, sd["attention.v.weight"])[..., 0]
        weights = softmax(scores, axis=-1)
        context = np.einsum("bt,bth->bh", weights, out)

        risk = sigmoid(linear(relu(linear(context, sd["risk_head.0.weight"], sd["risk_head.0.bias"])),
                              sd["risk_head.3.weight"], sd["risk_head.3.bias"]))[..., 0]
        tech = linear(relu(linear(context, sd["technique_head.0.weight"], sd["technique_head.0.bias"])),
                      sd["technique_head.3.weight"], sd["technique_head.3.bias"])
        grad = linear(relu(linear(context, sd["gradation_head.0.weight"], sd["gradation_head.0.bias"])),
                      sd["gradation_head.3.weight"], sd["gradation_head.3.bias"])
        return {"risk_score": risk, "technique_logits": tech,
                "gradation_logits": grad, "attention_weights": weights, "context": context}


# ---------------------------------------------------------------------------
# Branch B
# ---------------------------------------------------------------------------

class BranchB:
    def __init__(self, wdt_state, risk_state, d_model=64, n_heads=4, n_layers=3):
        self.sd = wdt_state
        self.rd = risk_state
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers

    def _encode(self, x):
        T = x.shape[1]
        causal = np.triu(np.full((T, T), -np.inf, dtype=np.float64), 1)
        for i in range(self.n_layers):
            x = encoder_layer(x, self.sd, f"transformer.layers.{i}.", self.n_heads,
                              additive_mask=causal)
        return x

    def rollout(self, h_seq, K=8, delta_t=2.0, decay=0.95, max_ctx=10):
        sd = self.sd
        curr = np.array(h_seq, dtype=np.float32)
        preds = []
        for k in range(K):
            ctx = curr[:, -max_ctx:, :]
            B, T, _ = ctx.shape
            x = linear(ctx, sd["in_proj.weight"], sd["in_proj.bias"])
            dt = np.full((B, T, 1), delta_t, dtype=np.float32)
            t_emb = np.cos(linear(dt, sd["time_encoder.linear.weight"], sd["time_encoder.linear.bias"]))
            x = x + t_emb
            h = self._encode(x)
            delta = linear(relu_free_gelu(linear(h[:, -1, :], sd["out_head.0.weight"], sd["out_head.0.bias"])),
                           sd["out_head.3.weight"], sd["out_head.3.bias"])
            if k > 0:
                delta = delta * float(decay ** k)
            nxt = ctx[:, -1, :] + delta
            preds.append(nxt)
            curr = np.concatenate([curr, nxt[:, None, :]], axis=1)
        return np.stack(preds, axis=1).astype(np.float32)     # [B, K, 12]

    def step_risks(self, h_future):
        rd = self.rd
        B, K, D = h_future.shape
        flat = h_future.reshape(B * K, D)
        h = gelu(linear(flat, rd["mlp.0.weight"], rd["mlp.0.bias"]))
        r = sigmoid(linear(h, rd["mlp.3.weight"], rd["mlp.3.bias"]))
        return r.reshape(B, K)


def relu_free_gelu(x):
    """out_head uses GELU (Linear -> GELU -> Dropout -> Linear)."""
    return gelu(x)


# ---------------------------------------------------------------------------
# DeepOP
# ---------------------------------------------------------------------------

class DeepOP:
    def __init__(self, state, vocab_tokens: Sequence[str], d_model=72, n_heads=6,
                 n_layers=2, window_sizes=(2, 4, 8), d_latent=12):
        self.sd = state
        self.tokens = list(vocab_tokens)
        self.V = len(self.tokens)
        self.d_model, self.n_heads, self.n_layers = d_model, n_heads, n_layers
        self.window_sizes = list(window_sizes)
        self.heads_per_scale = n_heads // len(self.window_sizes)
        self.d_latent = d_latent
        self.pad_idx, self.bos_idx = 0, 1

    def _cwa(self, x, prefix):
        """Multi-scale causal window self-attention (dense-mask form, identical result)."""
        sd = self.sd
        B, N, E = x.shape
        hd = E // self.n_heads
        q = linear(x, sd[prefix + "q_proj.weight"], sd[prefix + "q_proj.bias"])
        k = linear(x, sd[prefix + "k_proj.weight"], sd[prefix + "k_proj.bias"])
        v = linear(x, sd[prefix + "v_proj.weight"], sd[prefix + "v_proj.bias"])
        q = q.reshape(B, N, self.n_heads, hd).transpose(0, 2, 1, 3)
        k = k.reshape(B, N, self.n_heads, hd).transpose(0, 2, 1, 3)
        v = v.reshape(B, N, self.n_heads, hd).transpose(0, 2, 1, 3)

        i_idx = np.arange(N)[:, None]
        j_idx = np.arange(N)[None, :]
        outs = []
        for s, win in enumerate(self.window_sizes):
            h0 = s * self.heads_per_scale
            h1 = h0 + self.heads_per_scale
            W = min(win, N)
            valid = (j_idx <= i_idx) & ((i_idx - j_idx) < W)
            mask = np.where(valid, 0.0, -np.inf)
            scores = (q[:, h0:h1] * (hd ** -0.5)) @ k[:, h0:h1].transpose(0, 1, 3, 2) + mask
            attn = np.nan_to_num(softmax(scores, axis=-1), nan=0.0)
            outs.append(attn @ v[:, h0:h1])
        out = np.concatenate(outs, axis=1).transpose(0, 2, 1, 3).reshape(B, N, E)
        return linear(out, sd[prefix + "out_proj.weight"], sd[prefix + "out_proj.bias"])

    def forward(self, h_future, tokens):
        sd = self.sd
        B, S = tokens.shape
        K = h_future.shape[1]
        x = np.asarray(sd["token_embed.weight"])[tokens] * math.sqrt(self.d_model)
        x = x + np.asarray(sd["pos_embed"])[:, :S, :]
        memory = linear(h_future, sd["future_proj.weight"], sd["future_proj.bias"])

        for i in range(self.n_layers):
            p = f"layers.{i}."
            t2 = self._cwa(x, p + "cwa_self_attn.")
            x = layer_norm(x + t2, sd[p + "norm1.weight"], sd[p + "norm1.bias"])
            t2 = mha(x, memory, sd, p + "cross_attn.", self.n_heads)
            x = layer_norm(x + t2, sd[p + "norm2.weight"], sd[p + "norm2.bias"])
            t2 = linear(gelu(linear(x, sd[p + "ffn.0.weight"], sd[p + "ffn.0.bias"])),
                        sd[p + "ffn.3.weight"], sd[p + "ffn.3.bias"])
            x = layer_norm(x + t2, sd[p + "norm3.weight"], sd[p + "norm3.bias"])

        gate = float(np.asarray(sd["future_gate"]))
        if S <= K:
            x = x + gate * memory[:, :S, :]
        else:
            x[:, :K, :] = x[:, :K, :] + gate * memory
        x = layer_norm(x, sd["norm.weight"], sd["norm.bias"])
        dec_logits = linear(x, sd["fc_out.weight"], sd["fc_out.bias"])

        protos = np.asarray(sd["prototypes"])
        active = (np.linalg.norm(protos, axis=-1) > 1e-4).astype(np.float32)
        if active.sum() > 0:
            proto_logits = float(np.asarray(sd["proto_scale"])) * np.einsum(
                "bkd,vd->bkv", l2_normalize(h_future), l2_normalize(protos + 1e-8)
            ) * active[None, None, :]
        else:
            proto_logits = np.zeros((B, K, self.V), dtype=np.float32)

        direct = linear(h_future[:, : min(S, K), :], sd["direct_head.weight"], sd["direct_head.bias"])
        if S <= K:
            return dec_logits + direct + proto_logits[:, :S, :]
        d_part = linear(h_future, sd["direct_head.weight"], sd["direct_head.bias"]) + proto_logits
        pad = np.repeat(d_part[:, -1:, :], S - K, axis=1)
        return dec_logits + np.concatenate([d_part, pad], axis=1)

    def forecast_sequence(self, h_future, max_steps=8, observed_token=None, benign_idx=3):
        B = h_future.shape[0]
        curr = np.full((B, 1), self.bos_idx, dtype=np.int64)
        step_conf: List[List[float]] = [[] for _ in range(B)]
        for step in range(max_steps):
            logits = self.forward(h_future, curr)
            nxt = logits[:, -1, :].copy()
            nxt[:, self.bos_idx] = -1e9
            nxt[:, self.pad_idx] = -1e9
            if step == 0 and observed_token is not None:
                for b in range(B):
                    ot = int(observed_token[b])
                    if ot not in (self.bos_idx, self.pad_idx):
                        nxt[b, ot] += 1.8 if ot == benign_idx else 1.2
            probs = softmax(nxt, axis=-1)
            tok = nxt.argmax(axis=-1)
            for b in range(B):
                step_conf[b].append(float(probs[b, tok[b]]))
            curr = np.concatenate([curr, tok[:, None]], axis=1)
        pred = curr[:, 1:]
        names = [[self.tokens[int(t)] for t in pred[b]] for b in range(B)]
        return pred, names, step_conf
