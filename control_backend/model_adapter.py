"""
control_backend/model_adapter.py
Live Antigravity dual-branch + DeepOP inference adapter.

Consumes UnifiedFlowRecord windows from Containerlab SPAN,
runs TGNE-TA → Branch A / Branch B (WDT) / DeepOP CWA, and emits PredictionEvent.
"""

from datetime import datetime
import logging
import os
import time
from typing import Dict, List, Optional, Any

import numpy as np
import torch

import model_contract as MC
from control_backend.schema import (
    ModelMetadata,
    StateMetadata,
    ForecastPoint,
    ExplainabilityGroup,
    ExplainabilityFeature,
    ExplainabilityPayload,
    PredictionData,
    LatencyData,
    EarlyWarningData,
    PredictionEvent,
    FocusEdge,
    TgneStage,
    BranchAStage,
    BranchBStage,
    DeepOpStage,
    StageProvenance,
)
from data_unification.unified_schema import UnifiedFlowRecord, CoarseCategory
from data_unification.temporal_config import (
    LIVE_WINDOW_SIZE_SEC,
    DEFAULT_ROLLOUT_HORIZON_LIVE,
    DEFAULT_HISTORY_STEPS,
)
from data_unification.multi_dataset_stream import HostTrajectoryExtractor
from data_unification.behavioral_fingerprint import BehavioralFlowFingerprinter
from explainability.unified_explanation import FEATURE_NAMES

logger = logging.getLogger("antigravity.model_adapter")

TECHNIQUE_TO_MITRE = {
    "Benign": ("Benign", "None", "TA0000", "No malicious activity observed"),
    "T1046": ("Reconnaissance", "T1046 Network Service Discovery", "TA0043", "Network service discovery"),
    "T1595": ("Reconnaissance", "T1595 Active Scanning", "TA0043", "Active scanning"),
    "T1110": ("Credential Access", "T1110 Brute Force", "TA0006", "Brute force authentication"),
    "T1190": ("Initial Access", "T1190 Exploit Public-Facing Application", "TA0001", "Exploit public-facing app"),
    "T1189": ("Initial Access", "T1189 Drive-by Compromise", "TA0001", "Drive-by compromise"),
    "T1071": ("Command and Control", "T1071 Application Layer Protocol", "TA0011", "Application-layer C2"),
    "T1071.001": ("Command and Control", "T1071.001 Web Protocols", "TA0011", "Web-protocol C2"),
    "T1568.001": ("Command and Control", "T1568.001 Fast Flux DNS", "TA0011", "Fast-flux DNS"),
    "T1204": ("Execution", "T1204 User Execution", "TA0002", "User execution"),
    "T1005": ("Collection", "T1005 Data from Local System", "TA0009", "Local data collection"),
    "T1498": ("Impact", "T1498 Network Denial of Service", "TA0040", "Network DoS"),
    "T1498.001": ("Impact", "T1498.001 Direct Network Flood", "TA0040", "Direct network flood"),
    "T1020": ("Exfiltration", "T1020 Automated Exfiltration", "TA0010", "Automated exfiltration"),
}

FEATURE_GROUP_MAP = {
    **{f"H_emb_{i}": "TGNE Latent" for i in range(MC.TGNE_LATENT_DIM)},
    "flow_count": "Connectivity",
    "fwd_bytes": "Volume",
    "bwd_bytes": "Volume",
    "total_bytes": "Volume",
    "fwd_packets": "Volume",
    "bwd_packets": "Volume",
    "total_packets": "Volume",
    "unique_peers": "Connectivity",
    "unique_dst_ports": "Connectivity",
    "tcp_ratio": "Connectivity",
    "udp_ratio": "Connectivity",
    "avg_flow_duration": "Timing",
    "byte_rate": "Volume",
    "packet_rate": "Volume",
    "active_conn_density": "Connectivity",
}

def model_metadata() -> ModelMetadata:
    """Single place the API and every PredictionEvent get their model metadata from."""
    return ModelMetadata(
        name=MC.MODEL_NAME,
        version=MC.MODEL_VERSION,
        feature_count=MC.BRANCH_A_INPUT_DIM,
        history_steps=MC.HISTORY_STEPS,
        window_seconds=MC.WINDOW_SECONDS,
        forecast_steps=MC.FORECAST_STEPS,
        checkpoint=MC.CHECKPOINT_LABEL,
        threshold=MC.ALERT_THRESHOLD,
    )


class AntigravityModelAdapter:
    """Dual-Branch + DeepOP live inference → dashboard PredictionEvent."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.h_state_history: List[torch.Tensor] = []
        self.feature_history: List[torch.Tensor] = []
        self.h_state_history_by_target: Dict[str, List[torch.Tensor]] = {}
        self.feature_history_by_target: Dict[str, List[torch.Tensor]] = {}
        self.alert_threshold = MC.ALERT_THRESHOLD
        self.window_seconds = LIVE_WINDOW_SIZE_SEC
        self.forecast_steps = DEFAULT_ROLLOUT_HORIZON_LIVE
        self.history_steps = DEFAULT_HISTORY_STEPS
        self.fingerprinter = BehavioralFlowFingerprinter()
        self.models_loaded = False
        # The temporal contract is authoritative for the whole chain; a drift here
        # means checkpoints trained on 2s/5/8 would be driven at another cadence.
        MC.validate_temporal_contract_file()
        if (
            self.window_seconds != MC.WINDOW_SECONDS
            or self.forecast_steps != MC.FORECAST_STEPS
            or self.history_steps != MC.HISTORY_STEPS
        ):
            raise MC.ContractViolation(
                f"[CONTRACT] runtime temporal settings (window={self.window_seconds}s, "
                f"history={self.history_steps}, forecast={self.forecast_steps}) disagree with "
                f"model_contract ({MC.WINDOW_SECONDS}s / {MC.HISTORY_STEPS} / {MC.FORECAST_STEPS})."
            )
        self._load_models()

    def _load_models(self):
        import sys

        # Prefer repo/bita over ambient editable installs (e.g. SIH/src/model.py)
        from control_backend.lab_config import REPO_ROOT

        repo = REPO_ROOT
        bita = os.path.join(repo, "bita")
        os.chdir(repo)
        sys.path = [
            p
            for p in sys.path
            if p
            and "SIH" not in p
            and os.path.abspath(p) != os.path.abspath(bita)
        ]
        sys.path.insert(0, bita)
        if repo not in sys.path:
            sys.path.insert(0, repo)
        for k in list(sys.modules):
            if k == "model" or k.startswith("model."):
                del sys.modules[k]

        from branch_a_gnn_lstm.lstm_multitask import MultiTaskLSTM
        from branch_a_gnn_lstm.sequence_dataset import TECHNIQUE_VOCAB
        from branch_a_gnn_lstm.train_branch_a import build_or_load_tgne_ta
        from branch_b_world_model.rollout_encoder_decoder import HostWorldDynamicsTransformer
        from branch_b_world_model.infiltration_head import InfiltrationRiskHead
        from deepop_decoder.forecast_decoder import DeepOPForecastDecoder
        from deepop_decoder.joint_vocab import get_joint_vocab, consolidate_network_technique

        self.technique_vocab = TECHNIQUE_VOCAB
        if len(self.technique_vocab) != MC.BRANCH_A_NUM_TECHNIQUES:
            raise MC.ContractViolation(
                f"[CONTRACT] TECHNIQUE_VOCAB has {len(self.technique_vocab)} entries but the trained "
                f"Branch A technique head has {MC.BRANCH_A_NUM_TECHNIQUES} classes."
            )

        # ---- Stage 1: TGNE / BiTA -----------------------------------------
        self.tgn = build_or_load_tgne_ta()
        tgne_latent = int(getattr(self.tgn, "embedding_dimension", MC.TGNE_LATENT_DIM))
        if tgne_latent != MC.TGNE_LATENT_DIM:
            raise MC.ContractViolation(
                f"[CONTRACT] TGNE embedding_dimension={tgne_latent}, contract={MC.TGNE_LATENT_DIM}"
            )
        logger.info(
            "[TGNE] loaded ckpt=%s input_dim(edge)=%s latent_dim=%s categories=%s",
            os.path.relpath(MC.TGNE_CHECKPOINT, repo),
            MC.TGNE_EDGE_FEAT_DIM,
            tgne_latent,
            MC.TGNE_NUM_CATEGORIES,
        )
        self.extractor = HostTrajectoryExtractor(
            tgne_ta_model=self.tgn,
            window_size_sec=self.window_seconds,
            n_temporal_attrs=MC.HOST_TEMPORAL_ATTR_DIM,
        )

        # ---- Stage 2: Branch A --------------------------------------------
        self.branch_a = MultiTaskLSTM(
            input_dim=MC.BRANCH_A_INPUT_DIM,
            hidden_dim=MC.BRANCH_A_HIDDEN_DIM,
            num_layers=MC.BRANCH_A_NUM_LAYERS,
        ).to(self.device)
        ckpt = self._load_checkpoint("Branch A", MC.BRANCH_A_CHECKPOINT, MC.BRANCH_A_STATE_KEY)
        # strict=True: any architecture drift must fail here, not silently at inference.
        self.branch_a.load_state_dict(ckpt, strict=True)
        self.branch_a.eval()
        logger.info(
            "[BRANCH_A] loaded ckpt=branch_a_lstm.pt input_dim=%s hidden=%s techniques=%s gradations=%s",
            MC.BRANCH_A_INPUT_DIM,
            MC.BRANCH_A_HIDDEN_DIM,
            MC.BRANCH_A_NUM_TECHNIQUES,
            MC.BRANCH_A_NUM_GRADATIONS,
        )

        # ---- Stage 3: Branch B --------------------------------------------
        self.wdt = HostWorldDynamicsTransformer(
            d_latent=MC.BRANCH_B_D_LATENT,
            d_model=MC.BRANCH_B_D_MODEL,
            n_heads=MC.BRANCH_B_N_HEADS,
            n_layers=MC.BRANCH_B_N_LAYERS,
            dim_feedforward=MC.BRANCH_B_FEEDFORWARD,
            max_horizon=MC.FORECAST_STEPS,
        ).to(self.device)
        self.risk_head = InfiltrationRiskHead(
            d_latent=MC.BRANCH_B_D_LATENT, hidden_dim=MC.BRANCH_B_RISK_HIDDEN
        ).to(self.device)
        bb = self._load_checkpoint("Branch B", MC.BRANCH_B_CHECKPOINT, None)
        for key in (MC.BRANCH_B_WDT_KEY, MC.BRANCH_B_RISK_KEY):
            if key not in bb:
                raise RuntimeError(
                    f"Branch B checkpoint {MC.BRANCH_B_CHECKPOINT} is missing '{key}'."
                )
        self.wdt.load_state_dict(bb[MC.BRANCH_B_WDT_KEY], strict=True)
        self.risk_head.load_state_dict(bb[MC.BRANCH_B_RISK_KEY], strict=True)
        self.wdt.eval()
        self.risk_head.eval()
        logger.info(
            "[BRANCH_B] loaded ckpt=host_wdt.pt d_latent=%s d_model=%s layers=%s K=%s (%ss horizon)",
            MC.BRANCH_B_D_LATENT,
            MC.BRANCH_B_D_MODEL,
            MC.BRANCH_B_N_LAYERS,
            MC.FORECAST_STEPS,
            MC.FORECAST_HORIZON_SECONDS,
        )

        # ---- Stage 4: DeepOP ----------------------------------------------
        self.vocab = get_joint_vocab()
        self.consolidate_network_technique = consolidate_network_technique
        self.deepop = DeepOPForecastDecoder(
            d_latent=MC.DEEPOP_D_LATENT,
            d_model=MC.DEEPOP_D_MODEL,
            vocab_size=self.vocab.vocab_size,
            n_heads=MC.DEEPOP_N_HEADS,
            num_layers=MC.DEEPOP_N_LAYERS,
            window_sizes=list(MC.DEEPOP_WINDOW_SIZES),
            dim_feedforward=MC.DEEPOP_FEEDFORWARD,
            max_seq_len=MC.DEEPOP_MAX_SEQ_LEN,
        ).to(self.device)
        dp = self._load_checkpoint("DeepOP", MC.DEEPOP_CHECKPOINT, None)
        ckpt_vocab = int(dp.get("vocab_size", MC.DEEPOP_VOCAB_SIZE))
        if ckpt_vocab != self.vocab.vocab_size:
            raise MC.ContractViolation(
                f"[CONTRACT] DeepOP checkpoint declares vocab_size={ckpt_vocab} but the runtime "
                f"joint vocabulary has {self.vocab.vocab_size} tokens."
            )
        self.deepop.load_state_dict(dp[MC.DEEPOP_STATE_KEY], strict=True)
        self.deepop.eval()
        logger.info(
            "[DEEPOP] loaded ckpt=cwa_forecast_decoder.pt d_latent=%s d_model=%s heads=%s vocab=%s tokens=%s",
            MC.DEEPOP_D_LATENT,
            MC.DEEPOP_D_MODEL,
            MC.DEEPOP_N_HEADS,
            self.vocab.vocab_size,
            self.vocab.tokens,
        )

        self.models_loaded = True
        logger.info(
            "[PIPELINE] TGNE -> Branch A -> Branch B -> DeepOP ready (device=%s) | %s",
            self.device,
            MC.summary(),
        )

    @staticmethod
    def _load_checkpoint(stage: str, path: str, state_key: Optional[str]):
        """Load a production checkpoint. Missing/corrupt checkpoints fail loudly —
        a silently randomly-initialised model would emit plausible-looking garbage."""
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{stage} checkpoint not found at {path}. The MVP path requires the real "
                "production checkpoints; refusing to run an untrained model."
            )
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if state_key is None:
            return blob
        if not isinstance(blob, dict) or state_key not in blob:
            raise RuntimeError(
                f"{stage} checkpoint {path} does not contain '{state_key}'. "
                f"Top-level keys: {list(blob) if isinstance(blob, dict) else type(blob)}"
            )
        return blob[state_key]

    def reset_history(self):
        self.h_state_history.clear()
        self.feature_history.clear()
        self.h_state_history_by_target.clear()
        self.feature_history_by_target.clear()

    def _build_embedding(
        self, target_ip: str, flows: List[UnifiedFlowRecord]
    ) -> tuple[np.ndarray, int, int]:
        """Stage 1: real graph construction + real TGNE forward pass.

        Returns (H_t[target] [12], graph_nodes, graph_edges).
        With no observed flows for the host there is no graph and therefore no
        embedding: the zero vector here is the *absence of observation*, not a
        stand-in for a model output (Branch A was trained with zero-padded
        warm-up windows, so this is the trained representation of "no traffic").
        """
        h_emb = np.zeros(MC.TGNE_LATENT_DIM, dtype=np.float32)
        if not flows:
            return h_emb, 0, 0

        try:
            trajectories = self.extractor.extract_trajectories(flows)
        except Exception as e:  # graph construction / TGNE forward must not fail silently
            raise RuntimeError(f"TGNE graph construction or forward pass failed: {e}") from e

        nodes = {ip for r in flows for ip in (r.src_ip, r.dst_ip) if ip}
        edges = {(r.src_ip, r.dst_ip) for r in flows if r.src_ip and r.dst_ip}

        if target_ip in trajectories and len(trajectories[target_ip]) > 0:
            emb = np.asarray(trajectories[target_ip][-1].embedding, dtype=np.float32)
            MC.assert_last_dim("TGNE H_t", emb.shape, MC.TGNE_LATENT_DIM)
            return emb, len(nodes), len(edges)

        raise RuntimeError(
            f"TGNE produced no embedding for target host {target_ip} "
            f"despite {len(flows)} observed flows ({len(nodes)} hosts, {len(edges)} edges)."
        )

    def _explain(self, feature_vector: np.ndarray) -> ExplainabilityPayload:
        x = (
            torch.from_numpy(feature_vector.astype(np.float32))
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        x.requires_grad_(True)
        # Stay in eval() — autograd works fine in eval, and running this pass in
        # train() applied dropout, so the saliency explained a *different* forward
        # pass than the risk the dashboard displays.
        was_training = self.branch_a.training
        self.branch_a.eval()
        try:
            out = self.branch_a(x)
            risk = out["risk_score"]
            if risk.ndim > 0:
                risk = risk.reshape(-1)[0]
            risk.backward()
            grads = (
                x.grad[0, -1, :].detach().cpu().numpy()
                if x.grad is not None
                else np.zeros(MC.BRANCH_A_INPUT_DIM, dtype=np.float32)
            )
            inputs = x[0, -1, :].detach().cpu().numpy()
            attributions = np.abs(grads * inputs)
        finally:
            if was_training:
                self.branch_a.train()

        total = float(attributions.sum()) + 1e-12
        attributions = attributions / total

        group_scores: Dict[str, float] = {}
        top_features: List[ExplainabilityFeature] = []
        ranked = sorted(
            zip(FEATURE_NAMES, attributions.tolist()),
            key=lambda t: t[1],
            reverse=True,
        )
        for name, score in ranked:
            group = FEATURE_GROUP_MAP.get(name, "General")
            group_scores[group] = group_scores.get(group, 0.0) + float(score)
        for name, score in ranked[:8]:
            top_features.append(
                ExplainabilityFeature(
                    feature=name,
                    score=round(float(score), 4),
                    group=FEATURE_GROUP_MAP.get(name, "General"),
                )
            )

        groups = [
            ExplainabilityGroup(name=g, percentage=round(100.0 * pct, 1))
            for g, pct in sorted(group_scores.items(), key=lambda kv: -kv[1])
        ]
        return ExplainabilityPayload(
            available=True,
            method="Input x Gradient Saliency",
            groups=groups,
            top_features=top_features,
        )

    def _alert_level(self, risk: float) -> str:
        # Bands are anchored to the single calibrated operating threshold so the
        # dashboard badge and the `alert` flag can never disagree.
        thr = self.alert_threshold
        if risk >= min(1.0, thr + 0.20):
            return "CRITICAL"
        if risk >= thr:
            return "ELEVATED"
        if risk >= thr * 0.75:
            return "WARNING"
        return "NOMINAL"

    def predict_window(
        self,
        target_ip: str,
        flows: List[UnifiedFlowRecord],
        window_id: int = 0,
        attack_active: bool = False,
        attack_phase: Optional[str] = None,
        is_mitigated: bool = False,
        packet_count: int = 0,
        pipeline_latency_ms: float = 0.0,
        active_flows: Optional[int] = None,
    ) -> PredictionEvent:
        t0 = time.perf_counter()

        if is_mitigated:
            flows = []

        host_flows = [
            record
            for record in flows
            if record.src_ip == target_ip or record.dst_ip == target_ip
        ]
        if host_flows:
            window_end = max(record.end_time for record in host_flows)
            window_start = window_end - self.window_seconds
            host_flows = [
                record
                for record in host_flows
                if record.end_time >= window_start or record.start_time >= window_start
            ]

        # ================= STAGE 1: telemetry -> 2s window -> graph -> TGNE ==========
        temp_attrs = self.extractor.compute_host_temporal_attributes(
            host_ip=target_ip,
            window_records=host_flows,
            window_duration=self.window_seconds,
        )
        MC.assert_last_dim("host_temporal_attrs", temp_attrs.shape, MC.HOST_TEMPORAL_ATTR_DIM)
        h_emb, graph_nodes, graph_edges = self._build_embedding(target_ip, host_flows)
        logger.debug(
            "[TGNE] embedding generated dim=%s |H|=%.4f nodes=%s edges=%s",
            h_emb.shape[-1], float(np.linalg.norm(h_emb)), graph_nodes, graph_edges,
        )

        feature_vector = np.concatenate([h_emb, temp_attrs]).astype(np.float32)
        MC.assert_last_dim("branch_a_feature_vector", feature_vector.shape, MC.BRANCH_A_INPUT_DIM)
        feature_history = self.feature_history_by_target.setdefault(target_ip, [])
        feature_history.append(
            torch.from_numpy(feature_vector).float().to(self.device)
        )
        if len(feature_history) > self.history_steps:
            feature_history.pop(0)
        self.feature_history = feature_history
        feature_history = list(feature_history)
        while len(feature_history) < self.history_steps:
            feature_history.insert(0, torch.zeros_like(feature_history[0]))
        x_tensor = torch.stack(feature_history).unsqueeze(0)

        # ================= STAGE 2: Branch A (current risk + technique) =============
        MC.assert_shape(
            "branch_a_input",
            tuple(x_tensor.shape),
            (1, MC.HISTORY_STEPS, MC.BRANCH_A_INPUT_DIM),
        )
        with torch.no_grad():
            branch_a_out = self.branch_a(x_tensor)
            risk_pred = branch_a_out["risk_score"]
            obs_logits = branch_a_out["technique_logits"]
            MC.assert_last_dim("branch_a_technique_logits", obs_logits.shape, MC.BRANCH_A_NUM_TECHNIQUES)
            MC.assert_last_dim(
                "branch_a_gradation_logits",
                branch_a_out["gradation_logits"].shape,
                MC.BRANCH_A_NUM_GRADATIONS,
            )
            obs_probs = torch.softmax(obs_logits, dim=-1)
            pred_class_idx = int(obs_probs.argmax(dim=-1).item())
            obs_confidence = float(obs_probs.reshape(-1)[pred_class_idx].item())
            obs_technique = self.technique_vocab[pred_class_idx]
            gradation_idx = int(branch_a_out["gradation_logits"].argmax(dim=-1).item())
            raw_risk = float(risk_pred.item()) if risk_pred.numel() == 1 else float(risk_pred.mean().item())
            # Single forward pass feeds every Branch A consumer (risk, technique,
            # gradation, stage_probabilities) — no duplicated inference anywhere.
            probs_vec = obs_probs.reshape(-1)
            stage_probs = {
                self.technique_vocab[i]: round(float(probs_vec[i].item()), 4)
                for i in range(len(self.technique_vocab))
                if float(probs_vec[i].item()) > 0.01
            }
        logger.debug(
            "[BRANCH_A] risk=%.4f technique=%s confidence=%.4f gradation=%s",
            raw_risk, obs_technique, obs_confidence, gradation_idx,
        )

        if is_mitigated:
            obs_risk = max(0.02, raw_risk * 0.15)
            obs_technique = "Benign"
        else:
            obs_risk = raw_risk

        curr_h = torch.from_numpy(h_emb).float().unsqueeze(0).to(self.device)
        h_state_history = self.h_state_history_by_target.setdefault(target_ip, [])
        h_state_history.append(curr_h)
        if len(h_state_history) > self.history_steps:
            h_state_history.pop(0)
        self.h_state_history = h_state_history
        while len(h_state_history) < self.history_steps:
            h_state_history.insert(0, torch.zeros_like(curr_h))
        h_seq = torch.stack(h_state_history, dim=1)

        # ================= STAGE 3: Branch B (future latent trajectory) =============
        MC.assert_shape(
            "branch_b_input",
            tuple(h_seq.shape),
            (1, MC.HISTORY_STEPS, MC.BRANCH_B_D_LATENT),
        )
        with torch.no_grad():
            h_future, radii = self.wdt.rollout_with_uncertainty(
                h_seq, K=self.forecast_steps, delta_t_step=self.window_seconds, stabilize_horizon=True
            )
            conf_radii = [float(r.item()) if hasattr(r, "item") else float(r) for r in radii]
            MC.assert_shape(
                "branch_b_future_latent",
                tuple(h_future.shape),
                (1, MC.FORECAST_STEPS, MC.BRANCH_B_D_LATENT),
            )

            step_risks, _ = self.risk_head.forward_trajectory(h_future)
            MC.assert_shape("branch_b_step_risks", tuple(step_risks.shape), (1, MC.FORECAST_STEPS))
            fut_risks = [float(r) for r in step_risks.cpu().squeeze(0).numpy().tolist()]
            latent_norms = [
                round(float(v), 4)
                for v in torch.linalg.norm(h_future[0], dim=-1).cpu().numpy().tolist()
            ]
        logger.debug(
            "[BRANCH_B] future_latent_shape=%s risk_trajectory=%s",
            tuple(h_future.shape), [round(r, 4) for r in fut_risks],
        )

        # ================= STAGE 4: DeepOP (future technique trajectory) ============
        # DeepOP's cross-attention memory is *exactly* the Branch B rollout tensor —
        # no independent/synthetic input is constructed anywhere on this path.
        MC.assert_last_dim("deepop_memory_input", h_future.shape, MC.DEEPOP_D_LATENT)
        with torch.no_grad():
            observed_tactic = TECHNIQUE_TO_MITRE.get(
                obs_technique,
                ("Benign" if obs_technique == "Benign" else "Recon", "", "", ""),
            )[0]
            coarse_cat, coarse_technique = self.consolidate_network_technique(
                observed_tactic,
                obs_technique,
            )
            obs_token_id = self.vocab.encode(coarse_cat, coarse_technique)
            if not 0 <= obs_token_id < self.vocab.vocab_size:
                raise MC.ContractViolation(
                    f"[CONTRACT] observed token id {obs_token_id} outside DeepOP vocabulary "
                    f"[0, {self.vocab.vocab_size})"
                )
            obs_token_tensor = torch.tensor(
                [obs_token_id], dtype=torch.long, device=self.device
            )
            pred_tokens, decoded_names, _atk_probs, token_probs = self.deepop.forecast_sequence(
                h_future,
                max_steps=self.forecast_steps,
                observed_token=obs_token_tensor,
                return_probs=True,
            )
            MC.assert_shape("deepop_tokens", tuple(pred_tokens.shape), (1, MC.FORECAST_STEPS))

        forecast_techniques: List[str] = []
        for item in (decoded_names[0] if decoded_names else []):
            if isinstance(item, tuple):
                c, t = item
                forecast_techniques.append(f"{c}.{t}" if t else str(c))
            else:
                forecast_techniques.append(str(item))
        if len(forecast_techniques) != self.forecast_steps:
            raise MC.ContractViolation(
                f"[CONTRACT] DeepOP returned {len(forecast_techniques)} decoded steps, "
                f"expected {self.forecast_steps}"
            )
        deepop_conf = [round(float(p), 4) for p in (token_probs[0] if token_probs else [])]
        logger.debug("[DEEPOP] future_technique=%s confidence=%s", forecast_techniques, deepop_conf)

        deepop_observed_token = f"{coarse_cat}.{coarse_technique}" if coarse_technique else coarse_cat

        if is_mitigated:
            fut_risks = [max(0.01, r * 0.1) for r in fut_risks]
            forecast_techniques = ["Benign"] * self.forecast_steps

        # Clamp risks
        obs_risk = float(np.clip(obs_risk, 0.0, 1.0))
        fut_risks = [float(np.clip(r, 0.0, 1.0)) for r in fut_risks]
        max_future = max(fut_risks) if fut_risks else obs_risk

        mitre = TECHNIQUE_TO_MITRE.get(
            obs_technique,
            ("Unknown", obs_technique, "TA0000", "Model-predicted technique"),
        )
        alert = (not is_mitigated) and (
            obs_risk >= self.alert_threshold or max_future >= self.alert_threshold
        )

        explain = self._explain(feature_vector)
        inf_ms = (time.perf_counter() - t0) * 1000.0
        now_ts = time.time()

        forecast_points = [
            ForecastPoint(
                horizon_seconds=(i + 1) * self.window_seconds,
                risk=round(fut_risks[i], 4),
                confidence=round(max(0.0, 1.0 - conf_radii[i]), 4)
                if i < len(conf_radii)
                else None,
                predicted_stage=forecast_techniques[i],
                technique_confidence=deepop_conf[i] if i < len(deepop_conf) else None,
            )
            for i in range(self.forecast_steps)
        ]

        # Per-stage provenance: every number below came out of THIS inference pass.
        stages = StageProvenance(
            tgne=TgneStage(
                embedding_dim=int(h_emb.shape[-1]),
                embedding_norm=round(float(np.linalg.norm(h_emb)), 4),
                graph_nodes=graph_nodes,
                graph_edges=graph_edges,
                n_neighbors=MC.TGNE_N_NEIGHBORS,
            ),
            branch_a=BranchAStage(
                input_dim=MC.BRANCH_A_INPUT_DIM,
                history_steps=self.history_steps,
                risk=round(obs_risk, 4),
                technique=obs_technique,
                confidence=round(obs_confidence, 4),
                gradation=gradation_idx,
                target_host=target_ip or None,
            ),
            branch_b=BranchBStage(
                latent_dim=MC.BRANCH_B_D_LATENT,
                forecast_steps=self.forecast_steps,
                forecast_horizon_seconds=self.forecast_steps * self.window_seconds,
                risk_trajectory=[round(r, 4) for r in fut_risks],
                latent_trajectory_norms=latent_norms,
                max_future_risk=round(max_future, 4),
            ),
            deepop=DeepOpStage(
                latent_dim=MC.DEEPOP_D_LATENT,
                vocab_size=self.vocab.vocab_size,
                observed_token=deepop_observed_token,
                technique_trajectory=list(forecast_techniques),
                confidence=deepop_conf,
            ),
        )

        logger.info(
            "[INFERENCE] window=%s host=%s | TGNE |H|=%.3f (%sn/%se) -> BRANCH_A risk=%.4f tech=%s "
            "(p=%.2f) -> BRANCH_B K=%s max_future=%.4f -> DEEPOP next=%s | %.1fms",
            window_id, target_ip, stages.tgne.embedding_norm, graph_nodes, graph_edges,
            obs_risk, obs_technique, obs_confidence, self.forecast_steps, max_future,
            forecast_techniques[0] if forecast_techniques else "-", inf_ms,
        )

        focus_ips, focus_edges = self._focus_binding(target_ip, flows)
        return PredictionEvent(
            type="prediction",
            mode="LIVE",
            timestamp=datetime.fromtimestamp(now_ts).isoformat() + "Z",
            wall_clock=time.strftime("%H:%M:%S", time.localtime(now_ts)),
            model=model_metadata(),
            state=StateMetadata(
                window_id=int(window_id),
                sequence_ready=len(self.h_state_history) >= self.history_steps,
                packet_count=int(packet_count or sum(r.total_packets for r in flows)),
                active_flows=active_flows if active_flows is not None else len(flows),
                pipeline_latency_ms=float(pipeline_latency_ms),
                buffer_length=len(self.h_state_history),
            ),
            prediction=PredictionData(
                risk=round(obs_risk, 4),
                max_future_risk=round(max_future, 4),
                hazard_score=round(max_future, 4),
                malicious_confidence=round(obs_risk, 4),
                precursor_confidence=round(max(0.0, max_future - obs_risk), 4),
                alert=alert,
                alert_level=self._alert_level(max(obs_risk, max_future)),
                threshold=self.alert_threshold,
                predicted_stage=obs_technique,
                mitre_tactic=mitre[0],
                mitre_technique=mitre[1],
                mitre_tactic_id=mitre[2],
                mitre_description=mitre[3],
                stage_probabilities=stage_probs,
            ),
            forecast=forecast_points,
            stages=stages,
            explainability=explain,
            latency=LatencyData(
                telemetry_ms=round(float(pipeline_latency_ms), 2),
                inference_ms=round(inf_ms, 2),
                total_ms=round(float(pipeline_latency_ms) + inf_ms, 2),
            ),
            early_warning=EarlyWarningData(
                is_alert=alert,
                alert_timestamp=now_ts if alert else None,
                lead_time_seconds=self.forecast_steps * self.window_seconds if alert else None,
                target_milestone_desc=forecast_techniques[0] if alert else None,
            ),
            attack_active=attack_active,
            attack_phase=attack_phase,
            focus_ips=focus_ips,
            focus_edges=focus_edges,
            target_ip=target_ip or None,
        )

    @staticmethod
    def _focus_binding(
        target_ip: str, flows: List[UnifiedFlowRecord]
    ) -> tuple[List[str], List[FocusEdge]]:
        """Bind prediction highlight to observed IPs/edges (no lab node IDs)."""
        ips: List[str] = []
        if target_ip:
            ips.append(target_ip)
        edge_keys = set()
        edges: List[FocusEdge] = []
        for r in flows or []:
            for ip in (r.src_ip, r.dst_ip):
                if ip and ip not in ips:
                    # Prefer peers of the scored host; cap list for UI
                    if not target_ip or ip == target_ip or r.src_ip == target_ip or r.dst_ip == target_ip:
                        ips.append(ip)
            if target_ip and (r.src_ip == target_ip or r.dst_ip == target_ip):
                key = (r.src_ip, r.dst_ip)
                if key not in edge_keys and r.src_ip and r.dst_ip:
                    edge_keys.add(key)
                    edges.append(FocusEdge(src=r.src_ip, dst=r.dst_ip))
        return ips[:12], edges[:24]


def select_primary_target(
    flows: List[UnifiedFlowRecord],
    fallback: Optional[str] = None,
) -> str:
    """Pick the most relevant host to score — site CIDRs + activity, not lab hardcoding."""
    from control_backend.site_config import get_site_config, select_primary_target_ip

    site = get_site_config()
    return select_primary_target_ip(
        flows,
        site=site,
        fallback=fallback if fallback is not None else (
            site.asset_ips()[0] if site.asset_ips() else ""
        ),
    )


def flows_from_span_dicts(raw_flows: List[Dict[str, Any]]) -> List[UnifiedFlowRecord]:
    """Convert SPAN flow snapshots → UnifiedFlowRecord with zero label leakage."""
    out: List[UnifiedFlowRecord] = []
    for f in raw_flows or []:
        try:
            out.append(
                UnifiedFlowRecord(
                    src_ip=str(f.get("src_ip", "")),
                    dst_ip=str(f.get("dst_ip", "")),
                    src_port=int(f.get("src_port", 0)),
                    dst_port=int(f.get("dst_port", 0)),
                    protocol=int(f.get("protocol", 6)),
                    start_time=float(f.get("start_time", time.time())),
                    end_time=float(f.get("end_time", time.time())),
                    fwd_bytes=int(f.get("fwd_bytes", 0)),
                    bwd_bytes=int(f.get("bwd_bytes", 0)),
                    fwd_packets=int(f.get("fwd_packets", 0)),
                    bwd_packets=int(f.get("bwd_packets", 0)),
                    raw_label="UNLABELED",
                    raw_label_source="SPAN_LIVE",
                    is_attack=False,
                    coarse_category=CoarseCategory.UNKNOWN.value,
                    attck_technique_ids=[],
                    metadata={"source": "span_live"},
                )
            )
        except Exception:
            continue
    return out


# Module singleton used by tests and telemetry service
model_adapter = AntigravityModelAdapter()
