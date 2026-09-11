"""
Branch A: GNN/LSTM Multi-Task Attack-Sequence Prediction Package.

Imports are LAZY (PEP 562) so that torch-free consumers (contract checks, the
offline verifier, unit tests) can `from branch_a_gnn_lstm.technique_vocab import
TECHNIQUE_VOCAB` without importing torch. The public API is unchanged.
"""

import importlib
from typing import Any

_EXPORTS = {
    "HostSequenceDataset": "sequence_dataset",
    "create_host_sequence_samples": "sequence_dataset",
    "TECHNIQUE_VOCAB": "technique_vocab",
    "TECH_TO_IDX": "technique_vocab",
    "GRADATION_LEVELS": "technique_vocab",
    "TemporalSelfAttention": "attention",
    "MultiTaskLSTM": "lstm_multitask",
    "MultiTaskUncertaintyLoss": "lstm_multitask",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'branch_a_gnn_lstm' has no attribute '{name}'")
    value = getattr(importlib.import_module(f"branch_a_gnn_lstm.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
