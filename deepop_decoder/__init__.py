"""
DeepOP-Style ATT&CK CWA Forecasting Decoder Package.

Imports are LAZY (PEP 562) so the joint vocabulary can be imported (e.g. by the
offline contract verifier) without pulling in torch. Public API is unchanged.
"""

import importlib
from typing import Any

_EXPORTS = {
    "JointAttackVocab": "joint_vocab",
    "get_joint_vocab": "joint_vocab",
    "consolidate_network_technique": "joint_vocab",
    "PAD_TOKEN": "joint_vocab",
    "BOS_TOKEN": "joint_vocab",
    "EOS_TOKEN": "joint_vocab",
    "CausalWindowAttention": "cwa",
    "CWADecoderLayer": "forecast_decoder",
    "DeepOPForecastDecoder": "forecast_decoder",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'deepop_decoder' has no attribute '{name}'")
    value = getattr(importlib.import_module(f"deepop_decoder.{module}"), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
