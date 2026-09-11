"""
Data Unification Package.

Provides unified schema, label resolution, and dataset adapters for
CIC-IDS2017, CIC-IDS2018, CTU-13, and Warden.

Imports are LAZY (PEP 562). The offline dataset adapters pull in heavy,
training-only dependencies (pandas/pyarrow), and eagerly importing them here
meant that `from data_unification.unified_schema import UnifiedFlowRecord` —
which the live SOC backend does on startup — failed on any machine without
pyarrow installed. The public API is unchanged: `from data_unification import
CTU13Adapter` still works and imports the adapter on first use.
"""

import importlib
from typing import Any

_EXPORTS = {
    # name -> submodule
    "UnifiedFlowRecord": "unified_schema",
    "LabelSource": "unified_schema",
    "CoarseCategory": "unified_schema",
    "records_to_dataframe": "unified_schema",
    "LabelResolver": "label_resolver",
    "get_default_resolver": "label_resolver",
    "CIC2017Adapter": "cic2017_adapter",
    "CIC2018Adapter": "cic2018_adapter",
    "CTU13Adapter": "ctu13_adapter",
    "WardenAdapter": "warden_adapter",
    "AuthEventRecord": "auth_log_adapter",
    "AuthLogAdapter": "auth_log_adapter",
    "AuthEventType": "auth_log_adapter",
    "AuthLogSource": "auth_log_adapter",
    "BoundedHostSlidingBuffer": "multi_dataset_stream",
    "BehavioralFlowFingerprinter": "behavioral_fingerprint",
    "BehavioralProfile": "behavioral_fingerprint",
    "FlowBehavioralResult": "behavioral_fingerprint",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'data_unification' has no attribute '{name}'")
    mod = importlib.import_module(f"data_unification.{module}")
    value = getattr(mod, name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals()) + __all__)
