"""
Branch A technique vocabulary — the ordered label space of the trained
`technique_head` in saved_models/branch_a/branch_a_lstm.pt (14 classes).

Kept in its own torch-free module so the vocabulary can be imported by the
contract checks, the offline verifier and tests without pulling in torch.
Index order is part of the checkpoint contract: DO NOT reorder or insert.
"""

TECHNIQUE_VOCAB = [
    "Benign",
    "T1046",       # Network Service Discovery
    "T1595",       # Active Scanning
    "T1110",       # Brute Force
    "T1190",       # Exploit Public-Facing Application
    "T1189",       # Drive-by Compromise
    "T1071",       # Application Layer Protocol (C2)
    "T1071.001",   # Web / IRC Protocols
    "T1568.001",   # Fast Flux DNS
    "T1204",       # User Execution
    "T1005",       # Data from Local System / Heartbleed
    "T1498",       # Network Denial of Service
    "T1498.001",   # Direct Network Flood
    "T1020",       # Automated Exfiltration
]

TECH_TO_IDX = {t: i for i, t in enumerate(TECHNIQUE_VOCAB)}

GRADATION_LEVELS = {
    "Benign": 0,
    "Recon": 1,
    "InitialAccess": 2,
    "Execution": 2,
    "C2": 3,
    "LateralMovement": 3,
    "Exfiltration": 3,
    "Impact": 3,
    "Unknown": 1,
}
