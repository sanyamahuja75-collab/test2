#!/bin/bash
# CyberWorld MVP demo — full TGNE -> Branch A -> Branch B -> DeepOP -> dashboard
# chain driven by a real PCAP replay. No mirror NIC, no root, no Containerlab.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PCAP="${CYBERWORLD_REPLAY_PCAP:-${ROOT_DIR}/captures/live.pcap}"

echo "================================================================="
echo "CYBERWORLD MVP DEMO  (PCAP replay -> Dual-Branch + DeepOP -> SOC)"
echo "  capture : ${PCAP}"
echo "  console : http://localhost:8000"
echo "  steps   : in the UI click  START SENSOR  ->  START ML"
echo "================================================================="

echo
echo "[1/2] Pre-flight: checkpoints vs model_contract.py"
python3 scripts/ensure_checkpoints.py

echo
echo "[2/2] Launching SOC console in replay mode..."
exec python3 run_dashboard.py --site local-default --replay "${PCAP}" "$@"
