#!/bin/bash
# Monitor experiment progress by tailing the latest log file
#
# Usage:
#   bash scripts/monitor_experiments.sh

LOG_DIR="experimental_results/logs"

if [ ! -d "${LOG_DIR}" ]; then
    echo "❌ Log directory not found: ${LOG_DIR}"
    echo "   Have you started the experiments yet?"
    exit 1
fi

# Find the latest log file
LATEST_LOG=$(ls -t "${LOG_DIR}"/run_all_experiments_*.log 2>/dev/null | head -1)

if [ -z "${LATEST_LOG}" ]; then
    echo "❌ No log files found in ${LOG_DIR}"
    echo "   Have you started the experiments yet?"
    exit 1
fi

echo "=========================================="
echo "Monitoring: ${LATEST_LOG}"
echo "=========================================="
echo ""
echo "Press Ctrl+C to stop monitoring"
echo ""

# Tail the log file
tail -f "${LATEST_LOG}"
