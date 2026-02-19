#!/bin/bash
# Check the status of all experiments
#
# Usage:
#   bash scripts/check_experiment_status.sh

echo "=========================================="
echo "Experiment Status Summary"
echo "=========================================="
echo ""

# Array of experiments
experiments=(
    "exp0_baseline"
    "exp1_preproc"
    "exp2_queue"
    "exp3_gaussian_nll"
    "exp4_gaussian_nce"
    "exp5_kl_anneal"
    "exp6_whiten"
)

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

completed=0
failed=0
pending=0

for exp in "${experiments[@]}"; do
    exp_dir="experimental_results/${exp}"
    checkpoint="${exp_dir}/checkpoints/best.ckpt"
    metrics="${exp_dir}/evaluation/metrics.json"
    
    if [ -f "${metrics}" ]; then
        # Completed
        if command -v jq &> /dev/null; then
            R1=$(jq -r '.retrieval."R@1" // "N/A"' "${metrics}" 2>/dev/null)
            oracle=$(jq -r '.oracle.passed // "N/A"' "${metrics}" 2>/dev/null)
            
            if [ "${oracle}" == "false" ]; then
                echo -e "${YELLOW}⚠️  ${exp}${NC}: R@1=${R1} (Oracle FAILED)"
            else
                echo -e "${GREEN}✅ ${exp}${NC}: R@1=${R1}"
            fi
        else
            echo -e "${GREEN}✅ ${exp}${NC}: Complete (install jq for metrics)"
        fi
        completed=$((completed + 1))
        
    elif [ -f "${checkpoint}" ]; then
        # Trained but not evaluated
        echo -e "${YELLOW}🔄 ${exp}${NC}: Trained, evaluation pending"
        completed=$((completed + 1))
        
    elif [ -d "${exp_dir}" ]; then
        # In progress or failed
        if [ -f "${exp_dir}/logs/train.log" ]; then
            last_line=$(tail -1 "${exp_dir}/logs/train.log" 2>/dev/null)
            echo -e "${YELLOW}🔄 ${exp}${NC}: Training in progress..."
            echo "   Last: ${last_line:0:80}"
        else
            echo -e "${RED}❌ ${exp}${NC}: Failed or incomplete"
            failed=$((failed + 1))
        fi
        
    else
        # Not started
        echo -e "⏳ ${exp}: Pending"
        pending=$((pending + 1))
    fi
done

echo ""
echo "=========================================="
echo "Summary:"
echo "  ✅ Completed: ${completed}/7"
echo "  ❌ Failed: ${failed}/7"
echo "  ⏳ Pending: ${pending}/7"
echo "=========================================="
echo ""

# Show GPU status if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s (%s): %s util, %s / %s memory, %s temp\n", $1, $2, $3, $4, $5, $6}'
    echo ""
fi

# Show latest log
LOG_DIR="experimental_results/logs"
if [ -d "${LOG_DIR}" ]; then
    LATEST_LOG=$(ls -t "${LOG_DIR}"/run_all_experiments_*.log 2>/dev/null | head -1)
    if [ -n "${LATEST_LOG}" ]; then
        echo "Latest log: ${LATEST_LOG}"
        echo "  Monitor: bash scripts/monitor_experiments.sh"
        echo "  View: tail -f ${LATEST_LOG}"
    fi
fi
