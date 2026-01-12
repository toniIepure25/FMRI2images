#!/bin/bash
################################################################################
# GPU Availability Monitor and Auto-Trainer
################################################################################
#
# This script monitors GPU memory and automatically starts training when
# sufficient memory becomes available. Perfect for shared GPU environments.
#
# Usage:
#   ./wait_and_train.sh [OPTIONS]
#
# Options:
#   --min-memory GB      Minimum free GPU memory required (default: 5)
#   --check-interval SEC Seconds between checks (default: 300)
#   --max-wait MIN       Maximum wait time in minutes (default: 480 = 8 hours)
#   --notify             Send desktop notification when ready
#   --auto-start         Automatically start training when ready
#   --help               Show this help
#
################################################################################

set -u

# Default configuration
MIN_FREE_MEMORY_GB=5
CHECK_INTERVAL_SEC=300  # 5 minutes
MAX_WAIT_MIN=480        # 8 hours
SEND_NOTIFY=false
AUTO_START=false
CONFIG_FILE="experiments/ultimate_novel_subj01.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --min-memory)
            MIN_FREE_MEMORY_GB="$2"
            shift 2
            ;;
        --check-interval)
            CHECK_INTERVAL_SEC="$2"
            shift 2
            ;;
        --max-wait)
            MAX_WAIT_MIN="$2"
            shift 2
            ;;
        --notify)
            SEND_NOTIFY=true
            shift
            ;;
        --auto-start)
            AUTO_START=true
            shift
            ;;
        --help)
            head -n 18 "$0" | tail -n 15
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

################################################################################
# Functions
################################################################################

get_gpu_free_memory() {
    # Returns free GPU memory in MB
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0"
}

get_gpu_info() {
    local free_mb=$(get_gpu_free_memory)
    local total_mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")
    local used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")
    local util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2>/dev/null || echo "0")
    
    echo "$free_mb $total_mb $used_mb $util"
}

mb_to_gb() {
    echo "scale=2; $1 / 1024" | bc
}

send_notification() {
    local title="$1"
    local message="$2"
    
    if command -v notify-send &> /dev/null; then
        notify-send "$title" "$message"
    fi
    
    # Also try terminal bell
    echo -e "\a"
}

print_status() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local free_mb=$1
    local total_mb=$2
    local used_mb=$3
    local util=$4
    
    local free_gb=$(mb_to_gb $free_mb)
    local total_gb=$(mb_to_gb $total_mb)
    local used_gb=$(mb_to_gb $used_mb)
    
    local color=$RED
    if (( $(echo "$free_gb >= $MIN_FREE_MEMORY_GB" | bc -l) )); then
        color=$GREEN
    elif (( $(echo "$free_gb >= $(echo "$MIN_FREE_MEMORY_GB * 0.7" | bc)" | bc -l) )); then
        color=$YELLOW
    fi
    
    echo -e "${BLUE}[$timestamp]${NC} GPU: ${color}${free_gb}GB free${NC} / ${total_gb}GB total | ${used_gb}GB used (${util}% util)"
}

get_process_info() {
    echo -e "${CYAN}Current GPU processes:${NC}"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | \
    while IFS=, read -r pid name memory; do
        echo "  PID $pid: $name (${memory} MB)"
    done
}

recommend_batch_size() {
    local free_mb=$1
    
    if (( free_mb < 3000 )); then
        echo "2"
    elif (( free_mb < 5000 )); then
        echo "4"
    elif (( free_mb < 8000 )); then
        echo "8"
    elif (( free_mb < 12000 )); then
        echo "16"
    else
        echo "32"
    fi
}

################################################################################
# Main Monitoring Loop
################################################################################

main() {
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}${BOLD}  GPU Availability Monitor${NC}"
    echo -e "${CYAN}${BOLD}═══════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo -e "${CYAN}Configuration:${NC}"
    echo "  Minimum free memory: ${MIN_FREE_MEMORY_GB} GB"
    echo "  Check interval: ${CHECK_INTERVAL_SEC} seconds"
    echo "  Maximum wait time: ${MAX_WAIT_MIN} minutes"
    echo "  Auto-start training: $([[ "$AUTO_START" = true ]] && echo "Yes" || echo "No")"
    echo ""
    
    # Check if nvidia-smi exists
    if ! command -v nvidia-smi &> /dev/null; then
        echo -e "${RED}Error: nvidia-smi not found. No GPU available.${NC}"
        exit 1
    fi
    
    local start_time=$(date +%s)
    local max_wait_sec=$((MAX_WAIT_MIN * 60))
    local check_count=0
    
    echo -e "${YELLOW}Monitoring GPU... Press Ctrl+C to stop${NC}"
    echo ""
    
    while true; do
        check_count=$((check_count + 1))
        
        # Get GPU info
        read -r free_mb total_mb used_mb util <<< $(get_gpu_info)
        
        # Print status
        print_status "$free_mb" "$total_mb" "$used_mb" "$util"
        
        # Check if enough memory is available
        local free_gb=$(mb_to_gb $free_mb)
        if (( $(echo "$free_gb >= $MIN_FREE_MEMORY_GB" | bc -l) )); then
            echo ""
            echo -e "${GREEN}${BOLD}🎉 GPU HAS SUFFICIENT MEMORY!${NC}"
            echo -e "${GREEN}   Free: ${free_gb}GB (required: ${MIN_FREE_MEMORY_GB}GB)${NC}"
            echo ""
            
            # Show process info
            get_process_info
            echo ""
            
            # Recommend batch size
            local batch=$(recommend_batch_size $free_mb)
            echo -e "${CYAN}Recommended configuration:${NC}"
            echo "  batch_size: $batch"
            echo "  gradient_accumulation_steps: $((32 / batch))"
            echo "  Effective batch size: 32"
            echo ""
            
            # Send notification
            if [[ "$SEND_NOTIFY" = true ]]; then
                send_notification "GPU Ready!" "Free memory: ${free_gb}GB"
            fi
            
            # Auto-start training
            if [[ "$AUTO_START" = true ]]; then
                echo -e "${YELLOW}Auto-starting training in 5 seconds...${NC}"
                echo -e "${YELLOW}Press Ctrl+C to cancel${NC}"
                sleep 5
                
                echo -e "${GREEN}Starting training...${NC}"
                source venv/bin/activate
                source .env
                python scripts/train_ultimate_novel.py --config "$CONFIG_FILE"
            else
                echo -e "${CYAN}Ready to train! Run:${NC}"
                echo -e "${BLUE}  source venv/bin/activate && source .env${NC}"
                echo -e "${BLUE}  python scripts/train_ultimate_novel.py --config $CONFIG_FILE${NC}"
            fi
            
            exit 0
        fi
        
        # Check timeout
        local elapsed=$(($(date +%s) - start_time))
        if (( elapsed >= max_wait_sec )); then
            echo ""
            echo -e "${RED}Maximum wait time reached (${MAX_WAIT_MIN} minutes)${NC}"
            echo -e "${YELLOW}GPU still not available. Consider:${NC}"
            echo "  1. Waiting longer (increase --max-wait)"
            echo "  2. Using a different GPU"
            echo "  3. Training with smaller batch size"
            exit 1
        fi
        
        # Show progress every 10 checks
        if (( check_count % 10 == 0 )); then
            local elapsed_min=$((elapsed / 60))
            local remaining_min=$(((max_wait_sec - elapsed) / 60))
            echo -e "${CYAN}  Progress: ${elapsed_min} min elapsed, ${remaining_min} min remaining${NC}"
            get_process_info
            echo ""
        fi
        
        # Wait before next check
        sleep "$CHECK_INTERVAL_SEC"
    done
}

# Trap Ctrl+C
trap 'echo -e "\n${YELLOW}Monitoring stopped by user${NC}"; exit 130' INT

# Run main
main
