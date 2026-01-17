#!/bin/bash
# Verify research implementation is complete and ready

set -e

echo "="
echo "Research Implementation Verification"
echo "="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

check_pass() {
    echo -e "${GREEN}✓${NC} $1"
}

check_fail() {
    echo -e "${RED}✗${NC} $1"
    FAILED=1
}

FAILED=0

echo "1. Checking Experiment Configs..."
for exp in exp0_baseline exp1_preproc exp2_queue exp3_gaussian_nll exp4_gaussian_nce exp5_kl_anneal exp6_whiten; do
    if [ -f "configs/experiments/${exp}.yaml" ]; then
        check_pass "configs/experiments/${exp}.yaml"
    else
        check_fail "configs/experiments/${exp}.yaml MISSING"
    fi
done
echo ""

echo "2. Checking Core Components..."
components=(
    "src/fmri2img/embedding_preproc.py"
    "src/fmri2img/models/unified_model.py"
    "src/fmri2img/contrastive/queue.py"
    "src/fmri2img/losses/infonce_queue.py"
    "src/fmri2img/losses/gaussian_nll.py"
    "src/fmri2img/losses/gaussian_nce.py"
    "src/fmri2img/training/kl_schedule.py"
    "src/fmri2img/eval/embedding_metrics.py"
    "src/fmri2img/eval/probabilistic_metrics.py"
)

for comp in "${components[@]}"; do
    if [ -f "${comp}" ]; then
        check_pass "${comp}"
    else
        check_fail "${comp} MISSING"
    fi
done
echo ""

echo "3. Checking Scripts..."
scripts=(
    "scripts/build_embedding_preproc.py"
    "scripts/build_all_preprocessors.sh"
    "scripts/train_unified.py"
    "scripts/eval_stage1_embedding.py"
    "scripts/run_all_experiments.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "${script}" ]; then
        # Check if .sh files are executable
        if [[ "${script}" == *.sh ]]; then
            if [ -x "${script}" ]; then
                check_pass "${script} (executable)"
            else
                check_fail "${script} (not executable)"
            fi
        else
            check_pass "${script}"
        fi
    else
        check_fail "${script} MISSING"
    fi
done
echo ""

echo "4. Checking Documentation..."
docs=(
    "READY_TO_RUN.md"
    "EXPERIMENT_GUIDE.md"
    "README_RESEARCH_UPGRADE.md"
    "IMPLEMENTATION_COMPLETE.md"
    "docs/paper_outline.md"
    "docs/evaluation_protocol.md"
    "docs/ablation_plan.md"
)

for doc in "${docs[@]}"; do
    if [ -f "${doc}" ]; then
        check_pass "${doc}"
    else
        check_fail "${doc} MISSING"
    fi
done
echo ""

echo "5. Checking Tests..."
if [ -f "tests/test_research_components.py" ]; then
    check_pass "tests/test_research_components.py"
else
    check_fail "tests/test_research_components.py MISSING"
fi
echo ""

echo "6. Running Tests..."
if python -m pytest tests/test_research_components.py -v --tb=short 2>&1 | grep -q "passed"; then
    check_pass "Tests passed"
else
    check_fail "Tests failed or pytest not available"
fi
echo ""

echo "="
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "You're ready to run experiments!"
    echo ""
    echo "Next steps:"
    echo "  1. Build preprocessors: bash scripts/build_all_preprocessors.sh"
    echo "  2. Run experiments: bash scripts/run_all_experiments.sh 0"
    echo "  3. Follow EXPERIMENT_GUIDE.md for complete workflow"
else
    echo -e "${RED}✗ SOME CHECKS FAILED${NC}"
    echo ""
    echo "Please fix the missing components before proceeding."
fi
echo "="
