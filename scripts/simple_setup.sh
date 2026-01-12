#!/bin/bash
# Simple Setup Script (No MinIO)
# Just run this to get started!

set -e  # Exit on error

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║     Bachelor Thesis Setup - Simple Start (No MinIO)       ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Get project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"
cd "$PROJECT_DIR"

echo "📍 Working directory: $PROJECT_DIR"
echo ""

# Step 1: Check if .env exists
echo "════════════════════════════════════════════════════════════"
echo "Step 1: Configuration"
echo "════════════════════════════════════════════════════════════"

if [ -f .env ]; then
    echo "✓ .env file already exists"
    echo ""
    read -p "Do you want to recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp .env.jupyterhub .env
        echo "✓ Created new .env from template"
    fi
else
    cp .env.jupyterhub .env
    echo "✓ Created .env from template"
fi

echo ""
echo "⚠️  IMPORTANT: Edit .env file to set your username"
echo ""
echo "Current USER setting:"
grep "^USER=" .env || echo "USER not set"
echo ""
read -p "Press Enter to edit .env now (or Ctrl+C to skip)..." -r
echo ""

# Open editor
if command -v nano &> /dev/null; then
    nano .env
elif command -v vim &> /dev/null; then
    vim .env
else
    echo "Please edit .env manually and set your username"
    echo "Find line: USER=\${USER:-student01}"
    echo "Change to: USER=\${USER:-your_actual_username}"
fi

echo ""
echo "✓ Configuration complete"
echo ""

# Step 2: Install Python environment
echo "════════════════════════════════════════════════════════════"
echo "Step 2: Installing Python Environment"
echo "════════════════════════════════════════════════════════════"
echo "This will take 5-10 minutes..."
echo ""

if [ -d venv ]; then
    echo "⚠️  Virtual environment already exists"
    echo ""
    read -p "Reinstall? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash scripts/setup_env.sh --force
    else
        echo "✓ Keeping existing environment"
    fi
else
    bash scripts/setup_env.sh
fi

echo ""
echo "✓ Python environment ready"
echo ""

# Step 3: Activate and verify
echo "════════════════════════════════════════════════════════════"
echo "Step 3: Verifying Setup"
echo "════════════════════════════════════════════════════════════"
echo ""

# Source activation script
if [ -f activate_env.sh ]; then
    source activate_env.sh
    echo "✓ Environment activated"
else
    echo "⚠️  activate_env.sh not found, trying venv directly"
    source venv/bin/activate
fi

echo ""
echo "Running preflight checks..."
bash scripts/preflight.sh

echo ""
echo "✓ Setup verification complete"
echo ""

# Step 4: Offer to run smoke test
echo "════════════════════════════════════════════════════════════"
echo "Step 4: Run Smoke Test (Optional)"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "The smoke test verifies your entire pipeline works"
echo "It takes 2-3 minutes and needs NO data download"
echo ""
read -p "Run smoke test now? (Y/n): " -n 1 -r
echo

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "Running smoke test..."
    echo "════════════════════════════════════════════════════════════"
    bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml
    
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "✅ SMOKE TEST COMPLETED!"
    echo "════════════════════════════════════════════════════════════"
else
    echo ""
    echo "Skipped smoke test. You can run it later with:"
    echo "  bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "🎉 SETUP COMPLETE!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "What you have now:"
echo "  ✓ Python environment installed"
echo "  ✓ All dependencies ready"
echo "  ✓ Pipeline tested (if you ran smoke test)"
echo ""
echo "What you can do:"
echo ""
echo "  1. Run more smoke tests:"
echo "     source activate_env.sh"
echo "     bash scripts/run_experiment_simple.sh configs/experiments/smoke_test.yaml"
echo ""
echo "  2. Read about minimal data requirements:"
echo "     cat MINIMAL_DATA_SETUP.md"
echo ""
echo "  3. When ready, download subj01 data (40GB):"
echo "     - Register at https://naturalscenesdataset.org/"
echo "     - Download subj01 folder only"
echo "     - See MINIMAL_DATA_SETUP.md for details"
echo ""
echo "  4. Run real experiments:"
echo "     source activate_env.sh"
echo "     bash scripts/run_experiment_simple.sh configs/experiments/jupyterhub_quickstart.yaml"
echo ""
echo "Next steps:"
echo "  📖 Read: SIMPLE_START.md"
echo "  📖 Read: MINIMAL_DATA_SETUP.md"
echo "  📖 Keep handy: QUICK_REFERENCE.md"
echo ""
echo "You're ready to go! 🚀"
echo ""
