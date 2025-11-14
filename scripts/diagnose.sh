#!/bin/bash
# filepath: scripts/diagnose.sh
# Diagnose common issues

echo "🔍 Running diagnostics..."
echo ""

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not activated!"
    echo "   Attempting to activate .venv..."
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        echo "   ✅ Activated .venv"
    else
        echo "   ❌ .venv not found. Please run: python3 -m venv .venv && source .venv/bin/activate"
        exit 1
    fi
else
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
fi
echo ""

# Check Python environment
echo "1. Python Environment:"
which python3
python3 --version
echo ""

# Check installed packages
echo "2. Key Packages:"
python3 -c "import torch; print(f'   PyTorch: {torch.__version__}')" 2>/dev/null || echo "   ❌ PyTorch not installed"
python3 -c "import pandas; print(f'   Pandas: {pandas.__version__}')" 2>/dev/null || echo "   ❌ Pandas not installed"
python3 -c "import pyarrow; print(f'   PyArrow: {pyarrow.__version__}')" 2>/dev/null || echo "   ❌ PyArrow not installed"
python3 -c "import diffusers; print(f'   Diffusers: {diffusers.__version__}')" 2>/dev/null || echo "   ❌ Diffusers not installed"
python3 -c "import transformers; print(f'   Transformers: {transformers.__version__}')" 2>/dev/null || echo "   ❌ Transformers not installed"
echo ""

# Check CUDA
echo "3. CUDA Availability:"
python3 -c "import torch; print(f'   CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "   ❌ Cannot check CUDA"
python3 -c "import torch; print(f'   CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>/dev/null || echo "   ❌ Cannot check CUDA device"
echo ""

# Check directory structure
echo "4. Directory Structure:"
for dir in data/indices outputs/clip_cache checkpoints logs; do
    if [ -d "$dir" ]; then
        echo "   ✅ $dir"
    else
        echo "   ❌ $dir (missing)"
        mkdir -p "$dir"
        echo "      Created: $dir"
    fi
done
echo ""

# Check if index builder works
echo "5. Testing Index Builder:"
python3 -m fmri2img.data.nsd_index_builder --help > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "   ✅ Index builder is accessible"
else
    echo "   ❌ Index builder failed (run: pip install -e .)"
fi
echo ""

# Check data.yaml
echo "6. Configuration:"
if [ -f "configs/data.yaml" ]; then
    echo "   ✅ configs/data.yaml found"
else
    echo "   ❌ configs/data.yaml missing"
fi
echo ""

# Check if package is installed
echo "7. Package Installation:"
python3 -c "import fmri2img; print(f'   ✅ fmri2img installed at: {fmri2img.__file__}')" 2>/dev/null || echo "   ❌ fmri2img not installed (run: pip install -e .)"
echo ""

echo "✅ Diagnostics complete!"