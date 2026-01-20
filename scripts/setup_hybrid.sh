#!/bin/bash
# =============================================================================
# Quick Setup for Hybrid Configuration (AWS S3 + MinIO)
# =============================================================================
# This script sets up the hybrid configuration automatically
# =============================================================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================================================${NC}"
echo -e "${BLUE}  Hybrid Setup: AWS S3 (NSD) + MinIO (Your Artifacts)${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""

# Get username
echo -e "${YELLOW}Step 1: Configure Username${NC}"
read -p "Enter your JupyterHub username [${USER}]: " USERNAME
USERNAME=${USERNAME:-$USER}

# Copy and configure .env
echo -e "${YELLOW}Step 2: Create .env file${NC}"
cp .env.hybrid .env
sed -i "s/USER=\${USER:-student01}/USER=\${USER:-${USERNAME}}/g" .env
echo -e "${GREEN}✓ Created .env with username: ${USERNAME}${NC}"

# Setup MinIO client
echo ""
echo -e "${YELLOW}Step 3: Setup MinIO Client${NC}"
if ! command -v mc &> /dev/null; then
    echo "Installing MinIO client to ~/bin ..."
    
    # Create bin directory if it doesn't exist
    mkdir -p ~/bin
    
    # Download MinIO client
    wget -q --show-progress https://dl.min.io/client/mc/release/linux-amd64/mc -O ~/bin/mc
    chmod +x ~/bin/mc
    
    # Add to PATH for this session
    export PATH="$HOME/bin:$PATH"
    
    echo -e "${GREEN}✓ Installed MinIO client to ~/bin/mc${NC}"
    
    # Add to shell profile if not already there
    if ! grep -q 'export PATH="$HOME/bin:$PATH"' ~/.bashrc 2>/dev/null; then
        echo 'export PATH="$HOME/bin:$PATH"' >> ~/.bashrc
        echo -e "${GREEN}✓ Added ~/bin to PATH in ~/.bashrc${NC}"
    fi
    
    echo -e "${BLUE}  Note: Restart shell or run: export PATH=\"\$HOME/bin:\$PATH\"${NC}"
else
    echo -e "${GREEN}✓ MinIO client already installed${NC}"
fi

# Configure MinIO alias
echo -e "${YELLOW}Step 4: Configure MinIO Access${NC}"
mc alias set uniminio https://ubbc1u.ro:9000 \
    cjChy0M8fUFIUWJ0SVd9 \
    TeIDlHx94IJS8ACQPNsmyFqo8IfnKbmzheKYkfxf \
    --api S3v4

if mc ls uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/ &>/dev/null; then
    echo -e "${GREEN}✓ MinIO connection successful${NC}"
else
    echo -e "${YELLOW}⚠ MinIO connection issue - check credentials${NC}"
fi

# Test AWS S3 access
echo ""
echo -e "${YELLOW}Step 5: Verify AWS S3 Access${NC}"
if python3 -c "import fsspec; fs = fsspec.filesystem('s3', anon=True); print('✓ AWS S3 accessible')" 2>/dev/null; then
    echo -e "${GREEN}✓ AWS S3 NSD bucket accessible${NC}"
else
    echo -e "${YELLOW}⚠ Install s3fs: pip install s3fs fsspec aiobotocore${NC}"
fi

# Install Python environment
echo ""
echo -e "${YELLOW}Step 6: Python Environment${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}✓ Python environment exists${NC}"
else
    read -p "Install Python environment now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        bash scripts/setup_env.sh
    else
        echo "Skipped. Run: bash scripts/setup_env.sh"
    fi
fi

# Summary
echo ""
echo -e "${BLUE}========================================================================${NC}"
echo -e "${GREEN}✅ Hybrid Setup Complete!${NC}"
echo -e "${BLUE}========================================================================${NC}"
echo ""
echo "Configuration:"
echo "  • User: ${USERNAME}"
echo "  • NSD Data: AWS S3 (direct streaming, no download)"
echo "  • Artifacts: MinIO bucket 6f628397-de73-4c99-b4b6-51e20859fea2"
echo "  • Config file: .env"
echo ""
echo "Next steps:"
echo ""
echo "1. Activate environment:"
echo "   ${BLUE}source activate_env.sh${NC}"
echo ""
echo "2. Build CLIP caches (streams from AWS S3):"
echo "   ${BLUE}python3 scripts/build_target_clip_cache_robust.py --use-s3${NC}"
echo "   ${BLUE}bash scripts/build_clip_for_training.sh${NC}"
echo "   ${BLUE}bash scripts/build_all_preprocessors.sh${NC}"
echo ""
echo "3. Upload caches to MinIO:"
echo "   ${BLUE}mc cp -r cache/ uniminio/6f628397-de73-4c99-b4b6-51e20859fea2/cache/${NC}"
echo ""
echo "4. Run experiments:"
echo "   ${BLUE}bash scripts/run_all_experiments.sh 0${NC}"
echo ""
echo "Benefits:"
echo "  ✓ No 65GB NSD download needed"
echo "  ✓ Streams directly from AWS S3"
echo "  ✓ Only ~6GB artifacts to MinIO"
echo "  ✓ Saves 3-4 hours setup time"
echo ""
echo -e "${GREEN}Ready to go! 🚀${NC}"
echo ""
