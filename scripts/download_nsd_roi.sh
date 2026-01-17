#!/bin/bash
# Download NSD ROI masks for subject 01

set -e

ROI_DIR="data/nsd/ppdata/subj01/func1pt8mm"
mkdir -p "$ROI_DIR"

echo "Downloading NSD ROI masks..."

# Download nsdgeneral ROI (commonly used, ~15k voxels in visual cortex)
if [ ! -f "$ROI_DIR/roi_nsdgeneral.nii.gz" ]; then
    echo "Downloading nsdgeneral ROI..."
    wget -q --show-progress \
        "https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj01/func1pt8mm/roi/nsdgeneral.nii.gz" \
        -O "$ROI_DIR/roi_nsdgeneral.nii.gz"
    echo "✓ Downloaded nsdgeneral ROI"
else
    echo "✓ nsdgeneral ROI already exists"
fi

echo ""
echo "ROI masks ready at: $ROI_DIR"
echo "Use with: NSDDataset(..., roi_mask_path='$ROI_DIR/roi_nsdgeneral.nii.gz')"
