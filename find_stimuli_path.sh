#!/bin/bash

# Find the correct S3 path for NSD stimuli

echo "=========================================="
echo "Finding NSD Stimuli in S3 Bucket"
echo "=========================================="
echo ""

echo "[1] Listing top-level nsddata_stimuli structure..."
aws s3 ls s3://natural-scenes-dataset/nsddata_stimuli/ --no-sign-request

echo ""
echo "[2] Listing stimuli subdirectory..."
aws s3 ls s3://natural-scenes-dataset/nsddata_stimuli/stimuli/ --no-sign-request | head -20

echo ""
echo "[3] Looking for nsd_stimuli.hdf5..."
aws s3 ls s3://natural-scenes-dataset/nsddata_stimuli/stimuli/ --recursive --no-sign-request | grep "hdf5"

echo ""
echo "[4] Looking for PNG files..."
aws s3 ls s3://natural-scenes-dataset/nsddata_stimuli/stimuli/ --recursive --no-sign-request | grep "\.png" | head -10

echo ""
echo "[5] Checking experiments subdirectory..."
aws s3 ls s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/ --no-sign-request | head -10

echo ""
echo "=========================================="
echo "Alternative: Check what files exist"
echo "=========================================="
echo ""

# Try different possible paths
PATHS=(
    "nsddata_stimuli/stimuli/"
    "nsddata_stimuli/stimuli/nsd/"
    "nsddata_stimuli/"
    "nsddata/"
)

for path in "${PATHS[@]}"; do
    echo "Checking: s3://natural-scenes-dataset/${path}"
    aws s3 ls "s3://natural-scenes-dataset/${path}" --no-sign-request 2>&1 | head -5
    echo ""
done
