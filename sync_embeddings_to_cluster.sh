#!/bin/bash
# Sync embedding cache files to GPU cluster

CLUSTER_USER="md5_sd8f61177fd2312b9b32bd118ad1"
CLUSTER_HOST="GPU20C-N5"
CLUSTER_PATH="~/Bachelor_V2/cache/clip_embeddings/"

echo "Syncing embedding cache files to cluster..."
echo "Destination: ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}"
echo ""

# Create remote directory if it doesn't exist
ssh ${CLUSTER_USER}@${CLUSTER_HOST} "mkdir -p ~/Bachelor_V2/cache/clip_embeddings"

# Sync embedding files
rsync -avz --progress \
    cache/clip_embeddings/*.parquet \
    ${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}

echo ""
echo "✓ Sync complete!"
echo ""
echo "Verify on cluster with:"
echo "  ssh ${CLUSTER_USER}@${CLUSTER_HOST} 'ls -lh ~/Bachelor_V2/cache/clip_embeddings/'"
