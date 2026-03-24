#!/bin/bash

# Build and push wifi-connect image to Google Artifact Registry
# Usage: ./scripts/build-and-push.sh

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Image configuration
REGISTRY="us-central1-docker.pkg.dev"
PROJECT="omgrab-dev"
REPOSITORY="omgrab-dev-client-runtime"
IMAGE_NAME="wifi-connect"
FULL_IMAGE="${REGISTRY}/${PROJECT}/${REPOSITORY}/${IMAGE_NAME}:latest"

echo "=================================="
echo "Building wifi-connect image"
echo "=================================="
echo ""
echo "Image: ${FULL_IMAGE}"
echo ""

cd "$PROJECT_DIR"

# Check for authentication
if ! docker-credential-gcr list 2>/dev/null | grep -q "$REGISTRY"; then
    echo "Warning: May need to authenticate with GCR."
    echo "Run: docker-credential-gcr configure-docker --registries=${REGISTRY}"
    echo ""
fi

# Build the image for ARM64 (Raspberry Pi)
echo "Building image for linux/arm64..."
docker build \
    --platform linux/arm64 \
    -f Dockerfile.template \
    -t "${FULL_IMAGE}" \
    .

echo ""
echo "Build complete!"
echo ""

# Push to registry
echo "Pushing to ${REGISTRY}..."
docker push "${FULL_IMAGE}"

echo ""
echo "=================================="
echo "Push Complete!"
echo "=================================="
echo ""
echo "Image available at:"
echo "  ${FULL_IMAGE}"
echo ""
