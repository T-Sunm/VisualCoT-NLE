#!/usr/bin/env bash
set -euo pipefail

# Download COCO 2014 train and val datasets from Google Drive
# Output: data/raw/coco/images/

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${PROJECT_ROOT}/data/raw/coco/images"

TRAIN_FILE_ID="1nzGeMFX_X5vuILI8-LTHWnVYM5U3TgGj"
VAL_FILE_ID="1ddgAovc1b7oLN26ax9nFusBnRq_TQNKp"

TRAIN_ZIP="${DATA_DIR}/train2014.zip"
VAL_ZIP="${DATA_DIR}/val2014.zip"

# Check and install gdown if needed
if ! command -v gdown &> /dev/null; then
    echo "Installing gdown..."
    pip install gdown
fi

# Create directory
mkdir -p "$DATA_DIR"
echo "Output directory: $DATA_DIR"
echo ""

# Download and extract train dataset
echo "[1/2] Downloading COCO Train 2014..."
if [ -f "$TRAIN_ZIP" ]; then
    read -p "File exists. Re-download? (y/n) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] && rm "$TRAIN_ZIP" || SKIP_TRAIN=1
fi

if [ -z "${SKIP_TRAIN:-}" ]; then
    gdown "https://drive.google.com/uc?id=${TRAIN_FILE_ID}" -O "$TRAIN_ZIP"
    echo "Extracting train2014..."
    unzip -q "$TRAIN_ZIP" -d "$DATA_DIR"
    rm "$TRAIN_ZIP"
fi

# Download and extract val dataset
echo "[2/2] Downloading COCO Val 2014..."
if [ -f "$VAL_ZIP" ]; then
    read -p "File exists. Re-download? (y/n) " -n 1 -r
    echo
    [[ $REPLY =~ ^[Yy]$ ]] && rm "$VAL_ZIP" || SKIP_VAL=1
fi

if [ -z "${SKIP_VAL:-}" ]; then
    gdown "https://drive.google.com/uc?id=${VAL_FILE_ID}" -O "$VAL_ZIP"
    echo "Extracting val2014..."
    unzip -q "$VAL_ZIP" -d "$DATA_DIR"
    rm "$VAL_ZIP"
fi

echo ""
echo "Download complete!"
echo "Location: $DATA_DIR"
