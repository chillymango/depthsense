#!/bin/bash
# salloc -N1 -t0:60:00 --cpus-per-task 8 --ntasks-per-node=1 --gres=gpu:V100:1 --mem-per-gpu=8G

cd "$(dirname "$0")"

MODEL_DIR="checkpoints"

# ================================ This is to download DepthAnythingV2 pretrained model ========================================
DEPTHANYTHING_MODEL_NAME="depth_anything_v2_vitl.pth"
DEPTHANYTHING_MODEL_PATH="${MODEL_DIR}/${DEPTHANYTHING_MODEL_NAME}"

# Hugging Face model URL (with redirect support)
URL="https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"

# Create destination directory if not exists
mkdir -p "${MODEL_DIR}"

# Check if model file exists
if [ -f "${DEPTHANYTHING_MODEL_PATH}" ]; then
    echo "[✓] Model already exists at ${DEPTHANYTHING_MODEL_PATH}"
else
    echo "[↓] Downloading Depth Anything V2 (ViT-L)..."
    curl -L "${URL}" -o "${DEPTHANYTHING_MODEL_PATH}"
    echo "[✓] Download complete: ${DEPTHANYTHING_MODEL_PATH}"
fi
# ================================================================================================================================


ENCODER="vitl"
# ============ [Modify Here] Configuration ===================
MODEL="depthsense" # 'depthsense' or 'depthanythingv2'
DEPTHSENSE_MODEL_NAME='checkpoint_epoch_7.pt' # update for depthsense

DATASET_NAME='diode'
DATA_DIR="/home/hice1/ylee904/scratch/depthsense/depthsense/data"
OUTPUT_DIR="./output/${DATASET_NAME}"
OUTPUT_JSON="benchmark_${DATASET_NAME}_${MODEL}_results.json"

MAX_DEPTH=80.0
BATCH_SIZE=64
# ============================================================


# Set MODEL_PATH depending on MODEL
if [ "$MODEL" = "depthsense" ]; then
    MODEL_PATH="${MODEL_DIR}/${DEPTHSENSE_MODEL_NAME}"
else
    MODEL_PATH="$DEPTHANYTHING_MODEL_PATH"
fi

echo "Using model path: $MODEL_PATH"

# Construct and print the command
CMD="python benchmark.py \
  --model $MODEL \
  --encoder $ENCODER \
  --model-path $MODEL_PATH \
  --data-dir $DATA_DIR \
  --output-dir $OUTPUT_DIR \
  --max-depth $MAX_DEPTH \
  --batch-size $BATCH_SIZE \
  --output-json $OUTPUT_JSON"

# Print and run
echo "[•] Running: $CMD"
eval "$CMD"