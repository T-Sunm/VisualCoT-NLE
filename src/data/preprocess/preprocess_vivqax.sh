BASE_DIR="/home/research/workspace/VisualCoT-NLE"
VIVQA_PATH="/home/research/workspace/data/raw/vivqa-x/annotations"
VIVQA_IMAGES="/home/research/workspace/data/raw/coco/images"
OUTPUT_DIR="${BASE_DIR}/data/processed"

mkdir -p "${OUTPUT_DIR}/"

for SPLIT in "train" "val"
do
    echo "Processing ${SPLIT} split..."
    
    # Make line2sample mapping
    python -m src.vctp.data.preprocess.make_line2sample \
        --input "${VIVQA_PATH}/${SPLIT}.json" \
        --output "${OUTPUT_DIR}/vivqax_qa_line2sample_idx_${SPLIT}.json"
    
    # Make CLIP features
    python -m src.vctp.data.preprocess.make_clip_features \
        --questions "${VIVQA_PATH}/${SPLIT}.json" \
        --images "${VIVQA_IMAGES}/" \
        --ifeatures "${OUTPUT_DIR}/coco_clip_vitb16_${SPLIT}_vivqax_convertedidx_image.npy" \
        --qfeatures "${OUTPUT_DIR}/coco_clip_vitb16_${SPLIT}_vivqax_question.npy"
done