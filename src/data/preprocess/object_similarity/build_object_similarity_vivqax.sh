BASE_DIR="/home/research/workspace/VisualCoT-NLE"
SG_PATH="/home/research/workspace/data/raw/scene-graph"
VIVQA_PATH="/home/research/workspace/data/raw/vivqa-x/annotations"
OUTPUT_PATH="${BASE_DIR}/data/processed/object_similarity"
SPLIT="test"
export PYTHONPATH="${BASE_DIR}"
mkdir -p "${OUTPUT_PATH}"

python -m src.data.preprocess.object_similarity.cli \
    --dataset vivqax \
    --split "${SPLIT}" \
    --sg_path "${SG_PATH}" \
    --annotations_dir "${VIVQA_PATH}" \
    --output_path "${OUTPUT_PATH}/${SPLIT}_object_select_vivqax_answer.pkl" \
    --metric answer \
    --device cuda