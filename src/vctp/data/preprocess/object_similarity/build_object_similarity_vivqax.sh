BASE_DIR="/home/research/workspace/VisualCoT-NLE"
SG_PATH="${BASE_DIR}/data/processed/input_text/scene_graph_text"
VIVQA_PATH="/home/research/workspace/data/raw/vivqa-x/annotations"
OUTPUT_PATH="${BASE_DIR}/data/processed/object_similarity"

mkdir -p "${OUTPUT_PATH}"

python -m src.vctp.data.preprocess.object_similarity.cli \
    --dataset vivqax \
    --split train \
    --sg_path "${SG_PATH}" \
    --annotations_dir "${VIVQA_PATH}" \
    --output_path "${OUTPUT_PATH}/train_object_select_vivqax_answer.pkl" \
    --metric answer \
    --device cuda