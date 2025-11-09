BASE_DIR="/home/research/workspace/VisualCoT-NLE"
SG_PATH="/home/research/workspace/data/raw/scene-graph"  # Sửa path này
VIVQA_PATH="/home/research/workspace/data/raw/vivqa-x/annotations"
OUTPUT_PATH="${BASE_DIR}/data/processed/object_similarity"

export PYTHONPATH="${BASE_DIR}/src" # Thêm thư mục 'src' vào PYTHONPATH để Python tìm thấy package 'vctp'
mkdir -p "${OUTPUT_PATH}"

python -m src.vctp.data.preprocess.object_similarity.cli \
    --dataset vivqax \
    --split train \
    --sg_path "${SG_PATH}" \
    --annotations_dir "${VIVQA_PATH}" \
    --output_path "${OUTPUT_PATH}/train_object_select_vivqax_answer.pkl" \
    --metric answer \
    --device cuda