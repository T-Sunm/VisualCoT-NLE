#!/bin/bash

# Configuration
BASE_DIR="."
SG_PATH="${BASE_DIR}/data/processed/input_text/scene_graph_text"
COCO_PATH="${BASE_DIR}/data/raw/coco_annotations"
OUTPUT_PATH="${BASE_DIR}/data/processed/object_similarity"

# Run preprocessing
python -m src.vctp.data.preprocess.object_similarity.cli \
    --dataset aokvqa \
    --split train \
    --sg_path "${SG_PATH}" \
    --annotations_dir "${COCO_PATH}" \
    --output_path "${OUTPUT_PATH}/train_object_select_aokvqa_answer.pkl" \
    --metric answer \
    --device cuda