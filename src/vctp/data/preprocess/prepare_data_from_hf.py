# Tên file: prepare_data_from_hf.py
from datasets import load_dataset
import os
import json
from pathlib import Path

# --- Cấu hình ---
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")

# Dữ liệu thô tải về từ Hugging Face sẽ được lưu ở đây
ANNOTATION_DIR = RAW_DATA_DIR / "aokvqa_annotations"
IMAGE_DIR = RAW_DATA_DIR / "aokvqa_images"

# Tạo các thư mục nếu chưa tồn tại
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)  # Tạo luôn thư mục processed

# --- Tải dataset từ Hugging Face ---
print("Đang tải dataset A-OKVQA từ Hugging Face Hub...")
# 'HuggingFaceM4/A-OKVQA' là tên dataset trên Hub
# Tải 10 mẫu đầu tiên để test, bạn có thể bỏ `split='train[:10]'` để tải toàn bộ
dataset_slice = load_dataset("HuggingFaceM4/a-okvqa", split="train[:10]")
print("Tải dataset thành công.")

# --- Bước 1: Lưu annotations ra file JSON trong data/raw ---
output_json_path = ANNOTATION_DIR / "aokvqa_v1p0_train_from_hf.json"
annotations_list = []

print(f"Đang xử lý và lưu annotations vào {output_json_path}...")
for idx, item in enumerate(dataset_slice):
    # Tạo image_id tùy chỉnh dựa trên thứ tự của mẫu
    custom_image_id = idx

    # Script gốc cần các key này, chúng ta sẽ trích xuất chúng
    annotations_list.append(
        {
            "question_id": item["question_id"],
            "image_id": custom_image_id,  # Sử dụng ID tự tạo
            "question": item["question"],
            "choices": item["choices"],
            "correct_choice_idx": item["correct_choice_idx"],
            "direct_answers": item["direct_answers"],
            "rationales": item["rationales"],
        }
    )

    # --- Bước 2: Lưu ảnh ra thư mục ---
    # Script `make_clip_features.py` yêu cầu tên file ảnh là {image_id} được đệm 12 số 0
    image_filename = f"{str(custom_image_id).zfill(12)}.jpg"  # Dùng ID tự tạo để đặt tên file
    image_path = IMAGE_DIR / image_filename

    # Lưu ảnh nếu chưa tồn tại
    if not image_path.exists():
        item["image"].save(image_path)

with open(output_json_path, "w") as f:
    json.dump(annotations_list, f, indent=4)

print("Hoàn thành việc lưu annotations và hình ảnh vào 'data/raw'.")
print(f" - File annotations: {output_json_path}")
print(f" - Thư mục hình ảnh: {IMAGE_DIR}")
print("\nBước tiếp theo: Chạy các script tiền xử lý để tạo output trong 'data/processed'.")
