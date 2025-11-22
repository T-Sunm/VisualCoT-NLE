"""
Pipeline chính cho Visual CoT
"""
from pathlib import Path
from typing import Dict, List
import tqdm
import yaml
import json
import argparse

from core.think.attention_obj import load_object_selector_from_config
from core.think.caption_obj import ObjectCaptioner
from core.think.reasoning import Reasoner
from utils.models.blip import BLIPClient
from utils.models.llms import VLLMClient
from utils.models.clip import CLIPClient
from utils.preprocessing_text import extract_explanation
from tqdm import tqdm


PREDICT_EXAMPLES = [
    {
        "context": "ski: đôi ván trượt tuyết trên tuyết, person: một người mặc quần áo mùa đông",
        "question": "Người đó đang làm gì?",
        "answer": "trượt tuyết",
    },
    {
        "context": "dog: một chú chó golden retriever đang chạy, frisbee: một chiếc đĩa bay màu đỏ trên không",
        "question": "Chú chó đang cố gắng bắt gì?",
        "answer": "đĩa bay",
    },
    {
        "context": "table: một chiếc bàn ăn bằng gỗ, food: các đĩa mì ống và salad",
        "question": "Cảnh này có thể đang diễn ra ở đâu?",
        "answer": "phòng ăn",
    }
]

CONFIRM_EXAMPLES = [
    {
        "context": "ski: đôi ván trượt tuyết trên tuyết, person: một người mặc quần áo mùa đông",
        "question": "Người đó đang làm gì?",
        "answer": "trượt tuyết",
        "explanation": "Người đó đang mặc quần áo mùa đông và đứng gần ván trượt tuyết trên tuyết, vì vậy họ có thể đang trượt tuyết."
    },
    {
        "context": "dog: một chú chó golden retriever đang chạy, frisbee: một chiếc đĩa bay màu đỏ trên không",
        "question": "Chú chó đang cố gắng bắt gì?",
        "answer": "đĩa bay",
        "explanation": "Có một chiếc đĩa bay trên không và chú chó đang chạy về phía nó, vì vậy chú chó đang cố gắng bắt chiếc đĩa bay."
    },
    {
        "context": "table: một chiếc bàn ăn bằng gỗ, food: các đĩa mì ống và salad",
        "question": "Cảnh này có thể đang diễn ra ở đâu?",
        "answer": "phòng ăn",
        "explanation": "Có một chiếc bàn ăn với thức ăn được bày ra trên đó, điều này cho thấy cảnh đang ở trong phòng ăn."
    }
]


class VisualCoTPipeline:
    """Pipeline: Detect -> Select -> Caption -> Reason -> Confirm -> Loop"""
    
    def __init__(self, attention_module, caption_module, reason_module, clip_module, 
                 max_iter=3, clip_threshold=0.2):
        self.attention = attention_module
        self.captioner = caption_module
        self.think = reason_module
        self.clip = clip_module
        self.max_iter = max_iter
        self.clip_threshold = clip_threshold
    
    def run(self, image_path: str, sg_path: str, question: str, key: str = None) -> Dict:
        objects = self.attention.detect_objects(sg_path)
        
        noticed_objects = []
        P_con = ""  # Context tích lũy (P_con,i từ algorithm)
        previous_answer = None
        
        for i in range(self.max_iter):
            # 1-2. Select & Caption object
            idx = self.attention.select(objects, key)
            if idx is None or idx >= len(objects): break
            
            obj = objects.pop(idx)
            cap_i = self.captioner.caption(image_path, obj["name"])
            noticed_objects.append({"name": obj["name"], "caption": cap_i})
            
            # Update context: P_con,i = P_con,i-1 + cap_i
            P_con += f"\n{obj['name']}: {cap_i}"
            
            # 3. THINK: LLMPredict - dự đoán answer (line 11)
            a_i = self.think.predict(question, P_con, examples=PREDICT_EXAMPLES)
            
            # 4. CONFIRM: LLMConfirm - generate rationale (line 13)
            r_i = self.think.confirm(question, P_con, a_i, examples=CONFIRM_EXAMPLES)
            
            # 5. VERIFY: Check rationale với CLIP (line 14)
            clip_score = self.clip(image_path, r_i)
            
            if clip_score >= self.clip_threshold:
                # Add rationale vào context (line 15)
                P_con += f"\n{r_i}"
                print(f"✓ Verified (CLIP: {clip_score:.3f})")
            else:
                print(f"✗ Rejected (CLIP: {clip_score:.3f})")
            
            # 6. CONVERGENCE: Check a_i == a_{i-1} (line 17)
            print("current answer: ", a_i)
            print("explanation: ", r_i)
            if a_i == previous_answer:
                print("Converged!")
                break
            previous_answer = a_i
        
        return {"answer": a_i, "context": P_con, "explanation": r_i}


def load_samples(config_path: str, limit: int = 1):
    """Load samples từ config"""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    ds_cfg_path = cfg["datasets"][cfg["experiment"]["dataset"]]
    with open(ds_cfg_path) as f:
        ds_cfg = yaml.safe_load(f)["dataset"]
    
    with open(ds_cfg["annotations_file"]) as f:
        anns = json.load(f)[:limit]
    
    images_root = Path(ds_cfg["images_root"])
    sg_dir = Path(ds_cfg["scene_graph_attr_dir"])
    
    # Đọc image_split từ config, mặc định là train2014 nếu không có
    img_split = ds_cfg.get("image_split", "train2014")
    img_prefix = f"COCO_{img_split}_"

    samples = []
    for ann in anns:
        img_id = ann["image_id"]
        samples.append({
            "image_path": str(images_root / f"{img_split}/{img_prefix}{str(img_id).zfill(12)}.jpg"),
            "sg_path": str(sg_dir / f"{str(img_id).zfill(12)}.json"),
            "question": ann["question"],
            "answer": ann.get("answer", ""),
            "key": f"{img_id}<->{ann.get('question_id', 0)}",
            "image_id": img_id
        })
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/vivqax_baseline.yaml")
    parser.add_argument("--limit", type=int, default=300)  # Default 300
    parser.add_argument("--threshold", type=float, default=0.2, help="CLIP threshold")
    parser.add_argument("--output", default="results/visual_cot_results.json", help="Output file path")
    args = parser.parse_args()
    
    print("Initializing modules...")
    try:
        attention = load_object_selector_from_config(args.config)
        captioner = ObjectCaptioner(BLIPClient())
        llm_client = VLLMClient(model="Qwen/Qwen2.5-3B-Instruct")
        reasoner = Reasoner(llm_client)
        clip_client = CLIPClient()
        
        pipeline = VisualCoTPipeline(attention, captioner, reasoner, clip_client, 
                                    clip_threshold=args.threshold)
    except Exception as e:
        print(f"Error initializing modules: {e}")
        return

    samples = load_samples(args.config, args.limit)
    print(f"Loaded {len(samples)} samples. Starting inference...")
    
    results = []
    
    # Dùng tqdm để hiển thị tiến độ
    for sample in tqdm(samples, desc="Processing"):
        try:
            result = pipeline.run(
                sample["image_path"],
                sample["sg_path"],
                sample["question"],
                sample["key"]
            )
            
            # Lưu kết quả kèm thông tin sample
            output_item = {
                "question_id": sample["key"].split("<->")[-1], # Lấy ID câu hỏi
                "image_id": sample["image_id"],
                "question": sample["question"],
                "ground_truth": sample["answer"],
                "prediction": result["answer"],
                "context": result["context"],
                "explanation": result["explanation"]
            }
            results.append(output_item)
            
        except Exception as e:
            print(f"\nError processing sample {sample['key']}: {e}")
            # Vẫn lưu mẫu lỗi để biết
            results.append({
                "question_id": sample["key"].split("<->")[-1],
                "error": str(e)
            })
            continue

    # Lưu kết quả ra file JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    print(f"\nSuccessfully processed {len(results)} samples.")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()