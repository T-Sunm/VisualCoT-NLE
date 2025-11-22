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
from core.confirm.confirmation import Confirmer
from utils.models.blip import BLIPClient
from utils.models.llms import VLLMClient
from utils.models.clip import CLIPClient
from utils.preprocessing_text import extract_explanation
from tqdm import tqdm


PREDICT_EXAMPLES = [
    {
        "question": "Người đó đang làm gì?",
        "global_description": "An image of a person skiing on a snowy mountain. Key detail related to question: skiing.",
        "local_clues": [  # ĐỔI TỪ "visual_clues" THÀNH "local_clues"
            "- person: a person wearing winter clothes, slightly bent forward",
            "- ski: a pair of skis on the white snow",
            "- pole: a ski pole held in one hand"
        ],
        "verified_thoughts": [
            "- Người đó đang đi ván trượt trên tuyết, phù hợp với trang phục mùa đông."
        ],
        "answer": "trượt tuyết"
    },
    {
        "question": "Chú chó đang cố gắng bắt cái gì?",
        "global_description": "An image of a dog running on a grassy field. Key detail related to question: frisbee.",
        "local_clues": [  # ĐỔI
            "- dog: a golden retriever dog running forward",
            "- frisbee: a red flying disc in the air in front of the dog",
            "- grass: a green grass field"
        ],
        "verified_thoughts": [
            "- Có một chiếc đĩa bay trên không và chú chó đang chạy về phía nó."
        ],
        "answer": "đĩa bay"
    },
    {
        "question": "Cảnh này có thể đang diễn ra ở đâu?",
        "global_description": "An image of a table with food and drink. Key detail related to question: dining room.",
        "local_clues": [  # ĐỔI
            "- table: a wooden dining table",
            "- food: plates of pasta and salad on the table",
            "- wine: a glass of red wine next to the plates"
        ],
        "verified_thoughts": [
            "- Cảnh có một bàn ăn với nhiều đồ ăn và rượu vang, đây là đặc điểm của một phòng ăn."
        ],
        "answer": "phòng ăn"
    }
]

CONFIRM_EXAMPLES = [
    {
        "question": "Người đó đang làm gì?",
        "global_description": "An image of a person skiing on a snowy mountain. Key detail: skiing.",
        "local_clues": [  
            "- person: a person wearing winter clothes, slightly bent forward",
            "- ski: a pair of skis on the white snow",
            "- pole: a ski pole held in one hand"
        ],
        "answer": "trượt tuyết",
        "explanation": "Bức ảnh mô tả cảnh người đang trượt xuống dốc tuyết với ván trượt và quần áo mùa đông, nên họ đang trượt tuyết."
    },
    {
        "question": "Chú chó đang cố gắng bắt cái gì?",
        "global_description": "An image of a dog running on a grassy field. Key detail: frisbee.",
        "local_clues": [  
            "- dog: a golden retriever dog running forward",
            "- frisbee: a red flying disc in the air in front of the dog",
            "- grass: a green grass field"
        ],
        "answer": "đĩa bay",
        "explanation": "Dựa vào việc có một chiếc đĩa bay trên không và chú chó đang chạy về phía nó trong bối cảnh bãi cỏ, hành động hợp lý nhất là bắt đĩa bay."
    },
    {
        "question": "Cảnh này có thể đang diễn ra ở đâu?",
        "global_description": "An image of a table with food and drink. Key detail: dining room.",
        "local_clues": [  
            "- table: a wooden dining table",
            "- food: plates of pasta and salad on the table",
            "- wine: a glass of red wine next to the plates"
        ],
        "answer": "phòng ăn",
        "explanation": "Khung cảnh có bàn ăn được bày biện đầy đủ thức ăn và rượu vang là đặc trưng của một phòng ăn."
    }
]



class VisualCoTPipeline:
    """Pipeline: Detect -> Select -> Caption -> Reason -> Confirm -> Loop"""
    
    def __init__(self, attention_module, caption_module, reason_module, confirm_module, clip_module, 
                 max_iter=3, clip_threshold=0.2):
        self.attention = attention_module
        self.captioner = caption_module
        self.think = reason_module
        self.confirmer = confirm_module
        self.clip = clip_module
        self.max_iter = max_iter
        self.clip_threshold = clip_threshold
    
    def run(self, image_path: str, sg_path: str, question: str, key: str = None) -> Dict:
        objects = self.attention.detect_objects(sg_path)
        
        noticed_objects = []
        
        # Tách biệt: global description (string) và local clues (list)
        global_description = ""
        local_clues = []
        verified_thoughts = []
        
        # Lấy global caption TRƯỚC vòng lặp
        print("\n[INFO] Generating global image description...")
        global_description = self.captioner.caption_global(image_path, question)
        print(f"[GLOBAL] {global_description}")
        
        P_con = global_description
        previous_answer = None
        
        for i in range(self.max_iter):
            # 1-2. Select & Caption object
            idx = self.attention.select(objects, key)
            if idx is None or idx >= len(objects): break
            
            obj = objects.pop(idx)
            cap_i = self.captioner.caption(image_path, obj["name"])
            noticed_objects.append({"name": obj["name"], "caption": cap_i})
            
            # Format visual clue line (local objects)
            clue_line = f"- {obj['name']}: {cap_i}"
            local_clues.append(clue_line)
            
            # Update P_con
            P_con += f"\n{obj['name']}: {cap_i}"
            
            # 3. THINK: LLMPredict - Truyền global_description và local_clues riêng biệt
            a_i = self.think.predict(question, global_description, local_clues, verified_thoughts, examples=PREDICT_EXAMPLES)
            
            # 4. CONFIRM: LLMConfirm 
            r_i = self.confirmer.confirm(
                question=question, 
                global_description=global_description, 
                local_clues=local_clues, 
                answer=a_i, 
                examples=CONFIRM_EXAMPLES
            )
            
            # 5. VERIFY: Check rationale với CLIP
            clip_score = self.clip(image_path, r_i)
            
            if clip_score >= self.clip_threshold:
                # Add rationale vào danh sách verified thoughts
                thought_line = f"- {r_i}"
                verified_thoughts.append(thought_line)
                
                # Update P_con
                P_con += f"\n{r_i}"
                print(f"✓ Verified (CLIP: {clip_score:.3f})")
            else:
                print(f"✗ Rejected (CLIP: {clip_score:.3f})")
            
            # 6. CONVERGENCE: Check a_i == a_{i-1}
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
        confirmer = Confirmer(llm_client)
        clip_client = CLIPClient()
        
        pipeline = VisualCoTPipeline(attention, captioner, reasoner, confirmer, clip_client, 
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