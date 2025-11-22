"""
Reasoning Module - Sử dụng LLM để suy luận (Predict)
"""
from typing import List, Dict
from utils.models.llms import VLLMClient

# System Prompt chỉ chứa Role và Instruction
SYSTEM_INSTRUCTION = """You are a Visual Reasoning AI expert for Vietnamese.
You will receive:
- "Global Image Description": a general description of the whole scene in English.
- "Visual Clues": detailed object descriptions in English.
- "Verified Thoughts": previously verified answers and explanations in VIETNAMESE.
Your goal is to connect these pieces of information to give a short, direct answer to the main question.

Answer style:
- Respond ONLY with 1–3 Vietnamese words.
- Do NOT explain.
"""

class Reasoner:
    """Module suy luận sử dụng LLM theo VCTP paper"""
    
    def __init__(self, llm_client: VLLMClient = None):
        self.llm = llm_client
    
    def _build_few_shot_prompt(self, examples: List[Dict]) -> str:
        """Helper để tạo chuỗi Few-shot từ list dictionary"""
        if not examples:
            return ""
            
        prompt_parts = ["### FEW-SHOT EXAMPLES ###"]
        
        for i, ex in enumerate(examples, 1):
            part = f"[EXAMPLE {i}]\n"
            part += f"[Question]: {ex.get('question', '')}\n"
            
            # TÁCH RIÊNG: Global Description
            global_desc = ex.get('global_description', '')
            part += f"[Global Image Description]:\n{global_desc}\n"
            
            # TÁCH RIÊNG: Visual Clues (chỉ local objects)
            local_clues = ex.get('local_clues', [])
            if isinstance(local_clues, list):
                visual_clues_str = "\n".join(local_clues)
            else:
                visual_clues_str = str(local_clues)
            part += f"[Visual Clues]:\n{visual_clues_str}\n"
            
            # Xử lý Verified Thoughts
            verified = ex.get('verified_thoughts', '')
            if isinstance(verified, list):
                verified_str = "\n".join(verified)
            else:
                verified_str = str(verified).strip()
            
            part += f"[Verified Thoughts]:\n{verified_str}\n"
            part += f"[Final Answer]: {ex.get('answer', '')}\n"
            part += "---"
            
            prompt_parts.append(part)
            
        return "\n\n".join(prompt_parts)

    def predict(self, question: str, global_description: str, local_clues: List[str], verified_thoughts: List[str], examples: List[Dict] = []) -> str:
        """
        LLMPredict: Dự đoán answer
        Args:
            question: Câu hỏi
            global_description: Mô tả tổng quan ảnh (string)
            local_clues: Danh sách các object clues (list of strings)
            verified_thoughts: Các suy luận đã được verify
            examples: Few-shot examples
        """
        # 1. Format Global Description
        if not global_description:
            global_desc_str = "(No global description yet)"
        else:
            global_desc_str = global_description
        
        # 2. Format Visual Clues (chỉ local objects)
        if not local_clues:
            visual_clues_str = "(No visual clues yet)"
        else:
            visual_clues_str = "\n".join(local_clues)
            
        # 3. Format Verified Thoughts
        if not verified_thoughts:
            current_thoughts_str = "(No verified thoughts yet)"
        else:
            current_thoughts_str = "\n".join(verified_thoughts)

        # 4. Build Few-shot block
        few_shot_block = self._build_few_shot_prompt(examples)

        # 5. Construct Full Prompt - TÁCH RIÊNG 2 TRƯỜNG
        prompt = (
            f"{SYSTEM_INSTRUCTION}\n\n"
            f"{few_shot_block}\n\n"
            f"### CURRENT TASK ###\n\n"
            f"[Question]: {question}\n\n"
            f"[Global Image Description]:\n{global_desc_str}\n\n"
            f"[Visual Clues]:\n{visual_clues_str}\n\n"
            f"[Verified Thoughts]:\n{current_thoughts_str}\n\n"
            f"[Final Answer]:"
        )
        
        response = self.llm(prompt)

        return response.strip()

    def answer(self, question: str, context: str) -> str:
        return ""