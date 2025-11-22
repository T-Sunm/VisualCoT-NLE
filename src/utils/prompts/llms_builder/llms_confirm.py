"""
Confirm Module - Generate Rationale
"""
from typing import List, Dict
from utils.models.llms import VLLMClient

CONFIRM_SYSTEM_INSTRUCTION = """You are a Visual Reasoning AI expert for Vietnamese. You will receive a question, a predicted answer, and visual clues (global description + local details). Your goal is to generate ONE logical Vietnamese sentence that explains WHY the predicted answer is correct, based ONLY on the visual clues provided.

RULES:
1. Generate one single, complete Vietnamese sentence.
2. The sentence MUST justify the [Predicted Answer].
3. The sentence MUST be grounded in the [Visual Clues]."""

class Confirmer:
    """Module xác nhận (Confirm) sử dụng LLM"""
    
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
            
            # Xử lý Visual Clues: global + local (giống reasoning.py)
            visual_clues_lines = []
            
            # Global description (nếu có)
            if 'global_description' in ex:
                visual_clues_lines.append(ex['global_description'])
            
            # Local clues
            if 'local_clues' in ex:
                if isinstance(ex['local_clues'], list):
                    visual_clues_lines.extend(ex['local_clues'])
                else:
                    visual_clues_lines.append(str(ex['local_clues']))
            
            visual_clues_str = "\n".join(visual_clues_lines)
            part += f"[Visual Clues]:\n{visual_clues_str}\n"
            
            part += f"[Predicted Answer]: {ex.get('answer', '')}\n"
            
            # Lấy explanation làm rationale
            rationale = ex.get('explanation', ex.get('rationale', ''))
            part += f"[Rationale]: {rationale}\n"
            part += "---"
            prompt_parts.append(part)
            
        return "\n\n".join(prompt_parts)

    def confirm(self, question: str, global_description: str, local_clues: List[str], answer: str, examples: List[Dict] = []) -> str:
        """
        LLMConfirm: Generate rationale cho answer
        Args:
            question: Câu hỏi
            global_description: Mô tả tổng quan ảnh
            local_clues: Danh sách các object clues
            answer: Câu trả lời đã dự đoán
            examples: Few-shot examples
        """
        # 1. Format Visual Clues (giống reasoning.py)
        if not global_description and not local_clues:
            current_clues_str = "(No visual clues yet)"
        else:
            clues_parts = []
            if global_description:
                clues_parts.append(global_description)
            if local_clues:
                clues_parts.extend(local_clues)
            current_clues_str = "\n".join(clues_parts)
            
        # 2. Build Few-shot block
        few_shot_block = self._build_few_shot_prompt(examples)
        
        # 3. Construct Full Prompt
        prompt = (
            f"{CONFIRM_SYSTEM_INSTRUCTION}\n\n"
            f"{few_shot_block}\n\n"
            f"### CURRENT TASK ###\n\n"
            f"[Question]: {question}\n\n"
            f"[Visual Clues]:\n{current_clues_str}\n\n"
            f"[Predicted Answer]: {answer}\n\n"
            f"[Rationale]:"
        )
        
        # Debug logging
        print("\n" + "-"*40)
        print("[LLM CONFIRM] Input Prompt:")
        print(prompt)
        print("-" * 40)
        
        response = self.llm(prompt)
        
        print("[LLM CONFIRM] Output:")
        print(response)
        print("-" * 40 + "\n")
        
        # Clean response
        rationale = response.strip()
        if rationale.startswith("[Rationale]:"):
             rationale = rationale.replace("[Rationale]:", "").strip()
        
        return rationale