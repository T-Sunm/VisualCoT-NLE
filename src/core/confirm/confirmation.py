"""
Confirm Module - Generate Rationale
"""
from typing import List, Dict
from utils.models.llms import VLLMClient

# 1. System Instruction mới
CONFIRM_SYSTEM_INSTRUCTION = """You are a Visual Reasoning AI expert for Vietnamese. You will receive a question, a predicted answer, and a list of visual clues. Your goal is to generate ONE logical Vietnamese sentence that explains WHY the predicted answer is correct, based ONLY on the visual clues provided.

RULES:
1. Generate one single, complete Vietnamese sentence.
2. The sentence MUST justify the [Predicted Answer].
3. The sentence MUST be grounded in the [Visual Clues]."""

class Confirmer:
    """Module xác nhận (Confirm) sử dụng LLM"""
    
    def __init__(self, llm_client: VLLMClient = None):
        self.llm = llm_client
        # Removed old builder: self.confirm_builder = LLMsConfirmPromptBuilder()
    
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
            
        # 3. Build Few-shot block
        few_shot_block = self._build_few_shot_prompt(examples)
        
        # 4. Construct Full Prompt - TÁCH RIÊNG 2 TRƯỜNG
        prompt = (
            f"{CONFIRM_SYSTEM_INSTRUCTION}\n\n"
            f"{few_shot_block}\n\n"
            f"### CURRENT TASK ###\n\n"
            f"[Question]: {question}\n\n"
            f"[Global Image Description]:\n{global_desc_str}\n\n"
            f"[Visual Clues]:\n{visual_clues_str}\n\n"
            f"[Predicted Answer]: {answer}\n\n"
            f"[Rationale]:"
        )
        
        response = self.llm(prompt)

        rationale = response.strip()
        if rationale.startswith("[Rationale]:"):
            rationale = rationale.replace("[Rationale]:", "").strip()
        
        return rationale