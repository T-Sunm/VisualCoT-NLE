"""
Builder để tạo prompt cho LLMs confirmation/verification (LLMConfirm)
"""


class LLMsConfirmPromptBuilder:
    """Builder cho bước Confirm (Generate Rationale & Verify)"""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._set_prompts()
        
    def _set_prompts(self):
        # System prompt cho việc generate rationale
        self.system_prompt = (
            "You are a visual reasoning expert. "
            "Your task is to explain WHY an answer is correct based on object descriptions.\n"
            "REQUIREMENTS:\n"
            "- Output a concise explanation (rationale) in Vietnamese (1-2 sentences).\n"
            "- Link visual evidence from context to the answer.\n"
            "- Do NOT repeat the question or answer.\n"
        )

    def build_rationale_prompt(self, question: str, context: str, answer: str, examples: list = []) -> str:
        prompt = ""
        
        # Add Few-shot examples
        for ex in examples:
            prompt += (
                f"Context:\n{ex['context']}\n"
                f"Question: {ex['question']}\n"
                f"Answer: {ex['answer']}\n"
                f"Explanation: {ex['explanation']}\n\n===\n"
            )
            
        # Current case
        prompt += (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n"
            f"Answer: {answer}\n\n"
            f"Explanation:"
        )
        return prompt
    
    def build_verify_prompt(self, thought: str) -> str:
        """Prompt cho BLIP/CLIP để verify thought"""
        # Prompt này dành cho Vision Model (BLIP), không phải LLM
        return (
            f"Question: Does this sentence match the facts in the picture? "
            f"Please answer yes or no. Sentence: In this picture, {thought} Answer:"
        )
        
    def get_system_prompt(self) -> str:
        return self.system_prompt