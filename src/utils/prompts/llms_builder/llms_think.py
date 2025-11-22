"""
Builder để tạo prompt cho LLMs reasoning/thinking (LLMPredict)
"""


class LLMsThinkPromptBuilder:
    """Builder cho bước Prediction (LLMPredict)"""
    
    def __init__(self):
        self._set_prompts()
    
    def _set_prompts(self):
        self.system_prompt = (
            "You are a visual reasoning expert. "
            "Your task is to answer questions about images based on provided object descriptions.\n"
            "REQUIREMENTS:\n"
            "- Output ONLY the short answer in Vietnamese (Tiếng Việt).\n"
            "- NO Chinese (Không tiếng Trung), NO English (Không tiếng Anh).\n" # Thêm dòng này
            "- Be concise (1-3 words).\n"
            "- Do NOT explain.\n"
        )
        self.prompt_start = "===\n"
    
    def build(self, examples: list, question: str, context: str) -> str:
        """Xây dựng prompt cho bước Predict Answer"""
        prompt = f"{self.system_prompt}\n{self.prompt_start}"
        
        # Few-shot examples
        for ex in examples:
            prompt += f'Context: {ex["context"]}\n'
            prompt += f'Question: {ex["question"]}\n'
            prompt += f'Answer: {ex["answer"]}\n\n===\n'
        
        # Current case
        prompt += f'Context: {context}\n'
        prompt += f'Question: {question}\n'
        prompt += 'Answer:'
        
        return prompt
    
    def get_system_prompt(self) -> str:
        return self.system_prompt