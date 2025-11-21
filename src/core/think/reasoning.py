"""
Reasoning Module - Sử dụng LLM để suy luận (Predict & Confirm)
"""
from typing import List, Dict
from utils.models.llms import VLLMClient
from utils.prompts.llms_builder.llms_think import LLMsThinkPromptBuilder
from utils.prompts.llms_builder.llms_confirm import LLMsConfirmPromptBuilder


class Reasoner:
    """Module suy luận sử dụng LLM theo VCTP paper"""
    
    def __init__(self, llm_client: VLLMClient = None):
        self.llm = llm_client
        self.think_builder = LLMsThinkPromptBuilder()
        self.confirm_builder = LLMsConfirmPromptBuilder()
    
    def predict(self, question: str, context: str, examples: List[Dict] = []) -> str:
        """LLMPredict: Dự đoán answer"""
        prompt = self.think_builder.build(examples, question, context)
        system_prompt = self.think_builder.get_system_prompt()
        
        # Combine system prompt + user prompt
        full_prompt = prompt # VLLMClient hiện tại chưa xử lý system prompt tách biệt tốt với một số model
        if system_prompt:
             # Với VLLMClient hiện tại, ta có thể prepend system prompt nếu cần
             # hoặc dùng argument system_prompt nếu client đã update
             pass

        print("\n" + "-"*40)
        print("[LLM PREDICT] Input Prompt:")
        print(full_prompt)
        print("-" * 40)
        
        response = self.llm(full_prompt)
        
        print("[LLM PREDICT] Output:")
        print(response)
        print("-" * 40 + "\n")
        
        return response.strip()
    
    def confirm(self, question: str, context: str, answer: str, examples: List[Dict] = []) -> str:
        """
        LLMConfirm (Algorithm 1, line 13): Generate rationale cho answer
        """
        # Pass examples xuống builder
        prompt = self.confirm_builder.build_rationale_prompt(question, context, answer, examples)
        system_prompt = self.confirm_builder.get_system_prompt()
        
        # Thêm debug print như predict()
        print("\n" + "-"*40)
        print("[LLM CONFIRM] Input Prompt:")
        print(prompt)
        print("-" * 40)
        
        response = self.llm(prompt, system_prompt=system_prompt)
        
        print("[LLM CONFIRM] Output:")
        print(response)
        print("-" * 40 + "\n")
        
        # Clean response ... (giữ nguyên)
        if "Explanation:" in response:
            rationale = response.split("Explanation:")[-1].strip()
        else:
            rationale = response.strip()
        
        return rationale
    
    def answer(self, question: str, context: str) -> str:
        """Final answer generation"""
        return self.predict(question, context)