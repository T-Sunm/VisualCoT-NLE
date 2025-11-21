from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage 


class VLLMClient:
    """Minimal vLLM client wrapper."""
    
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        self.llm = ChatOpenAI(
            model=model,
            openai_api_key="EMPTY",
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )
    
    def __call__(self, prompt: str, system_prompt: str = None) -> str:
        """
        Send prompt and get response.
        
        Args:
            prompt: User prompt string
            system_prompt: Optional system prompt string
        """
        messages = []
        
        # Thêm system prompt nếu có
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))
            
        messages.append(HumanMessage(content=prompt))
        
        response = self.llm.invoke(messages)
        return response.content


# ===== USAGE =====
# client = VLLMClient("NousResearch/Meta-Llama-3-8B-Instruct")

# # Sử dụng như function
# result = client("What is the capital of France?")
# print(result)

# result = client("Explain quantum computing in one sentence")
# print(result)
