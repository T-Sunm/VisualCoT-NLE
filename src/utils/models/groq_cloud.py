from groq import Groq, RateLimitError
import base64
import os
import time
from typing import Optional


class GroqClient:
    def __init__(
        self,
        model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def __call__(
        self, 
        prompt: str, 
        image_path: Optional[str] = None
    ) -> str:
        """Send prompt and get response with auto-sleep and retry."""
        
        # Build content
        if image_path:
            base64_image = self.encode_image(image_path)
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ]
        else:
            content = prompt
        
        # Retry loop
        while True:
            try:
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": content}],
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                
                # Sleep chủ động 3s để đảm bảo không bao giờ vượt quá 20 req/phút (an toàn)
                time.sleep(3)
                
                return chat_completion.choices[0].message.content
                
            except RateLimitError as e:
                retry_after = int(e.response.headers.get("retry-after", 60))
                print(f"⚠️ Rate limit hit! Sleeping for {retry_after}s...")
                time.sleep(retry_after)
                continue  # Retry again
