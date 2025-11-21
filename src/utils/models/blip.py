from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from typing import Union


class BLIPClient:
    """Minimal BLIP2 client for image captioning and VQA."""
    
    def __init__(
        self,
        model_type: str = "Salesforce/blip2-opt-2.7b",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Blip2Processor.from_pretrained(model_type)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_type,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
    
    def __call__(
        self,
        image: Union[str, Image.Image],
        prompt: str = None
    ) -> str:
        """
        Generate caption or answer question about image.
        
        Args:
            image: Image path or PIL Image
            prompt: Optional text prompt/question
            
        Returns:
            Generated caption or answer
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Process inputs
        inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt"
        ).to(self.device, torch.float16 if self.device == "cuda" else torch.float32)
        
        # Generate and decode
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        result = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        return result


# ===== USAGE =====
# client = BLIPClient()

# # 1. Simple image captioning (no prompt)
# caption = client("image.jpg")
# print(caption)

# # 2. Conditional captioning (with prompt)
# caption = client("image.jpg", prompt="A photo of")
# print(caption)

# # 3. Visual Question Answering
# answer = client("image.jpg", prompt="What color is the car?")
# print(answer)

# # 4. With PIL Image
# from PIL import Image
# img = Image.open("photo.jpg")
# caption = client(img, prompt="Describe this image")
# print(caption)
