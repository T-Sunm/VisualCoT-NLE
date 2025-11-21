from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from typing import Union


class CLIPClient:
    """Minimal CLIP client for image-text similarity."""
    
    def __init__(
        self,
        model_type: str = "openai/clip-vit-base-patch32",
        device: str = None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_type)
        self.model = CLIPModel.from_pretrained(model_type).to(self.device)
    
    def __call__(
        self,
        image: Union[str, Image.Image],
        text: str
    ) -> float:
        """
        Compute image-text similarity.
        
        Args:
            image: Image path or PIL Image
            text: Text description
            
        Returns:
            Similarity score (0-100)
        """
        # Load image if path provided
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        
        # Process inputs
        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True,  # Thêm truncation
            max_length=77     # CLIP max tokens
        ).to(self.device)
        
        # Compute similarity
        with torch.no_grad():
            outputs = self.model(**inputs)
            similarity = outputs.logits_per_image[0][0].item()
        
        return similarity


# ===== USAGE =====
# client = CLIPClient()

# # Simple usage
# score = client("cat.jpg", "a cute cat")
# print(f"Similarity: {score:.2f}")

# score = client("cat.jpg", "a dog playing")
# print(f"Similarity: {score:.2f}")

# # With PIL Image
# from PIL import Image
# img = Image.open("photo.jpg")
# score = client(img, "beautiful sunset")
# print(f"Similarity: {score:.2f}")

# # Compare multiple descriptions
# descriptions = [
#     "a cat sleeping on the couch",
#     "a dog running in the park",
#     "a car on the street"
# ]
# for desc in descriptions:
#     score = client("cat.jpg", desc)
#     print(f"{desc}: {score:.2f}")
