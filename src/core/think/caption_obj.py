from utils.models.blip import BLIPClient
from utils.prompts.prompts_blip import BLIPPromptBuilder


class ObjectCaptioner:
    """Sinh caption cho object sử dụng BLIP"""
    
    def __init__(self, blip_client: BLIPClient = None):
        self.blip = blip_client or BLIPClient()
        self.prompt_builder = BLIPPromptBuilder()
    
    def caption(self, image_path: str, obj_name: str) -> str:
        """
        Sinh caption chi tiết cho object trong ảnh
        
        Args:
            image_path: Đường dẫn đến ảnh
            obj_name: Tên object cần mô tả
            
        Returns:
            Caption mô tả object
        """
        prompt = self.prompt_builder.object_description(obj_name)

        print(f"[BLIP CAPTION] Prompt: {prompt}")
        caption = self.blip(image_path, prompt)
        print(f"[BLIP CAPTION] Output: {caption}")
        
        if "Answer:" in caption:
            caption = caption.split("Answer:")[-1].strip()
        return caption