from utils.models.blip import BLIPClient

class ObjectCaptioner:
    """Caption objects using BLIP"""
    
    def __init__(self, blip_client: BLIPClient):
        self.blip = blip_client
    
    def _postprocess_output(self, raw_output: str, prompt: str = None) -> str:
        """
        Clean BLIP output: remove prompt echoes and extra whitespace
        """
        result = raw_output.strip()
        
        # Remove common prompt patterns that BLIP sometimes echoes
        if prompt:
            # Remove "Question: ... Answer:" pattern
            if "Answer:" in result:
                result = result.split("Answer:")[-1].strip()
            elif "Question:" in result:
                result = result.split("Question:")[-1].strip()
        
        # Remove leading/trailing punctuation artifacts
        result = result.strip(" .,;:")
        
        return result
        
    def caption(self, image_path: str, object_name: str) -> str:
        """Caption một object cụ thể"""
        prompt = f"Question: Describe the {object_name} Answer:"
        raw_output = self.blip(image_path, prompt)
        processed_output = self._postprocess_output(raw_output, prompt)
        return processed_output
    
    def caption_global(self, image_path: str, question: str = None) -> str:
        """
        Tạo global caption gồm 2 phần:
        1. General Description: Mô tả chung cảnh (từ ảnh)
        2. Key Detail: Chi tiết quan trọng nhất trong ảnh (từ ảnh, KHÔNG dựa vào question)
        """
        
        # 1. General Description: Mô tả cảnh chung
        base_caption = self.blip(image_path, "An image of")
        base_caption = base_caption.strip()
        
        # 2. Key Detail: Trích xuất chi tiết quan trọng từ ảnh
        key_detail_prompt = "What is the main subject or action in this image? Answer:"
        
        key_detail = self.blip(image_path, key_detail_prompt)
        key_detail = self._postprocess_output(key_detail, key_detail_prompt).strip()
        return f"{base_caption}. Key detail: {key_detail}."