import torch
from typing import Optional, List, Union
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration


class BLIP2Captioner:
    """
    BLIP2-based image captioner đơn giản chỉ dùng Hugging Face.
    Hỗ trợ tạo caption và phát hiện object cho Visual Question Answering.
    """

    def __init__(
        self,
        model_type: str = "Salesforce/blip2-opt-2.7b",
        device: Optional[str] = None,
        debug: bool = False,
    ):
        """
        Khởi tạo BLIP2 Captioner với Hugging Face.

        Args:
            model_type: Tên mô hình BLIP2 trên Hugging Face
            device: Device để chạy model (cuda/cpu)
            debug: Bật chế độ debug
        """
        self.debug = debug
        
        # Tự động chọn device
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Loading BLIP2 model {model_type} on {self.device}...")
        
        # Load processor và model từ Hugging Face
        self.processor = Blip2Processor.from_pretrained(model_type)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_type, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)
        
        print(f"BLIP2 model loaded successfully on {self.device}")

        # Cache cho ảnh hiện tại
        self.current_image = None
        self.current_inputs = None
        self.current_global_caption = None

    def preprocess_image(self, image: Union[str, Image.Image]) -> torch.Tensor:
        """
        Tiền xử lý ảnh cho BLIP2 model.

        Args:
            image: PIL Image hoặc đường dẫn đến ảnh

        Returns:
            Tensor ảnh đã được tiền xử lý
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        self.current_image = image
        return image

    def query_basic(
        self,
        image: Optional[Union[torch.Tensor, Image.Image]] = None,
        prompt: str = "",
        use_pred_answer: bool = False,
        max_new_tokens: int = 32,
    ) -> List[str]:
        """
        Phương thức query cơ bản với BLIP2.

        Args:
            image: Ảnh (dùng current_image nếu None)
            prompt: Prompt text cho query
            use_pred_answer: Có dùng chế độ QA không
            max_new_tokens: Số token tối đa tạo ra

        Returns:
            Danh sách các câu trả lời được tạo
        """
        if image is None:
            image = self.current_image
        
        # Xử lý inputs
        inputs = self.processor(image, prompt, return_tensors="pt").to(self.device)
        self.current_inputs = inputs
        
        # Sinh text
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
            )
        
        # Giải mã outputs
        generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)
        
        if self.debug:
            print(f"BLIP2 Query: {prompt}")
            print(f"BLIP2 Output: {generated_text}")
        
        return generated_text

    def generate_global_caption(
        self, 
        image: Optional[Union[torch.Tensor, Image.Image]] = None, 
        question: str = ""
    ) -> str:
        """
        Tạo caption toàn cảnh cho ảnh.

        Args:
            image: Ảnh cần caption
            question: Câu hỏi làm context (tùy chọn)

        Returns:
            Caption toàn cảnh
        """
        if image is not None:
            self.preprocess_image(image)

        # Tạo caption chung
        global_caption = self.query_basic(prompt="An image of ")[0]

        # Tạo caption liên quan đến câu hỏi (nếu có)
        if question:
            global_caption_question = self.query_basic(
                prompt=f"Question: Please look at the picture and answer the following question. {question} Answer:",
            )[0]
            full_caption = ". ".join([global_caption, global_caption_question])
        else:
            full_caption = global_caption

        if self.debug:
            print(f"Global Caption: {full_caption}")

        self.current_global_caption = full_caption
        return full_caption

    def generate_local_caption(
        self, 
        obj_name: str, 
        question: str, 
        image: Optional[Union[torch.Tensor, Image.Image]] = None
    ) -> str:
        """
        Tạo caption cục bộ cho một object cụ thể trong ảnh.

        Args:
            obj_name: Tên object cần focus
            question: Câu hỏi chính
            image: Ảnh cần phân tích

        Returns:
            Caption mô tả object
        """
        if image is not None:
            self.preprocess_image(image)

        # Lấy mô tả chi tiết của object
        local_caption = self.query_basic(
            prompt=f"Question: Look at the {obj_name} in this image. "
            f"Please give a detailed description of the {obj_name} in this image. Answer:",
            max_new_tokens=64,
        )[0]

        if self.debug:
            print(f"Local Caption for {obj_name}: {local_caption}")

        return local_caption

    def detect_objects(
        self, 
        image: Optional[Union[torch.Tensor, Image.Image]] = None, 
        max_objects: int = 10
    ) -> List[List]:
        """
        Phát hiện các object trong ảnh sử dụng BLIP2.

        Args:
            image: Ảnh cần phân tích
            max_objects: Số object tối đa cần phát hiện

        Returns:
            Danh sách [[confidence, object_name], ...] 
        """
        if image is not None:
            self.preprocess_image(image)

        obj_list = []

        while len(obj_list) < max_objects:
            if len(obj_list) == 0:
                tmp_obj_name_list = self.query_basic(
                    prompt="Give me the name of one object, creature, or entity in the image.",
                    max_new_tokens=20,
                )
            else:
                # Hỏi về các object ngoài những cái đã phát hiện
                tmp_prompt = (
                    "Give me the name of one object, creature, or entity in the image besides"
                )
                for tmp_idx, tmp_name in enumerate(obj_list):
                    tmp_prompt += f" {tmp_name}"
                    tmp_prompt += "," if tmp_idx < len(obj_list) - 1 else "?"

                tmp_obj_name_list = self.query_basic(prompt=tmp_prompt, max_new_tokens=20)

            # Parse tên object
            tmp_obj_name_list_refine = self._parse_object_names(tmp_obj_name_list)

            if self.debug:
                print(f"Detected objects: {tmp_obj_name_list_refine}")

            # Thêm object mới
            all_exist_flag = True
            for obj_name in tmp_obj_name_list_refine:
                if obj_name not in obj_list:
                    obj_list.append(obj_name)
                    all_exist_flag = False

            # Dừng nếu không tìm thấy object mới
            if all_exist_flag:
                break

        # Loại bỏ trùng lặp và format output
        obj_list = list(set(obj_list))
        attr_list = [[1.0, obj_name] for obj_name in obj_list]

        if self.debug:
            print(f"Final object list: {attr_list}")

        return attr_list

    def _parse_object_names(self, raw_result_list: List) -> List[str]:
        """
        Parse tên object từ kết quả thô, loại bỏ mạo từ.

        Args:
            raw_result_list: Kết quả thô từ BLIP2

        Returns:
            Danh sách tên object đã clean
        """
        def parse_recursive(raw_result):
            output_list = []
            for raw in raw_result if isinstance(raw_result, list) else [raw_result]:
                if isinstance(raw, str):
                    raw = raw.strip()
                    tmp_result_list = raw.split(",")
                    for tmp_result in tmp_result_list:
                        output_list.extend(tmp_result.split(" and "))
                elif isinstance(raw, list):
                    output_list.extend(parse_recursive(raw))
            return output_list

        output_list = parse_recursive(raw_result_list)

        # Loại bỏ mạo từ (a, an, the)
        output_list = [ele[2:] if ele.lower().startswith("a ") else ele for ele in output_list]
        output_list = [ele[3:] if ele.lower().startswith("an ") else ele for ele in output_list]
        output_list = [ele[4:] if ele.lower().startswith("the ") else ele for ele in output_list]

        # Dọn dẹp
        output_list = [ele.strip() for ele in output_list]
        output_list = [ele for ele in output_list if len(ele) > 0]

        return output_list

    def verify_thought_with_image(
        self, 
        thought: str, 
        image: Optional[Union[torch.Tensor, Image.Image]] = None
    ) -> str:
        """
        Xác minh xem một thought/reasoning có khớp với nội dung ảnh không.
        Sửa lại thought nếu không khớp.

        Args:
            thought: Thought hoặc reasoning cần xác minh
            image: Ảnh để xác minh

        Returns:
            Thought gốc nếu khớp, thought đã sửa nếu không khớp
        """
        if image is not None:
            self.preprocess_image(image)

        # Kiểm tra thought có khớp với ảnh không
        blip2_answer = self.query_basic(
            prompt=f"Question: Does this sentence match the facts in the picture? "
            f"Please answer yes or no. Sentence: In this picture, {thought} Answer:",
            max_new_tokens=10,
        )[0].lower()

        if self.debug:
            print(f"Thought verification: {blip2_answer} for '{thought}'")

        if "no" in blip2_answer:
            # Lấy bản sửa lại
            correction = self.query_basic(
                prompt=f"Question: Please correct the following sentence according to "
                f"the image. Sentence: {thought}",
                max_new_tokens=64,
            )[0]
            return correction
        else:
            return thought


# Function tiện ích để caption ảnh đơn giản
def caption_image(image_path: str, **kwargs) -> str:
    """
    Function đơn giản để caption ảnh sử dụng BLIP2.

    Args:
        image_path: Đường dẫn đến ảnh
        **kwargs: Các tham số bổ sung cho BLIP2Captioner

    Returns:
        Caption được tạo
    """
    captioner = BLIP2Captioner(**kwargs)
    image = Image.open(image_path).convert("RGB")
    captioner.preprocess_image(image)
    return captioner.generate_global_caption(question="")