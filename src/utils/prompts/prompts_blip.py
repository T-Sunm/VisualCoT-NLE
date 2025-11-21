"""
BLIP Prompt Builder - Tạo prompts cho BLIP2 model
"""


class BLIPPromptBuilder:
    """Builder để tạo các prompts cho BLIP2"""
    
    @staticmethod
    def global_caption() -> str:
        """Prompt để tạo global caption cho toàn bộ ảnh"""
        return "An image of "
    
    @staticmethod
    def global_caption_with_question(question: str) -> str:
        """Prompt để caption ảnh dựa trên câu hỏi"""
        return (
            f"Question: Please look at the picture and answer the following question. "
            f"{question} Answer:"
        )
    
    @staticmethod
    def object_description(obj_name: str) -> str:
        """Prompt để mô tả chi tiết một object"""
        return (
            f"Question: Look at the {obj_name} in this image. "
            f"Please give a detailed description of the {obj_name} in this image. Answer:"
        )
    
    @staticmethod
    def object_question_answer(obj_name: str, question: str) -> str:
        """Prompt để trả lời câu hỏi về một object cụ thể"""
        return (
            f"Question: Please look at the {obj_name} and answer the following question. "
            f"{question} Answer:"
        )
    
    @staticmethod
    def verify_thought(thought: str) -> str:
        """Prompt để verify xem thought có khớp với ảnh không"""
        return (
            f"Question: Does this sentence match the facts in the picture? "
            f"Please answer yes or no. Sentence: In this picture, {thought} Answer:"
        )
    
    @staticmethod
    def correct_thought(thought: str) -> str:
        """Prompt để sửa lại thought không khớp với ảnh"""
        return (
            f"Question: Please correct the following sentence according to "
            f"the image. Sentence: {thought}"
        )


# ===== USAGE =====
if __name__ == "__main__":
    builder = BLIPPromptBuilder()
    
    # 1. Global caption
    print(builder.global_caption())
    # Output: "An image of "
    
    # 2. Caption với question
    print(builder.global_caption_with_question("What is the person doing?"))
    # Output: "Question: Please look at the picture and answer the following question. What is the person doing? Answer:"
    
    # 3. Mô tả object
    print(builder.object_description("ski"))
    # Output: "Question: Look at the ski in this image. Please give a detailed description of the ski in this image. Answer:"
    
    # 4. Trả lời câu hỏi về object
    print(builder.object_question_answer("ski", "What color is it?"))
    # Output: "Question: Please look at the ski and answer the following question. What color is it? Answer:"
    
    # 5. Verify thought
    print(builder.verify_thought("the person is skiing"))
    # Output: "Question: Does this sentence match the facts in the picture? Please answer yes or no. Sentence: In this picture, the person is skiing Answer:"
    
    # 6. Correct thought
    print(builder.correct_thought("the person is swimming"))
    # Output: "Question: Please correct the following sentence according to the image. Sentence: the person is swimming"