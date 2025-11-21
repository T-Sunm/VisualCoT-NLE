def extract_explanation(thought: str) -> str:
        """Trích xuất phần Explanation từ thought"""
        # Format mong đợi: "Answer: ... Explanation: ..."
        if "Explanation:" in thought:
            return thought.split("Explanation:")[-1].strip()
        return thought 