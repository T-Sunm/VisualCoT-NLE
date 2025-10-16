import torch
from typing import Optional, List, Union, Dict
from PIL import Image
import openai
import time


class BLIP2Captioner:
    """
    BLIP2-based image captioner for generating various types of captions.
    Supports both local model inference and API-based inference.
    """

    def __init__(
        self,
        use_api: bool = False,
        api_urls: Optional[List[str]] = None,
        model_type: str = "pretrain_flant5xxl",
        device: Optional[str] = None,
        llm_engine: str = "chat",
        llm_engine_name: str = "gpt-3.5-turbo",
        apikey_list: Optional[List[str]] = None,
        debug: bool = False,
    ):
        """
        Initialize BLIP2 Captioner.

        Args:
            use_api: Whether to use API instead of local model
            api_urls: List of API endpoints for BLIP2 service
            model_type: BLIP2 model type (pretrain_flant5xl or pretrain_flant5xxl)
            device: Device to run model on (cuda/cpu)
            llm_engine: LLM engine for question generation (chat, gpt3, etc.)
            llm_engine_name: Specific LLM model name
            apikey_list: List of OpenAI API keys
            debug: Enable debug mode
        """
        self.use_api = use_api
        self.llm_engine = llm_engine
        self.llm_engine_name = llm_engine_name
        self.debug = debug

        # API key management for OpenAI
        self.apikey_list = apikey_list or []
        self.apikey_idx = 0
        if self.apikey_list:
            openai.api_key = self.apikey_list[self.apikey_idx]

        if use_api:
            # Use API-based inference
            self.blip2_api = api_urls or ["http://localhost:5000/api/generate"]
            self.blip2_model = None
            self.blip2_vis_processors = None
            self.blip2_device = None
        else:
            # Load local BLIP2 model
            from lavis.models import load_model_and_preprocess

            self.blip2_device = device or (
                torch.device("cuda") if torch.cuda.is_available() else "cpu"
            )

            self.blip2_model, self.blip2_vis_processors, _ = load_model_and_preprocess(
                name="blip2_t5", model_type=model_type, is_eval=True, device=self.blip2_device
            )
            print(f"BLIP2 model loaded successfully on {self.blip2_device}")

        # Current image cache
        self.current_blip2_image = None
        self.current_conversation = []
        self.current_global_caption = None

    def sleep(self, sleep_time: float = 1.5, switch_key: bool = False):
        """Sleep between API calls and optionally switch API key."""
        if switch_key and self.apikey_list:
            self.apikey_idx = (self.apikey_idx + 1) % len(self.apikey_list)
            openai.api_key = self.apikey_list[self.apikey_idx]
        time.sleep(sleep_time)

    def preprocess_image(self, image: Union[str, Image.Image]) -> Union[torch.Tensor, Image.Image]:
        """
        Preprocess image for BLIP2 model.

        Args:
            image: PIL Image or path to image

        Returns:
            Preprocessed image tensor (local) or PIL Image (API)
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        if self.use_api:
            self.current_blip2_image = image
        else:
            self.current_blip2_image = (
                self.blip2_vis_processors["eval"](image).unsqueeze(0).to(self.blip2_device)
            )

        return self.current_blip2_image

    def query_basic(
        self,
        image: Optional[Union[torch.Tensor, Image.Image]] = None,
        prompt: str = "",
        use_pred_answer: bool = False,
    ) -> List[str]:
        """
        Basic BLIP2 query method.

        Args:
            image: Image to query (uses current_blip2_image if None)
            prompt: Text prompt for the query
            use_pred_answer: Whether to use predict_answers mode

        Returns:
            List of generated text responses
        """
        if image is None:
            image = self.current_blip2_image

        if not self.use_api:
            # Local model inference
            if use_pred_answer:
                output = self.blip2_model.predict_answers(
                    {"image": image, "text_input": prompt}, max_len=25
                )
            else:
                output = self.blip2_model.generate({"image": image, "text_input": prompt})

            if self.debug:
                print(f"BLIP2 Query: {prompt}")
                print(f"BLIP2 Output: {output}")
        else:
            # API-based inference
            import utils_api

            output = utils_api.blip_completev2(
                images=[image],
                texts=[prompt],
                blip_urls=self.blip2_api,
                num_beams=5,
                length_penalty=-1.0,
                encoding_format="PNG",
            )

        return output

    def generate_global_caption(
        self, image: Optional[Union[torch.Tensor, Image.Image]] = None, question: str = ""
    ) -> str:
        """
        Generate global image caption.

        Args:
            image: Image to caption
            question: Optional question context

        Returns:
            Global caption string
        """
        if image is not None:
            self.preprocess_image(image)

        # Generate general caption
        global_caption = self.query_basic(prompt="An image of ")[0]

        # Generate question-aware caption
        global_caption_question = self.query_basic(
            prompt=f"Question: Please look at the picture and answer the following question. {question} Answer:",
            use_pred_answer=True,
        )[0]

        full_caption = ". ".join([global_caption, global_caption_question])

        if self.debug:
            print(f"Global Caption: {full_caption}")

        self.current_global_caption = full_caption
        return full_caption

    def generate_local_caption(
        self, obj_name: str, question: str, image: Optional[Union[torch.Tensor, Image.Image]] = None
    ) -> str:
        """
        Generate local caption for a specific object in the image.

        Args:
            obj_name: Name of the object to focus on
            question: Main question being answered
            image: Image to analyze

        Returns:
            Local caption describing the object
        """
        if image is not None:
            self.preprocess_image(image)

        # Get detailed description of the object
        local_caption_raw = self.query_basic(
            prompt=f"Question: Look at the {obj_name} in this image. "
            f"Please give a detailed description of the {obj_name} in this image. Answer:",
            use_pred_answer=True,
        )[0]

        # Generate follow-up question using LLM
        question_from_llm = self._generate_followup_question(obj_name, local_caption_raw, question)

        # Get answer to the follow-up question
        local_caption_question = self.query_basic(
            prompt=f"Question: Please look at the {obj_name} and answer the following question. "
            f"{question_from_llm} Answer:",
            use_pred_answer=True,
        )[0]

        # Combine all information
        local_caption = ". ".join(
            [local_caption_raw, question_from_llm + " The answer is " + local_caption_question]
        )

        if self.debug:
            print(f"Local Caption for {obj_name}: {local_caption}")

        return local_caption

    def _generate_followup_question(
        self, obj_name: str, observation: str, main_question: str
    ) -> str:
        """
        Generate a follow-up question about the object using LLM.

        Args:
            obj_name: Object name
            observation: Initial observation about the object
            main_question: Main question being answered

        Returns:
            Follow-up question string
        """
        if self.llm_engine == "chat":
            self.current_conversation.append(
                {
                    "role": "user",
                    "content": f"You will to look at the {obj_name} in the picture and find {observation}. "
                    f"To find the answer to {main_question}, you can ask one question about the {obj_name}. "
                    f"Please tell me the question you want to ask directly.",
                }
            )

            successful = False
            while not successful:
                try:
                    self.sleep()
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=self.current_conversation,
                        max_tokens=40,
                        temperature=0.0,
                        stream=False,
                    )
                    successful = True
                except Exception as e:
                    print(f"OpenAI API Error: {e}")
                    self.sleep(switch_key=True)

            question_from_llm = response["choices"][0]["message"]["content"]

        elif self.llm_engine == "chat-test":
            # Test mode
            self.current_conversation.append(
                {
                    "role": "user",
                    "content": f"You will to look at the {obj_name} in the picture and find {observation}. "
                    f"To find the answer to {main_question}, you can ask one question about the {obj_name}. "
                    f"Please tell me the question you want to ask directly.",
                }
            )
            question_from_llm = "Who are you?"

        elif self.llm_engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct"]:
            prompt = (
                f"I look at the {obj_name} in the picture and find {observation}. "
                f"To find the answer to {main_question}, I ask one question about the {obj_name}. "
                f"My question is:"
            )

            successful = False
            while not successful:
                try:
                    self.sleep()
                    response = openai.Completion.create(
                        engine=self.llm_engine_name,
                        prompt=prompt,
                        max_tokens=41,
                        logprobs=1,
                        temperature=0.0,
                        stream=False,
                        stop=["<|endoftext|>", "?", " ?"],
                    )
                    successful = True
                except Exception as e:
                    print(f"OpenAI API Error: {e}")
                    self.sleep(switch_key=True)

            question_from_llm = response["choices"][0]["text"].strip() + "?"
        else:
            question_from_llm = "Empty"

        return question_from_llm

    def detect_objects(
        self, image: Optional[Union[torch.Tensor, Image.Image]] = None, max_objects: int = 10
    ) -> List[List]:
        """
        Detect objects in the image using BLIP2.

        Args:
            image: Image to analyze
            max_objects: Maximum number of objects to detect

        Returns:
            List of [confidence, object_name] pairs
        """
        if image is not None:
            self.preprocess_image(image)

        obj_list = []

        while len(obj_list) < max_objects:
            if len(obj_list) == 0:
                tmp_obj_name_list = self.query_basic(
                    prompt="Give me the name of one object, creature, or entity in the image."
                )
            else:
                # Ask for objects besides already detected ones
                tmp_prompt = (
                    "Give me the name of one object, creature, or entity in the image besides"
                )
                for tmp_idx, tmp_name in enumerate(obj_list):
                    tmp_prompt += f" {tmp_name}"
                    tmp_prompt += "," if tmp_idx < len(obj_list) - 1 else "?"

                tmp_obj_name_list = self.query_basic(prompt=tmp_prompt)

            # Parse object names
            tmp_obj_name_list_refine = self._parse_object_names(tmp_obj_name_list)

            if self.debug:
                print(f"Detected objects: {tmp_obj_name_list_refine}")

            # Add new objects
            all_exist_flag = True
            for obj_name in tmp_obj_name_list_refine:
                if obj_name not in obj_list:
                    obj_list.append(obj_name)
                    all_exist_flag = False

            # Stop if no new objects found
            if all_exist_flag:
                break

        # Remove duplicates and format output
        obj_list = list(set(obj_list))
        attr_list = [[1.0, obj_name] for obj_name in obj_list]

        if self.debug:
            print(f"Final object list: {attr_list}")

        return attr_list

    def _parse_object_names(self, raw_result_list: List) -> List[str]:
        """
        Parse object names from raw results, removing articles.

        Args:
            raw_result_list: Raw results from BLIP2

        Returns:
            Cleaned list of object names
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

        # Remove articles (a, an, the)
        output_list = [ele[2:] if ele.lower().startswith("a ") else ele for ele in output_list]
        output_list = [ele[3:] if ele.lower().startswith("an ") else ele for ele in output_list]
        output_list = [ele[4:] if ele.lower().startswith("the ") else ele for ele in output_list]

        # Clean up
        output_list = [ele.strip() for ele in output_list]
        output_list = [ele for ele in output_list if len(ele) > 0]

        return output_list

    def verify_thought_with_image(
        self, thought: str, image: Optional[Union[torch.Tensor, Image.Image]] = None
    ) -> str:
        """
        Verify if a thought/reasoning matches the image content.
        Corrects the thought if it doesn't match.

        Args:
            thought: Thought or reasoning to verify
            image: Image to verify against

        Returns:
            Original thought if matches, corrected thought otherwise
        """
        if image is not None:
            self.preprocess_image(image)

        # Check if thought matches image
        blip2_answer = self.query_basic(
            prompt=f"Question: Does this sentence match the facts in the picture? "
            f"Please answer yes or no. Sentence: In this picture, {thought} Answer:"
        )[0]

        if self.debug:
            print(f"Thought verification: {blip2_answer} for '{thought}'")

        if blip2_answer.lower() == "no":
            # Get correction
            correction = self.query_basic(
                prompt=f"Question: Please correct the following sentence according to "
                f"the image. Sentence: {thought}"
            )[0]
            return correction
        else:
            return thought

    def caption_image(self, image_path: str) -> str:
        """
        Simple wrapper for backward compatibility.
        Generate a basic caption for an image.

        Args:
            image_path: Path to image file

        Returns:
            Generated caption
        """
        image = Image.open(image_path).convert("RGB")
        self.preprocess_image(image)
        return self.generate_global_caption(question="")


# Convenience function for simple captioning
def caption_image(image_path: str, **kwargs) -> str:
    """
    Simple function to caption an image using BLIP2.

    Args:
        image_path: Path to image
        **kwargs: Additional arguments for BLIP2Captioner

    Returns:
        Generated caption
    """
    captioner = BLIP2Captioner(**kwargs)
    return captioner.caption_image(image_path)
