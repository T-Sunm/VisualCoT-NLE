"""Question answerer with chain-of-thought reasoning."""

from typing import Dict, List, Optional, Tuple, Any

from ..llm import BaseLLMAdapter
from ..prompts import (
    QuestionAnsweringPromptBuilder,
    extract_answer_and_rationale,
    extract_logprobs_until_stop,
    process_answer,
)
from ..prompts import templates


class QuestionAnswerer:
    """Answer questions using chain-of-thought reasoning."""

    def __init__(
        self,
        llm: BaseLLMAdapter,
        engine: str = "gpt3",
        chain_of_thoughts: bool = True,
        choice_only: bool = False,
        remove_caption: bool = False,
        n_ensemble: int = 1,
        debug: bool = False,
    ):
        """
        Initialize question answerer.

        Args:
            llm: LLM adapter
            engine: Engine type
            chain_of_thoughts: Use chain-of-thought reasoning
            choice_only: Multiple choice mode
            remove_caption: Remove caption from context
            n_ensemble: Number of ensemble runs
            debug: Debug mode
        """
        self.llm = llm
        self.engine = engine
        self.chain_of_thoughts = chain_of_thoughts
        self.choice_only = choice_only
        self.remove_caption = remove_caption
        self.n_ensemble = n_ensemble
        self.debug = debug

        # Prompt builder
        self.prompt_builder = QuestionAnsweringPromptBuilder(
            engine=engine,
            chain_of_thoughts=chain_of_thoughts,
            choice_only=choice_only,
            remove_caption=remove_caption,
        )

    def answer(
        self,
        sample: Dict[str, Any],
        visual_context: List[str],
        thoughts: List[str] = None,
    ) -> Dict[str, Any]:
        """Generate a final answer."""
        question = sample["question"]
        key = sample.get("key")
        choices = sample.get("choices")
        train_context = sample.get("train_context", {})

        pred_answer_list, pred_prob_list, thought_list = [], [], []

        for _ in range(self.n_ensemble):
            # Split visual_context into context and scene_graph_text
            context = visual_context[0] if len(visual_context) > 0 else ""
            scene_graph_text = visual_context[1] if len(visual_context) > 1 else ""

            # Get examples from train_context
            examples = train_context.get("examples", [])

            # Build prompt using correct method name 'build'
            prompt = self.prompt_builder.build(
                question=question,
                context=context,
                scene_graph_text=scene_graph_text,
                choices=choices,
                examples=examples,
                thoughts=thoughts,
            )

            # Generate response
            response = self.llm.generate(prompt)

            # Extract text from response
            response_text = response.text if hasattr(response, "text") else str(response)
            
            # Parse response
            answer, rationale, confidence = extract_answer_and_rationale(
                response_text, chain_of_thoughts=True
            )

            pred_answer_list.append(answer if answer else "")
            pred_prob_list.append(confidence)
            thought_list.append(rationale if rationale else "")

        if not pred_prob_list:
            return {
                "answer": "Error: No answer generated",
                "rationale": "",
                "prompt": prompt,
            }

        best_idx = pred_prob_list.index(max(pred_prob_list))
        final_answer = pred_answer_list[best_idx]
        final_thought = thought_list[best_idx]

        return {
            "answer": final_answer,
            "rationale": final_thought,
            "prompt": prompt,
            "raw_response": response_text,
        }

    def _answer_single(
        self,
        question: str,
        context: str,
        scene_graph_text: str,
        choices: Optional[List[str]],
        examples: Optional[List[Dict]],
        thoughts: Optional[List[str]],
    ) -> Tuple[str, Optional[str], float]:
        """Single answer generation run."""
        # Build prompt
        prompt = self.prompt_builder.build(
            question=question,
            context=context,
            scene_graph_text=scene_graph_text,
            choices=choices,
            examples=examples,
            thoughts=thoughts,
        )

        # Generate based on engine type
        if self.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
            return self._answer_with_gpt3(prompt)
        elif self.engine in ["chat", "groq"]:
            return self._answer_with_chat(prompt)
        elif self.engine in ["opt", "llama", "bloom"]:
            return self._answer_with_local_model(prompt)
        elif self.engine == "chat-test":
            return "fake answer", "This is a fake thought.", 0.0
        else:
            return "unknown", None, 0.0

    def _answer_with_gpt3(self, prompt: str) -> Tuple[str, Optional[str], float]:
        """Answer using GPT-3 Completion API."""
        # Generate
        response = self.llm.generate(
            prompt=prompt, max_tokens=41, stop_tokens=["\n", "<|endoftext|>"]
        )

        text = response.text
        tokens = response.tokens
        logprobs = response.logprobs

        if self.chain_of_thoughts:
            # Extract answer and rationale
            answer, rationale, confidence = extract_answer_and_rationale(
                text, chain_of_thoughts=True
            )

            # Extract logprobs until first period
            if tokens and logprobs:
                plist, _ = extract_logprobs_until_stop(tokens, logprobs, stop_token=".")
                total_logprob = sum(plist) if plist else 0.0
            else:
                total_logprob = 0.0

            return answer, rationale, total_logprob
        else:
            # Just answer
            answer = process_answer(text.split("\n")[0])

            # Sum all logprobs
            if logprobs:
                valid_logprobs = [lp for lp in logprobs if lp is not None]
                total_logprob = sum(valid_logprobs) if valid_logprobs else 0.0
            else:
                total_logprob = 0.0

            return answer, None, total_logprob

    def _answer_with_chat(self, prompt: str) -> Tuple[str, Optional[str], float]:
        """Answer using ChatGPT API."""
        # Get system prompt - FIX: use templates module, support groq
        system_prompt = templates.QA_SYSTEM_PROMPT_CHAT if self.engine in ["chat", "groq"] else None

        # Generate
        response = self.llm.generate(prompt=prompt, max_tokens=40, system_prompt=system_prompt)

        text = response.text

        if self.chain_of_thoughts:
            answer, rationale, confidence = extract_answer_and_rationale(
                text, chain_of_thoughts=True
            )
            print(f"[DEBUG LLM PARSED] Rationale: '{rationale}'")
            return answer, rationale, confidence
        else:
            answer = process_answer(text)
            return answer, None, confidence

    def _answer_with_local_model(self, prompt: str) -> Tuple[str, Optional[str], float]:
        """Answer using local models (OPT, LLaMA)."""
        # Generate
        response = self.llm.generate(prompt=prompt, max_tokens=40)

        text = response.text.split("\n")[0]

        if self.chain_of_thoughts:
            # Extract answer from "The answer is X. Rationale."
            if "The answer is" in text:
                text = text.split("The answer is")[-1]

            answer, rationale, confidence = extract_answer_and_rationale(
                text, chain_of_thoughts=True
            )

            # Extract logprobs
            if response.logprobs:
                total_logprob = sum(response.logprobs)
            else:
                total_logprob = 0.0

            return answer, rationale, confidence
        else:
            answer = process_answer(text)

            # Sum logprobs
            if response.logprobs:
                total_logprob = sum(response.logprobs)
            else:
                total_logprob = 0.0

            return answer, None, total_logprob


class EnsembleQuestionAnswerer(QuestionAnswerer):
    """Question answerer with advanced ensemble strategies."""

    def __init__(
        self,
        llm: BaseLLMAdapter,
        ensemble_strategy: str = "max_logprob",
        **kwargs,
    ):
        """
        Initialize ensemble answerer.

        Args:
            llm: LLM adapter
            ensemble_strategy: Strategy for combining answers
                - max_logprob: Select answer with highest logprob
                - majority_vote: Select most common answer
                - weighted_vote: Weight by logprobs
            **kwargs: Additional arguments for QuestionAnswerer
        """
        super().__init__(llm, **kwargs)
        self.ensemble_strategy = ensemble_strategy


def answer(
    self,
    sample: Dict[str, Any],
    visual_context: List[str],
    thoughts: List[str] = None,
) -> Dict[str, Any]:
    """Generate a final answer."""
    question = sample["question"]
    key = sample.get("key")
    choices = sample.get("choices")
    train_context = sample.get("train_context", {})

    pred_answer_list, pred_prob_list, thought_list = [], [], []

    for _ in range(self.n_ensemble):
        # Split visual_context into context and scene_graph
        context = visual_context[0] if len(visual_context) > 0 else ""
        scene_graph_text = visual_context[1] if len(visual_context) > 1 else ""

        # Get examples from train_context
        examples = train_context.get("examples", [])

        # Build prompt using correct method name and parameters
        prompt = self.prompt_builder.build(
            question=question,
            context=context,
            scene_graph_text=scene_graph_text,
            choices=choices,
            examples=examples,
            thoughts=thoughts,
        )

        # Generate response from LLM
        response = self.llm.generate(prompt)

        # Extract text from LLMResponse
        response_text = response.text if hasattr(response, "text") else str(response)

        # Parse response using formatters
        answer, rationale, confidence = extract_answer_and_rationale(
            response_text, chain_of_thoughts=self.chain_of_thoughts
        )

        # Build parsed response dict
        parsed_response = {
            "answer": answer if answer else "",
            "thought": rationale if rationale else "",
            "probability": getattr(response, "total_logprob", 0.0),
        }

        pred_answer = parsed_response.get("answer", "")
        thought = parsed_response.get("thought", "")
        prob = parsed_response.get("probability", 0.0)

        pred_answer_list.append(pred_answer)
        pred_prob_list.append(prob)
        thought_list.append(thought)

    if not pred_prob_list:
        return {
            "answer": "Error: No answer generated",
            "rationale": "",
            "prompt": prompt,
        }

    best_idx = pred_prob_list.index(max(pred_prob_list))
    final_answer = pred_answer_list[best_idx]
    final_thought = thought_list[best_idx]

    return {
        "answer": final_answer,
        "rationale": final_thought,
        "prompt": prompt,
    }


# if __name__ == "__main__":
#     """Test question answerer with Groq LLM."""
#     import os
#     from pathlib import Path

#     # Load .env file
#     try:
#         from dotenv import load_dotenv

#         # Find .env file in project root
#         project_root = Path(__file__).parent.parent.parent.parent.parent
#         env_path = project_root / ".env"
#         if env_path.exists():
#             load_dotenv(env_path)
#             print(f"✓ Loaded environment from: {env_path}")
#         else:
#             print(f"⚠ No .env file found at: {env_path}")
#             print(f"  Create one from example.env and add your GROQ_API_KEY")
#     except ImportError:
#         print("⚠ python-dotenv not installed. Install with: pip install python-dotenv")

#     from ..llm import create_llm_adapter

#     print("=" * 80)
#     print("TESTING QUESTION ANSWERER WITH GROQ")
#     print("=" * 80)

#     # Check for API key
#     groq_key = os.getenv("GROQ_API_KEY")
#     print(f"✓ API key found: {groq_key[:20]}...")

#     # Create Groq LLM adapter
#     llm = create_llm_adapter(
#         engine="groq",
#         engine_name="openai/gpt-oss-20b",
#         temperature=0.0,
#         max_tokens=100,
#         debug=True,
#     )

#     # Initialize question answerer
#     answerer = QuestionAnswerer(
#         llm=llm, engine="groq", chain_of_thoughts=True, choice_only=False, n_ensemble=1, debug=True
#     )

#     # Prepare test data
#     sample = {
#         "question": "What is the man doing?",
#         "choices": ["tennis", "soccer", "basketball"],
#         "key": "57<->123",
#         "train_context": {
#             "examples": [
#                 {
#                     "question": "What sport is this?",
#                     "answer": "tennis",
#                     "context": "A player on court with racket",
#                     "rationale": "The racket indicates tennis",
#                 }
#             ]
#         },
#     }

#     visual_context = [
#         "A man playing tennis on clay court",  # global caption
#         "man is standing. racket is green. court is red.",  # scene graph
#     ]

#     thoughts = ["The man holds a racket"]

#     print("\n--- Input ---")
#     print(f"Question: {sample['question']}")
#     print(f"Choices: {sample['choices']}")
#     print(f"Visual Context: {visual_context}")
#     print(f"Thoughts: {thoughts}")

#     # Call answerer
#     print("\n--- Generating Answer ---")
#     result = answerer.answer(sample, visual_context, thoughts)

#     print("\n--- Output ---")
#     print(f"Raw Response: {result['raw_response']}")
#     print(f"Answer: {result['answer']}")
#     print(f"Rationale: {result['rationale']}")
#     print(f"\n--- Prompt Preview ---")
#     print(result["prompt"][:500] + "..." if len(result["prompt"]) > 500 else result["prompt"])

#     # Validation
#     print("\n" + "=" * 80)
#     print("VALIDATION")
#     print("=" * 80)
#     assert "answer" in result, "❌ Result should have 'answer' key"
#     assert "rationale" in result, "❌ Result should have 'rationale' key"
#     assert "prompt" in result, "❌ Result should have 'prompt' key"
#     assert result["answer"], f"❌ Answer should not be empty"

#     print("✓ All validations passed!")
#     print(f"✓ Answer generated: '{result['answer']}'")
#     print("\n" + "=" * 80)
#     print("TEST COMPLETE - QUESTION ANSWERER WITH GROQ WORKS!")
#     print("=" * 80)
