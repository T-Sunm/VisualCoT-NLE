"""Question answerer with chain-of-thought reasoning."""

from typing import Dict, List, Optional, Tuple

from ..llm import BaseLLMAdapter
from ..prompts import (
    QuestionAnsweringPromptBuilder,
    extract_answer_and_rationale,
    extract_logprobs_until_stop,
    process_answer,
)


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
        question: str,
        context: str,
        scene_graph_text: str,
        choices: Optional[List[str]] = None,
        examples: Optional[List[Dict]] = None,
        thoughts: Optional[List[str]] = None,
    ) -> Tuple[str, Optional[str], float]:
        """
        Answer a question with reasoning.

        Args:
            question: Question to answer
            context: Global image context/caption
            scene_graph_text: Scene graph description
            choices: Multiple choice options
            examples: Few-shot examples
            thoughts: Previous thoughts for iterative reasoning

        Returns:
            Tuple of (answer, rationale, confidence_score)
        """
        pred_answers = []
        pred_rationales = []
        pred_logprobs = []

        # Ensemble multiple runs
        for _ in range(self.n_ensemble):
            answer, rationale, logprob = self._answer_single(
                question=question,
                context=context,
                scene_graph_text=scene_graph_text,
                choices=choices,
                examples=examples,
                thoughts=thoughts,
            )

            pred_answers.append(answer)
            pred_rationales.append(rationale)
            pred_logprobs.append(logprob)

        # Select best answer by logprob
        best_idx = pred_logprobs.index(max(pred_logprobs))

        return pred_answers[best_idx], pred_rationales[best_idx], pred_logprobs[best_idx]

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

        if self.debug:
            print(f"[QuestionAnswerer] Prompt:\n{prompt}")

        # Generate based on engine type
        if self.engine in ["ada", "babbage", "curie", "davinci", "codex", "instruct", "gpt3"]:
            return self._answer_with_gpt3(prompt)
        elif self.engine == "chat":
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
            answer, rationale = extract_answer_and_rationale(text, chain_of_thoughts=True)

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
        # Get system prompt
        system_prompt = self.prompt_builder.QA_SYSTEM_PROMPT_CHAT if self.engine == "chat" else None

        # Generate
        response = self.llm.generate(prompt=prompt, max_tokens=40, system_prompt=system_prompt)

        text = response.text

        if self.chain_of_thoughts:
            answer, rationale = extract_answer_and_rationale(text, chain_of_thoughts=True)
            return answer, rationale, 0.0  # Chat API doesn't return logprobs
        else:
            answer = process_answer(text)
            return answer, None, 0.0

    def _answer_with_local_model(self, prompt: str) -> Tuple[str, Optional[str], float]:
        """Answer using local models (OPT, LLaMA)."""
        # Generate
        response = self.llm.generate(prompt=prompt, max_tokens=40)

        text = response.text.split("\n")[0]

        if self.chain_of_thoughts:
            # Extract answer from "The answer is X. Rationale."
            if "The answer is" in text:
                text = text.split("The answer is")[-1]

            answer, rationale = extract_answer_and_rationale(text, chain_of_thoughts=True)

            # Extract logprobs
            if response.logprobs:
                total_logprob = sum(response.logprobs)
            else:
                total_logprob = 0.0

            return answer, rationale, total_logprob
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
        question: str,
        context: str,
        scene_graph_text: str,
        choices: Optional[List[str]] = None,
        examples: Optional[List[Dict]] = None,
        thoughts: Optional[List[str]] = None,
    ) -> Tuple[str, Optional[str], float]:
        """Answer with ensemble strategy."""
        # Get all predictions
        pred_answers = []
        pred_rationales = []
        pred_logprobs = []

        for _ in range(self.n_ensemble):
            answer, rationale, logprob = self._answer_single(
                question=question,
                context=context,
                scene_graph_text=scene_graph_text,
                choices=choices,
                examples=examples,
                thoughts=thoughts,
            )

            pred_answers.append(answer)
            pred_rationales.append(rationale)
            pred_logprobs.append(logprob)

        # Apply ensemble strategy
        if self.ensemble_strategy == "max_logprob":
            best_idx = pred_logprobs.index(max(pred_logprobs))
            return pred_answers[best_idx], pred_rationales[best_idx], pred_logprobs[best_idx]

        elif self.ensemble_strategy == "majority_vote":
            # Count occurrences
            answer_counts = {}
            for ans in pred_answers:
                answer_counts[ans] = answer_counts.get(ans, 0) + 1

            # Get most common
            best_answer = max(answer_counts.items(), key=lambda x: x[1])[0]

            # Find first occurrence for rationale
            best_idx = pred_answers.index(best_answer)
            return best_answer, pred_rationales[best_idx], pred_logprobs[best_idx]

        elif self.ensemble_strategy == "weighted_vote":
            # Weight answers by logprobs
            answer_weights = {}
            answer_rationales = {}

            for ans, rat, logprob in zip(pred_answers, pred_rationales, pred_logprobs):
                if ans not in answer_weights:
                    answer_weights[ans] = 0.0
                    answer_rationales[ans] = rat
                answer_weights[ans] += logprob

            # Get best weighted answer
            best_answer = max(answer_weights.items(), key=lambda x: x[1])[0]

            return best_answer, answer_rationales[best_answer], answer_weights[best_answer]

        else:
            # Default: max logprob
            best_idx = pred_logprobs.index(max(pred_logprobs))
            return pred_answers[best_idx], pred_rationales[best_idx], pred_logprobs[best_idx]
