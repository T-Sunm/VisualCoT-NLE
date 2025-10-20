"""
BLIP2-based scoring functions
Scores answer/thought quality using BLIP2 VQA
"""

from typing import Optional, List, Union
from PIL import Image


def score_answer_with_blip2(
    answer: str,
    image_path: str,
    question: str,
    blip2_captioner,
) -> float:
    """
    Score answer quality using BLIP2.

    Args:
        answer: Predicted answer
        image_path: Path to image
        question: Original question
        blip2_captioner: BLIP2Captioner instance

    Returns:
        Confidence score (0.0-1.0)
    """
    # Ask BLIP2 the question and compare with predicted answer
    prompt = f"Question: {question} Answer:"

    blip2_answer = blip2_captioner.query_basic(
        image=image_path, prompt=prompt, use_pred_answer=True
    )[0]

    # Simple string matching (can be improved with semantic similarity)
    if answer.lower() in blip2_answer.lower() or blip2_answer.lower() in answer.lower():
        return 1.0
    else:
        return 0.0


def verify_thought_with_blip2(
    thought: str,
    image_path: str,
    blip2_captioner,
) -> float:
    """
    Verify if a thought matches the image using BLIP2.

    Args:
        thought: Thought/reasoning to verify
        image_path: Path to image
        blip2_captioner: BLIP2Captioner instance

    Returns:
        Confidence score (1.0 if matches, 0.0 if not)
    """
    prompt = (
        f"Question: Does this sentence match the facts in the picture? "
        f"Please answer yes or no. Sentence: In this picture, {thought} Answer:"
    )

    blip2_answer = blip2_captioner.query_basic(image=image_path, prompt=prompt)[0].lower()

    # Check if answer is yes
    if "yes" in blip2_answer:
        return 1.0
    else:
        return 0.0


def score_multiple_thoughts_with_blip2(
    thoughts: List[str],
    image_path: str,
    blip2_captioner,
) -> List[float]:
    """
    Score multiple thoughts using BLIP2.

    Args:
        thoughts: List of thoughts to verify
        image_path: Path to image
        blip2_captioner: BLIP2Captioner instance

    Returns:
        List of confidence scores
    """
    scores = []
    for thought in thoughts:
        score = verify_thought_with_blip2(thought, image_path, blip2_captioner)
        scores.append(score)
    return scores


def correct_thought_with_blip2(
    thought: str,
    image_path: str,
    blip2_captioner,
) -> str:
    """
    Correct a thought that doesn't match the image.

    Based on lines 671-683 from main_aokvqa.py

    Args:
        thought: Thought to correct
        image_path: Path to image
        blip2_captioner: BLIP2Captioner instance

    Returns:
        Corrected thought
    """
    # First verify if correction is needed
    if verify_thought_with_blip2(thought, image_path, blip2_captioner) > 0.5:
        return thought

    # Request correction
    prompt = (
        f"Question: Please correct the following sentence according to "
        f"the image. Sentence: {thought}"
    )

    correction = blip2_captioner.query_basic(image=image_path, prompt=prompt)[0]

    return correction


# Convenience function
def score(
    answer: str,
    image_path: str,
    question: Optional[str] = None,
    blip2_captioner=None,
) -> float:
    """
    Convenience function for BLIP2 scoring.

    Args:
        answer: Answer to score
        image_path: Path to image
        question: Optional question context
        blip2_captioner: BLIP2Captioner instance

    Returns:
        Confidence score
    """
    if blip2_captioner is None:
        raise ValueError("BLIP2 captioner required for scoring")

    if question:
        return score_answer_with_blip2(answer, image_path, question, blip2_captioner)
    else:
        return verify_thought_with_blip2(answer, image_path, blip2_captioner)
