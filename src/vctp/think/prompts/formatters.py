"""Answer formatters and parsers."""

from typing import List, Tuple, Optional


def process_answer(answer: str) -> str:
    """
    Clean and normalize answer text.

    This is used for initial cleaning of reference QA results.
    For final evaluation, use the official VQAv2 eval script.

    Args:
        answer: Raw answer string

    Returns:
        Cleaned answer string
    """
    # Remove punctuation
    answer = answer.replace(".", "").replace(",", "").lower()

    # Words to remove
    to_be_removed = {"a", "an", "the", "to", ""}

    # Split and filter
    answer_list = answer.split(" ")
    answer_list = [item for item in answer_list if item not in to_be_removed]

    return " ".join(answer_list)


def parse_sentence(raw_result_list: List) -> List[str]:
    """
    Parse comma and 'and' separated lists recursively.

    Args:
        raw_result_list: List of raw result strings or nested lists

    Returns:
        Flattened list of parsed items
    """
    output_list = []

    for raw_result in raw_result_list:
        if isinstance(raw_result, str):
            raw_result = raw_result.strip(" ")
            tmp_result_list = raw_result.split(",")
            tmp_output_list = []
            for tmp_result in tmp_result_list:
                tmp_output_list += tmp_result.split(" and ")
            output_list += tmp_output_list
        elif isinstance(raw_result, list):
            raw_output = parse_sentence(raw_result)
            output_list += raw_output

    # Clean up
    output_list = [ele.strip() for ele in output_list]
    output_list = [ele for ele in output_list if len(ele) > 0]

    return output_list


def parse_object_name(raw_result_list: List) -> List[str]:
    """
    Parse object names from raw results, removing articles.

    Args:
        raw_result_list: List of raw object name strings

    Returns:
        List of cleaned object names
    """
    output_list = []

    for raw_result in raw_result_list:
        if isinstance(raw_result, str):
            raw_result = raw_result.strip(" ")
            tmp_result_list = raw_result.split(",")
            tmp_output_list = []
            for tmp_result in tmp_result_list:
                tmp_output_list += tmp_result.split(" and ")
            output_list += tmp_output_list
        elif isinstance(raw_result, list):
            raw_output = parse_sentence(raw_result)
            output_list += raw_output

    # Remove articles
    output_list = [ele[2:] if ele.lower().startswith("a ") else ele for ele in output_list]
    output_list = [ele[3:] if ele.lower().startswith("an ") else ele for ele in output_list]
    output_list = [ele[4:] if ele.lower().startswith("the ") else ele for ele in output_list]

    # Clean up
    output_list = [ele.strip() for ele in output_list]
    output_list = [ele for ele in output_list if len(ele) > 0]

    return output_list


def extract_answer_and_rationale(
    response_text: str, chain_of_thoughts: bool = True
) -> Tuple[str, Optional[str], float]:
    """
    Extract answer and rationale from LLM response.

    Expected format:
        Answer: <answer text>
        Rationale: <rationale text>
        Confidence: 0.95

    Args:
        response_text: Raw response from LLM
        chain_of_thoughts: Whether CoT is used

    Returns:
        Tuple of (answer, rationale, confidence)
    """
    import re

    response_text = response_text.strip()

    # Extract answer (required)
    answer_match = re.search(r"answer:\s*(.+?)(?:\n|$)", response_text, re.IGNORECASE)
    if not answer_match:
        # Fallback: no tag, take first line
        answer = process_answer(response_text.split("\n")[0])
        return answer, None, 0.0

    answer = process_answer(answer_match.group(1).strip())

    # Extract rationale (optional, only if CoT)
    rationale = None
    if chain_of_thoughts:
        rationale_match = re.search(
            r"rationale:\s*(.+?)(?=confidence:|$)", response_text, re.IGNORECASE | re.DOTALL
        )
        if rationale_match:
            rationale = rationale_match.group(1).strip()

    # Extract confidence (optional)
    confidence = 0.0
    confidence_match = re.search(r"confidence:\s*(0\.\d+)", response_text, re.IGNORECASE)
    if confidence_match:
        confidence = float(confidence_match.group(1))

    return answer, rationale, confidence


def extract_logprobs_until_stop(
    tokens: List[str], token_logprobs: List[float], stop_token: str = "."
) -> Tuple[List[float], str]:
    """
    Extract log probabilities until stop token.

    Args:
        tokens: List of generated tokens
        token_logprobs: List of log probabilities
        stop_token: Token to stop at

    Returns:
        Tuple of (extracted_logprobs, extracted_text)
    """
    plist = []
    extracted_tokens = []

    for i, token in enumerate(tokens):
        if token.endswith(stop_token):
            break
        if i < len(token_logprobs) and token_logprobs[i] is not None:
            plist.append(token_logprobs[i])
            extracted_tokens.append(token)

    text = "".join(extracted_tokens)
    return plist, text


def make_choices_text(choices: List[str], answer_idx: int) -> Tuple[str, str]:
    """
    Format multiple choice options and extract answer.

    Args:
        choices: List of choice strings
        answer_idx: Index of correct answer

    Returns:
        Tuple of (choices_text, answer_text)
    """
    choices_text = ", ".join(choices) + "."
    answer_text = choices[answer_idx]

    return choices_text, answer_text


def compute_vqa_score(predicted_answer: str, reference_answers: List[str]) -> float:
    """
    Compute VQA-style accuracy score.

    This is a rough estimator for quick results check.
    Use official VQA eval script for final numbers.

    Args:
        predicted_answer: Predicted answer
        reference_answers: List of reference answers

    Returns:
        Score between 0 and 1
    """
    counter = 0
    for ref_answer in reference_answers:
        if predicted_answer == ref_answer:
            counter += 1

    # VQA scoring: min(1, count/3)
    score = min(1.0, float(counter) * 0.3)

    return score


def filter_thoughts_by_similarity(
    thoughts: List[str], similarities: List[float], threshold: float = 0.0
) -> Tuple[str, str]:
    """
    Filter thoughts based on similarity scores.

    Args:
        thoughts: List of thought strings
        similarities: List of similarity scores
        threshold: Minimum similarity threshold

    Returns:
        Tuple of (filtered_thoughts, all_thoughts)
    """
    filtered = []
    all_thoughts = []

    for thought, sim in zip(thoughts, similarities):
        thought = thought.strip()
        if sim > threshold and len(thought) > 0:
            filtered.append(thought)
        if len(thought) > 0:
            all_thoughts.append(thought)

    filtered_text = ". ".join(filtered).strip() + "." if filtered else ""
    all_text = ". ".join(all_thoughts).strip() + "." if all_thoughts else ""

    return filtered_text, all_text
