"""
Mock tests for Prompt Engineering components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def test_object_selection_prompt_builder():
    """Test ObjectSelectionPromptBuilder"""
    from vctp.think.prompts import ObjectSelectionPromptBuilder

    print("\n" + "=" * 60)
    print("Testing Object Selection Prompt Builder")
    print("=" * 60)

    # Test GPT-3 style prompt
    builder_gpt3 = ObjectSelectionPromptBuilder(engine="gpt3")

    prompt = builder_gpt3.build(
        question="What is the person doing?",
        object_list=["person", "tennis racket", "ball", "court"],
        examples=[{"question": "What sport is played?", "selected_object": "tennis racket"}],
    )

    print("\nGPT-3 Prompt Sample:")
    print(prompt[:200] + "...")

    assert "person" in prompt
    assert "tennis racket" in prompt
    assert "What is the person doing?" in prompt
    print("✓ GPT-3 prompt builder works")

    # Test Chat style prompt
    builder_chat = ObjectSelectionPromptBuilder(engine="chat")

    prompt_chat = builder_chat.build(
        question="What is the person doing?",
        object_list=["person", "tennis racket"],
        examples=[],
    )

    assert "person" in prompt_chat
    print("✓ Chat prompt builder works")

    # Test with attributes
    builder_attr = ObjectSelectionPromptBuilder(engine="gpt3", use_attributes=True)

    formatted = builder_attr.format_object_options(
        objects=[[0.95, "person", ["standing", "wearing hat"]], [0.90, "dog", ["brown"]]],
        use_attributes=True,
    )

    assert "standing" in formatted[0]
    assert "brown" in formatted[1]
    print("✓ Object formatting with attributes works")


def test_question_answering_prompt_builder():
    """Test QuestionAnsweringPromptBuilder"""
    from vctp.think.prompts import QuestionAnsweringPromptBuilder

    print("\n" + "=" * 60)
    print("Testing Question Answering Prompt Builder")
    print("=" * 60)

    # Test with chain-of-thought
    builder_cot = QuestionAnsweringPromptBuilder(engine="gpt3", chain_of_thoughts=True)

    examples = [
        {
            "context": "A tennis player on court.",
            "question": "What sport is this?",
            "answer": "tennis",
            "rationale": "The player holds a racket.",
        }
    ]

    prompt = builder_cot.build(
        question="What is the person doing?",
        context="A person on a tennis court.",
        scene_graph_text="person is holding racket. ball is flying.",
        examples=examples,
    )

    print("\nChain-of-Thought Prompt Sample:")
    print(prompt[:300] + "...")

    assert "person is holding racket" in prompt
    assert "What is the person doing?" in prompt
    print("✓ CoT prompt builder works")

    # Test without CoT
    builder_no_cot = QuestionAnsweringPromptBuilder(engine="gpt3", chain_of_thoughts=False)

    prompt_no_cot = builder_no_cot.build(
        question="What color is the ball?",
        context="A yellow tennis ball.",
        scene_graph_text="ball is yellow",
        examples=[],
    )

    assert "ball is yellow" in prompt_no_cot
    print("✓ Non-CoT prompt builder works")

    # Test with multiple choice
    builder_choice = QuestionAnsweringPromptBuilder(engine="gpt3", choice_only=True)

    prompt_choice = builder_choice.build(
        question="What is the person doing?",
        context="A person on court.",
        scene_graph_text="person holding racket",
        choices=["playing tennis", "running", "sitting"],
        examples=[],
    )

    assert "playing tennis" in prompt_choice
    assert "Choices" in prompt_choice
    print("✓ Multiple choice prompt builder works")


def test_few_shot_examples_manager():
    """Test FewShotExamplesManager"""
    from vctp.think.prompts import FewShotExamplesManager

    print("\n" + "=" * 60)
    print("Testing Few-Shot Examples Manager")
    print("=" * 60)

    # Setup mock data
    train_questions = {
        "123<->456": "What is the person doing?",
        "789<->012": "What color is the ball?",
        "345<->678": "What sport is this?",
    }

    train_answers = {
        "123<->456": ["playing tennis", "tennis"],
        "789<->012": ["yellow"],
        "345<->678": ["tennis"],
    }

    train_rationales = {
        "123<->456": ["The person holds a tennis racket."],
        "789<->012": ["The ball is yellow."],
        "345<->678": ["There is a tennis court."],
    }

    train_captions = {123: ["A person on a tennis court."], 789: ["A yellow ball."]}

    manager = FewShotExamplesManager(
        train_questions=train_questions,
        train_answers=train_answers,
        train_rationales=train_rationales,
        train_captions=train_captions,
    )

    # Test get random examples
    random_examples = manager.get_random_examples(n_shot=2)
    assert len(random_examples) == 2
    print(f"✓ Random example selection works: {len(random_examples)} examples")

    # Test format example
    example = manager.format_example("123<->456", include_rationale=True, include_choices=False)

    assert example["question"] == "What is the person doing?"
    assert example["answer"] == "playing tennis"
    assert "tennis racket" in example["rationale"]
    print("✓ Example formatting works")
    print(f"  Question: {example['question']}")
    print(f"  Answer: {example['answer']}")

    # Test format batch
    examples_batch = manager.format_examples_batch(
        example_keys=["123<->456", "789<->012"], include_rationale=True
    )

    assert len(examples_batch) == 2
    assert examples_batch[0]["question"] == "What is the person doing?"
    print(f"✓ Batch formatting works: {len(examples_batch)} examples")


def test_answer_formatters():
    """Test answer processing and extraction"""
    from vctp.think.prompts import (
        process_answer,
        extract_answer_and_rationale,
        compute_vqa_score,
    )

    print("\n" + "=" * 60)
    print("Testing Answer Formatters")
    print("=" * 60)

    # Test process_answer
    answer1 = process_answer("The person is playing tennis.")
    assert "person" in answer1
    assert "playing" in answer1
    assert "The" not in answer1  # Articles removed
    print(f"✓ Process answer: '{answer1}'")

    # Test extract_answer_and_rationale
    response_cot = "tennis. The person holds a racket on the court."
    answer, rationale = extract_answer_and_rationale(response_cot, chain_of_thoughts=True)

    assert answer == "tennis"
    assert "racket" in rationale
    print(f"✓ Extract CoT: answer='{answer}', rationale='{rationale[:30]}...'")

    # Test compute_vqa_score
    score1 = compute_vqa_score("tennis", ["tennis", "tennis", "playing tennis"])
    assert score1 == 0.6  # 2/3 * 0.3 = 0.6
    print(f"✓ VQA score computation: {score1}")

    score2 = compute_vqa_score("tennis", ["tennis", "tennis", "tennis", "tennis"])
    assert score2 == 1.0  # min(1.0, 4*0.3)
    print(f"✓ VQA score capped at 1.0: {score2}")


def test_blip2_prompt_builder():
    """Test BLIP2PromptBuilder"""
    from vctp.think.prompts import BLIP2PromptBuilder

    print("\n" + "=" * 60)
    print("Testing BLIP2 Prompt Builder")
    print("=" * 60)

    # Test global caption prompts
    general, question_prompt = BLIP2PromptBuilder.build_global_caption_prompt(
        "What is the person doing?"
    )

    assert "An image of" in general
    assert "What is the person doing?" in question_prompt
    print("✓ Global caption prompts built")

    # Test local caption prompt
    local_prompt = BLIP2PromptBuilder.build_local_caption_prompt(
        "tennis racket", "What sport is this?"
    )

    assert "tennis racket" in local_prompt
    print("✓ Local caption prompt built")

    # Test verify thought prompt
    verify_prompt = BLIP2PromptBuilder.build_verify_thought_prompt("the person is playing tennis")

    assert "playing tennis" in verify_prompt
    assert "yes or no" in verify_prompt
    print("✓ Thought verification prompt built")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Prompt Engineering Components")
    print("=" * 60)

    test_object_selection_prompt_builder()
    test_question_answering_prompt_builder()
    test_few_shot_examples_manager()
    test_answer_formatters()
    test_blip2_prompt_builder()

    print("\n" + "=" * 60)
    print("All Prompt tests passed! ✓")
    print("=" * 60)
