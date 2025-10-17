"""
Integration test demonstrating full See -> Think pipeline
This mimics the flow from main_aokvqa.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import Mock


def test_full_visual_cot_pipeline():
    """
    Test complete Visual CoT pipeline from See to Think
    Mimics the flow from main_aokvqa.py
    """
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Full Visual CoT Pipeline")
    print("=" * 70)

    # =========================================================================
    # STEP 1: Simulate See Module Output (Scene Graph Detection)
    # =========================================================================
    print("\n[STEP 1] See Module - Scene Graph Detection")
    print("-" * 70)

    # This simulates output from SceneGraphDetector
    # Format: [confidence, object_name, attributes, caption, ocr_text]
    scene_graph_attrs = [
        [0.95, "person", ["standing", "wearing white"], "A person in white clothes", ""],
        [0.90, "tennis racket", ["green", "held"], "A green tennis racket being held", ""],
        [0.85, "tennis ball", ["yellow", "flying"], "A yellow tennis ball in mid-air", ""],
        [0.80, "tennis court", ["outdoor", "clay"], "An outdoor clay tennis court", ""],
    ]

    print(f"✓ Detected {len(scene_graph_attrs)} objects:")
    for obj in scene_graph_attrs:
        print(f"  - {obj[1]} (confidence: {obj[0]:.2f})")

    # Global image caption (from BLIP2 or pre-computed)
    global_caption = "A tennis player on an outdoor clay court."
    print(f"✓ Global caption: '{global_caption}'")

    # =========================================================================
    # STEP 2: Initialize Think Module Components
    # =========================================================================
    print("\n[STEP 2] Initialize Think Module")
    print("-" * 70)

    from vctp.think.llm import LLMResponse
    from vctp.think.reasoning import QuestionAnswerer, ThoughtVerifier
    from vctp.think.interactive import (
        InteractiveLoop,
        InteractiveAttention,
        LLMAttentionStrategy,
    )
    from vctp.think.prompts import FewShotExamplesManager
    from vctp.think.context import InteractiveContextManager

    # Mock LLM (in real scenario, would use create_llm_adapter)
    mock_llm = Mock()
    mock_llm.config = Mock()

    # Mock LLM responses for Q&A
    qa_responses = [
        LLMResponse(
            text="playing tennis. The person is holding a tennis racket on a tennis court.",
            logprobs=[-0.05, -0.1, -0.08],
            tokens=["playing", " tennis", "."],
            total_logprob=-0.23,
        )
    ]
    mock_llm.generate = Mock(side_effect=qa_responses * 10)  # Repeat for multiple rounds

    print("✓ LLM adapter initialized (mocked)")

    # Mock Context Manager
    mock_context = Mock()
    mock_context.train_questions = {
        "111<->222": "What sport is being played?",
        "333<->444": "What is the person holding?",
    }
    mock_context.get_qa_context_examples = Mock(return_value=["111<->222", "333<->444"])
    mock_context.get_examples_with_object_selection = Mock(
        return_value=(
            ["111<->222"],
            [{"tennis racket": 0.9, "tennis ball": 0.7}],
        )
    )

    print("✓ Context manager initialized (mocked)")

    # Mock Examples Manager
    train_questions = {
        "111<->222": "What sport is being played?",
        "333<->444": "What is the person holding?",
    }
    train_answers = {"111<->222": ["tennis"], "333<->444": ["tennis racket"]}
    train_rationales = {
        "111<->222": ["There is a tennis court and racket."],
        "333<->444": ["The person holds a tennis racket."],
    }

    examples_manager = FewShotExamplesManager(
        train_questions=train_questions,
        train_answers=train_answers,
        train_rationales=train_rationales,
    )

    print("✓ Examples manager initialized")

    # =========================================================================
    # STEP 3: Interactive Object Selection (Round 1)
    # =========================================================================
    print("\n[STEP 3] Interactive Round 1 - Object Selection")
    print("-" * 70)

    from vctp.think.interactive import RandomAttentionStrategy

    # Use random strategy for testing (in real scenario, use LLMAttentionStrategy)
    attention_strategy = RandomAttentionStrategy()

    # Select object for round 1
    selected_idx_r1 = attention_strategy.select_object(
        question="What is the person doing?", objects=scene_graph_attrs
    )

    selected_object_r1 = scene_graph_attrs[selected_idx_r1]
    print(f"✓ Selected object: {selected_object_r1[1]}")

    # =========================================================================
    # STEP 4: Question Answering (Round 1)
    # =========================================================================
    print("\n[STEP 4] Reasoning Round 1 - Question Answering")
    print("-" * 70)

    question_answerer = QuestionAnswerer(
        llm=mock_llm, engine="gpt3", chain_of_thoughts=True, n_ensemble=1, debug=False
    )

    # Format scene graph for selected object
    scene_graph_text_r1 = f"{selected_object_r1[1]} is {', '.join(selected_object_r1[2])}"

    # Get few-shot examples
    examples = examples_manager.format_examples_batch(
        example_keys=["111<->222", "333<->444"], include_rationale=True
    )

    # Answer question
    answer_r1, rationale_r1, confidence_r1 = question_answerer.answer(
        question="What is the person doing?",
        context=global_caption,
        scene_graph_text=scene_graph_text_r1,
        examples=examples,
    )

    print(f"✓ Answer: {answer_r1}")
    print(f"✓ Rationale: {rationale_r1}")
    print(f"✓ Confidence: {confidence_r1:.3f}")

    # =========================================================================
    # STEP 5: Thought Verification (Optional)
    # =========================================================================
    print("\n[STEP 5] Thought Verification")
    print("-" * 70)

    # Mock thought verifier
    thought_verifier = ThoughtVerifier(use_clip=False, debug=False)

    verified_rationale, all_rationale, scores = thought_verifier.verify_thoughts(
        thoughts=rationale_r1, image_embedding=None
    )

    print(f"✓ Verified rationale: {verified_rationale[:60]}...")

    # =========================================================================
    # STEP 6: Full Interactive Loop (Multiple Rounds)
    # =========================================================================
    print("\n[STEP 6] Full Interactive Loop (3 Rounds)")
    print("-" * 70)

    # Setup interactive attention
    interactive_attention = InteractiveAttention(
        attention_strategy=attention_strategy, max_rounds=3, stop_on_convergence=True, debug=True
    )

    # Setup interactive loop
    interactive_loop = InteractiveLoop(
        interactive_attention=interactive_attention,
        question_answerer=question_answerer,
        context_manager=mock_context,
        examples_manager=examples_manager,
        thought_verifier=thought_verifier,
        n_shot_qa=2,
        n_ensemble=1,
        chain_of_thoughts=True,
        debug=False,
    )

    # Run on sample
    result = interactive_loop.run_single_sample(
        query_key="123<->456",
        question="What is the person doing?",
        objects=scene_graph_attrs,
        global_caption=global_caption,
        reference_answer=["playing tennis", "tennis"],
    )

    # =========================================================================
    # STEP 7: Display Final Results
    # =========================================================================
    print("\n[STEP 7] Final Results")
    print("=" * 70)

    print(f"\n{'Key:':<20} {result['key']}")
    print(f"{'Question:':<20} {result['question']}")
    print(f"{'Answer:':<20} {result['answer']}")
    print(f"{'Rationale:':<20} {result['rationale'][:60]}...")
    print(f"{'Confidence:':<20} {result['confidence']:.3f}")
    print(f"{'Accuracy:':<20} {result['accuracy']:.2%}")
    print(f"{'Rounds:':<20} {result['rounds']}")

    print(f"\n{'Selected Objects:':}")
    for round_idx, obj_list in enumerate(result["selected_objects"], 1):
        print(f"  Round {round_idx}: {', '.join(obj_list)}")

    # =========================================================================
    # Assertions
    # =========================================================================
    assert result["answer"] == "playing tennis"
    assert result["rounds"] <= 3
    assert result["accuracy"] > 0
    assert len(result["selected_objects"]) > 0

    print("\n" + "=" * 70)
    print("✓ INTEGRATION TEST PASSED")
    print("=" * 70)


def test_ablation_modes():
    """Test different ablation modes"""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST: Ablation Modes")
    print("=" * 70)

    from vctp.think.interactive import (
        RandomAttentionStrategy,
        OracleAttentionStrategy,
        AllRegionsAttentionStrategy,
    )

    objects = [
        [0.95, "person", ["standing"]],
        [0.90, "tennis racket", ["green"]],
        [0.85, "ball", ["yellow"]],
    ]

    # Test 1: Random Attention
    print("\n[Test 1] Random Attention (Ablation)")
    random_strategy = RandomAttentionStrategy()
    idx = random_strategy.select_object(question="test", objects=objects)
    print(f"✓ Random selection: {objects[idx][1]}")

    # Test 2: Oracle Attention
    print("\n[Test 2] Oracle Attention (Upper Bound)")
    oracle_scores = {"key": {"person": 0.3, "tennis racket": 0.9, "ball": 0.7}}
    oracle_strategy = OracleAttentionStrategy(oracle_attend_dict=oracle_scores)
    idx_oracle = oracle_strategy.select_object(question="test", objects=objects, query_key="key")
    assert objects[idx_oracle][1] == "tennis racket"
    print(f"✓ Oracle selection: {objects[idx_oracle][1]} (highest score)")

    # Test 3: All Regions (No Iteration)
    print("\n[Test 3] All Regions (No Iteration)")
    all_strategy = AllRegionsAttentionStrategy()
    idx_all = all_strategy.select_object(question="test", objects=objects)
    assert idx_all == -1
    print(f"✓ All regions mode: use all {len(objects)} objects at once")

    print("\n✓ All ablation modes work correctly")


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNING INTEGRATION TESTS")
    print("=" * 70)

    test_full_visual_cot_pipeline()
    test_ablation_modes()

    print("\n" + "=" * 70)
    print("🎉 ALL INTEGRATION TESTS PASSED!")
    print("=" * 70)
    print("\nThe Think module is correctly integrated with See module outputs.")
    print("Visual CoT pipeline from main_aokvqa.py has been successfully refactored!")
