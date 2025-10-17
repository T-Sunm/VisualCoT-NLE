"""
Mock tests for Interactive Attention components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import Mock


def test_attention_strategies():
    """Test different attention strategies"""
    from vctp.think.interactive import (
        RandomAttentionStrategy,
        OracleAttentionStrategy,
        AllRegionsAttentionStrategy,
    )

    print("\n" + "=" * 60)
    print("Testing Attention Strategies")
    print("=" * 60)

    # Mock objects from See module
    objects = [
        [0.95, "person", ["standing"]],
        [0.90, "tennis racket", ["green"]],
        [0.85, "ball", ["yellow"]],
    ]

    # Test Random strategy
    random_strategy = RandomAttentionStrategy()
    idx = random_strategy.select_object(question="What is happening?", objects=objects)

    assert 0 <= idx < len(objects)
    print(f"✓ Random strategy: selected index {idx} ({objects[idx][1]})")

    # Test Oracle strategy
    oracle_scores = {"123<->456": {"person": 0.3, "tennis racket": 0.9, "ball": 0.7}}

    oracle_strategy = OracleAttentionStrategy(oracle_attend_dict=oracle_scores)
    idx_oracle = oracle_strategy.select_object(
        question="What is happening?", objects=objects, query_key="123<->456"
    )

    assert idx_oracle == 1  # Should select tennis racket (highest score)
    print(f"✓ Oracle strategy: selected index {idx_oracle} ({objects[idx_oracle][1]})")

    # Test AllRegions strategy
    all_regions_strategy = AllRegionsAttentionStrategy()
    idx_all = all_regions_strategy.select_object(question="What is happening?", objects=objects)

    assert idx_all == -1  # Special value for all regions
    print(f"✓ AllRegions strategy: returned {idx_all} (use all regions)")


def test_interactive_attention():
    """Test InteractiveAttention multi-round execution"""
    from vctp.think.interactive import InteractiveAttention, RandomAttentionStrategy

    print("\n" + "=" * 60)
    print("Testing Interactive Attention")
    print("=" * 60)

    # Mock objects
    objects = [
        [0.95, "person", ["standing"], "A person standing", ""],
        [0.90, "tennis racket", ["green"], "A tennis racket", ""],
        [0.85, "ball", ["yellow"], "A yellow ball", ""],
    ]

    # Use random strategy for testing
    strategy = RandomAttentionStrategy()

    # Create interactive attention
    interactive = InteractiveAttention(
        attention_strategy=strategy, max_rounds=3, stop_on_convergence=False, debug=True
    )

    assert interactive.max_rounds == 3
    print("✓ InteractiveAttention initialized")

    # Mock reasoning callback
    round_count = [0]

    def mock_reasoning_callback(selected_objects, accumulated_thoughts):
        round_count[0] += 1
        return {
            "answer": "playing tennis",
            "rationale": f"Round {round_count[0]} reasoning.",
            "confidence": 0.8 + round_count[0] * 0.05,
            "selected_objects": [obj[1] for obj in selected_objects],
        }

    # Run rounds
    round_results, thoughts = interactive.run_rounds(
        question="What is the person doing?",
        objects=objects,
        reasoning_callback=mock_reasoning_callback,
        query_key="123<->456",
    )

    assert len(round_results) <= 3  # At most max_rounds
    assert len(round_results) > 0
    print(f"✓ Executed {len(round_results)} rounds")

    # Check round results structure
    for i, result in enumerate(round_results):
        assert "answer" in result
        assert "rationale" in result
        assert "confidence" in result
        print(f"  Round {i+1}: answer='{result['answer']}', conf={result['confidence']:.2f}")

    # Test get_round_summary
    summary = interactive.get_round_summary(round_results, thoughts)

    assert "answer" in summary
    assert "rounds" in summary
    assert summary["rounds"] == len(round_results)
    print(f"✓ Round summary: {summary['rounds']} rounds, answer='{summary['answer']}'")


def test_interactive_attention_convergence():
    """Test convergence detection"""
    from vctp.think.interactive import InteractiveAttention, RandomAttentionStrategy

    print("\n" + "=" * 60)
    print("Testing Convergence Detection")
    print("=" * 60)

    objects = [
        [0.95, "person", []],
        [0.90, "racket", []],
        [0.85, "ball", []],
    ]

    strategy = RandomAttentionStrategy()

    # Enable convergence detection
    interactive = InteractiveAttention(
        attention_strategy=strategy, max_rounds=5, stop_on_convergence=True, debug=False
    )

    # Callback that converges after round 2
    round_num = [0]

    def converging_callback(selected_objects, accumulated_thoughts):
        round_num[0] += 1
        # Same answer from round 2 onwards
        answer = "playing tennis" if round_num[0] >= 2 else "tennis"
        return {
            "answer": answer,
            "rationale": f"Round {round_num[0]}",
            "confidence": 0.8,
        }

    round_results, _ = interactive.run_rounds(
        question="Test", objects=objects, reasoning_callback=converging_callback
    )

    # Should stop early due to convergence
    assert len(round_results) < 5
    print(f"✓ Converged early at round {len(round_results)} (max_rounds=5)")


def test_interactive_loop_single_sample():
    """Test InteractiveLoop with single sample"""
    from vctp.think.interactive import (
        InteractiveLoop,
        InteractiveAttention,
        RandomAttentionStrategy,
    )
    from unittest.mock import Mock

    print("\n" + "=" * 60)
    print("Testing Interactive Loop - Single Sample")
    print("=" * 60)

    # Mock components
    strategy = RandomAttentionStrategy()
    interactive_attention = InteractiveAttention(
        attention_strategy=strategy, max_rounds=2, debug=False
    )

    # Mock question answerer
    mock_answerer = Mock()
    mock_answerer.answer = Mock(return_value=("playing tennis", "The person holds a racket.", 0.85))

    # Create loop (minimal setup)
    loop = InteractiveLoop(
        interactive_attention=interactive_attention,
        question_answerer=mock_answerer,
        context_manager=None,
        examples_manager=None,
        n_shot_qa=8,
        n_ensemble=1,
        chain_of_thoughts=True,
        debug=False,
    )

    # Mock sample from See module
    objects = [
        [0.95, "person", ["standing"], "A person standing", ""],
        [0.90, "tennis racket", ["green"], "A tennis racket", ""],
    ]

    # Run on single sample
    result = loop.run_single_sample(
        query_key="123<->456",
        question="What is the person doing?",
        objects=objects,
        global_caption="A person on a tennis court.",
        reference_answer=["playing tennis", "tennis"],
    )

    # Check result structure
    assert "key" in result
    assert "answer" in result
    assert "rationale" in result
    assert "accuracy" in result
    assert "rounds" in result

    assert result["key"] == "123<->456"
    assert result["answer"] == "playing tennis"
    assert result["rounds"] <= 2

    print(f"✓ Single sample processed")
    print(f"  Answer: {result['answer']}")
    print(f"  Rounds: {result['rounds']}")
    print(f"  Accuracy: {result['accuracy']}")


def test_llm_attention_strategy_interface():
    """Test LLMAttentionStrategy interface"""
    from vctp.think.interactive import LLMAttentionStrategy
    from unittest.mock import Mock

    print("\n" + "=" * 60)
    print("Testing LLM Attention Strategy Interface")
    print("=" * 60)

    # Mock object selector
    mock_selector = Mock()
    mock_selector.select_object = Mock(return_value=1)  # Select index 1

    # Mock context manager
    mock_context = Mock()
    mock_context.train_questions = {"789<->012": "What sport?"}
    mock_context.get_examples_with_object_selection = Mock(
        return_value=(
            ["789<->012"],
            [{"tennis racket": 0.9, "ball": 0.7}],
        )
    )

    # Create strategy
    strategy = LLMAttentionStrategy(
        object_selector=mock_selector, context_manager=mock_context, n_shot=8
    )

    objects = [
        [0.95, "person", []],
        [0.90, "tennis racket", []],
        [0.85, "ball", []],
    ]

    # Select object
    idx = strategy.select_object(
        question="What is happening?", objects=objects, query_key="123<->456"
    )

    assert idx == 1  # Mocked to return 1
    print(f"✓ LLM strategy selected index {idx}")

    # Verify context manager was called
    assert mock_context.get_examples_with_object_selection.called
    print("✓ Context manager integration works")


def test_integration_see_think_pipeline():
    """Test full See -> Think pipeline integration"""
    print("\n" + "=" * 60)
    print("Testing See -> Think Pipeline Integration")
    print("=" * 60)

    # Simulate complete flow from main_aokvqa.py

    # Step 1: See module output (scene graph attributes)
    scene_graph_attrs = [
        [0.95, "person", ["standing", "wearing white"], "A person in white clothes", ""],
        [0.90, "tennis racket", ["green", "held"], "A green tennis racket", ""],
        [0.85, "ball", ["yellow", "flying"], "A yellow ball in the air", ""],
    ]

    print("✓ See module output (scene graph):")
    for obj in scene_graph_attrs:
        print(f"  - {obj[1]} (conf: {obj[0]:.2f})")

    # Step 2: Interactive object selection
    from vctp.think.interactive import RandomAttentionStrategy

    strategy = RandomAttentionStrategy()
    selected_idx = strategy.select_object(
        question="What is the person doing?", objects=scene_graph_attrs
    )

    selected_object = scene_graph_attrs[selected_idx]
    print(f"✓ Object selected: {selected_object[1]}")

    # Step 3: Build scene graph text for reasoning
    noticed_objects = [selected_object]
    scene_graph_text = "\n".join(
        [f"{obj[1]} is {', '.join(obj[2])}" if obj[2] else obj[1] for obj in noticed_objects]
    )

    print(f"✓ Scene graph text:\n{scene_graph_text}")

    # Step 4: Mock question answering
    from vctp.think.prompts import QuestionAnsweringPromptBuilder

    builder = QuestionAnsweringPromptBuilder(engine="gpt3", chain_of_thoughts=True)

    prompt = builder.build(
        question="What is the person doing?",
        context="A person on a tennis court.",
        scene_graph_text=scene_graph_text,
        examples=[],
    )

    assert scene_graph_text in prompt
    assert "What is the person doing?" in prompt
    print("✓ QA prompt built successfully")

    print("✓ Full See -> Think pipeline validated")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Interactive Attention Components")
    print("=" * 60)

    test_attention_strategies()
    test_interactive_attention()
    test_interactive_attention_convergence()
    test_interactive_loop_single_sample()
    test_llm_attention_strategy_interface()
    test_integration_see_think_pipeline()

    print("\n" + "=" * 60)
    print("All Interactive Attention tests passed! ✓")
    print("=" * 60)
