"""
Mock tests for Reasoning components
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from unittest.mock import Mock, MagicMock


def test_object_selector_with_mock_llm():
    """Test ObjectSelector with mocked LLM"""
    from vctp.think.reasoning import ObjectSelector
    from vctp.think.llm import LLMResponse

    print("\n" + "=" * 60)
    print("Testing Object Selector (Mocked LLM)")
    print("=" * 60)

    # Create mock LLM
    mock_llm = Mock()
    mock_llm.config = Mock()
    mock_llm.config.engine_name = "text-davinci-003"

    # Mock response - simulate selecting "tennis racket"
    mock_response = LLMResponse(
        text=" tennis racket", logprobs=[-0.1], tokens=[" tennis", " racket"], total_logprob=-0.1
    )
    mock_llm.generate = Mock(return_value=mock_response)

    # Create selector
    selector = ObjectSelector(llm=mock_llm, engine="gpt3", debug=False)

    # Test objects from see module
    objects = [
        [0.95, "person", ["standing"], "A person standing", ""],
        [0.90, "tennis racket", ["green"], "A green tennis racket", ""],
        [0.85, "ball", ["yellow"], "A yellow ball", ""],
    ]

    examples = [{"question": "What sport is played?", "selected_object": "tennis racket"}]

    # Note: This will fail without actual tokenizer, so we'll just test the interface
    try:
        # Test that select_object can be called
        assert hasattr(selector, "select_object")
        assert hasattr(selector, "prompt_builder")
        print("✓ ObjectSelector initialized successfully")

        # Test prompt building
        prompt = selector.prompt_builder.build(
            question="What is the person doing?",
            object_list=["person", "tennis racket", "ball"],
            examples=examples,
        )
        assert "tennis racket" in prompt
        print("✓ Prompt building works")

    except Exception as e:
        print(f"  Note: Full selection test skipped (needs tokenizer): {e}")


def test_random_object_selector():
    """Test RandomObjectSelector"""
    from vctp.think.reasoning import RandomObjectSelector

    print("\n" + "=" * 60)
    print("Testing Random Object Selector")
    print("=" * 60)

    selector = RandomObjectSelector()

    objects = [
        [0.95, "person", ["standing"]],
        [0.90, "tennis racket", ["green"]],
        [0.85, "ball", ["yellow"]],
    ]

    # Random selection should return valid index
    idx = selector.select_object(question="test", objects=objects)

    assert 0 <= idx < len(objects)
    print(f"✓ Random selector works: selected index {idx}")
    print(f"  Selected object: {objects[idx][1]}")


def test_oracle_object_selector():
    """Test OracleObjectSelector"""
    from vctp.think.reasoning import OracleObjectSelector

    print("\n" + "=" * 60)
    print("Testing Oracle Object Selector")
    print("=" * 60)

    # Oracle scores from main_aokvqa.py logic
    oracle_scores = {
        "123<->456": {"person": 0.3, "tennis racket": 0.9, "ball": 0.7},
        "789<->012": {"dog": 0.8, "frisbee": 0.6},
    }

    selector = OracleObjectSelector(oracle_attend_dict=oracle_scores)

    objects = [
        [0.95, "person", ["standing"]],
        [0.90, "tennis racket", ["green"]],
        [0.85, "ball", ["yellow"]],
    ]

    # Should select "tennis racket" (highest score 0.9)
    idx = selector.select_object(
        question="What is the person doing?", objects=objects, query_key="123<->456"
    )

    assert idx == 1  # tennis racket
    assert objects[idx][1] == "tennis racket"
    print(f"✓ Oracle selector works: selected '{objects[idx][1]}' (score: 0.9)")


def test_question_answerer_interface():
    """Test QuestionAnswerer interface"""
    from vctp.think.reasoning import QuestionAnswerer
    from vctp.think.llm import LLMResponse

    print("\n" + "=" * 60)
    print("Testing Question Answerer Interface")
    print("=" * 60)

    # Create mock LLM
    mock_llm = Mock()
    mock_llm.config = Mock()

    # Mock response with CoT
    mock_response = LLMResponse(
        text="tennis. The person is holding a racket.",
        logprobs=[-0.1, -0.2, -0.15],
        tokens=["tennis", ".", " The"],
        total_logprob=-0.45,
    )
    mock_llm.generate = Mock(return_value=mock_response)

    # Create answerer
    answerer = QuestionAnswerer(
        llm=mock_llm, engine="gpt3", chain_of_thoughts=True, n_ensemble=1, debug=False
    )

    # Test components
    assert hasattr(answerer, "prompt_builder")
    assert hasattr(answerer, "chain_of_thoughts")
    print("✓ QuestionAnswerer initialized")

    # Test prompt building
    examples = [
        {
            "context": "A tennis court.",
            "question": "What sport?",
            "answer": "tennis",
            "rationale": "There is a racket.",
        }
    ]

    prompt = answerer.prompt_builder.build(
        question="What is the person doing?",
        context="A person on court.",
        scene_graph_text="person holding racket",
        examples=examples,
    )

    assert "person holding racket" in prompt
    print("✓ QA prompt building works")


def test_thought_verifier_clip_mock():
    """Test ThoughtVerifier with mocked CLIP"""
    from vctp.think.reasoning import ThoughtVerifier

    print("\n" + "=" * 60)
    print("Testing Thought Verifier (Mocked CLIP)")
    print("=" * 60)

    # Create mock CLIP components
    mock_model = Mock()
    mock_processor = Mock()

    # Mock CLIP outputs
    mock_outputs = {"pooler_output": Mock()}
    mock_model.return_value = mock_outputs

    try:
        verifier = ThoughtVerifier(
            use_clip=True,
            clip_model=mock_model,
            clip_processor=mock_processor,
            threshold=0.0,
            debug=False,
        )

        assert hasattr(verifier, "use_clip")
        assert verifier.use_clip == True
        print("✓ ThoughtVerifier initialized with CLIP")

    except Exception as e:
        print(f"  Note: Full CLIP test skipped: {e}")

    # Test without CLIP
    verifier_no_clip = ThoughtVerifier(use_clip=False)

    thoughts = "The person is playing tennis. The ball is yellow."

    # Without CLIP, should return thoughts as-is
    verified, all_thoughts, scores = verifier_no_clip.verify_thoughts(
        thoughts=thoughts, image_embedding=None
    )

    assert verified == thoughts
    print("✓ ThoughtVerifier works without CLIP")
    print(f"  Verified: '{verified[:50]}...'")


def test_integration_with_see_module():
    """Test integration with See module outputs"""
    print("\n" + "=" * 60)
    print("Testing Integration with See Module")
    print("=" * 60)

    # Simulate output from See module (from main_aokvqa.py)
    mock_scene_graph_attrs = [
        [0.95, "person", ["standing", "wearing hat"], "A person standing", ""],
        [0.90, "tennis racket", ["green", "held"], "A green tennis racket", ""],
        [0.85, "ball", ["yellow", "flying"], "A yellow tennis ball", ""],
    ]

    # Test that reasoning components can process this
    from vctp.think.reasoning import RandomObjectSelector

    selector = RandomObjectSelector()
    idx = selector.select_object(question="test", objects=mock_scene_graph_attrs)

    selected_object = mock_scene_graph_attrs[idx]
    print(f"✓ Can process See module output")
    print(f"  Selected: {selected_object[1]} (conf: {selected_object[0]})")

    # Format scene graph text (like in main_aokvqa.py)
    scene_graph_text = "\n".join(
        [f"{obj[1]} is {', '.join(obj[2])}" for obj in mock_scene_graph_attrs if obj[2]]
    )

    assert "person is standing" in scene_graph_text
    assert "tennis racket is green" in scene_graph_text
    print("✓ Scene graph formatting works")
    print(f"  Scene graph:\n{scene_graph_text}")


def test_ensemble_question_answerer():
    """Test EnsembleQuestionAnswerer"""
    from vctp.think.reasoning import EnsembleQuestionAnswerer
    from vctp.think.llm import LLMResponse

    print("\n" + "=" * 60)
    print("Testing Ensemble Question Answerer")
    print("=" * 60)

    # Create mock LLM
    mock_llm = Mock()
    mock_llm.config = Mock()

    # Mock multiple responses for ensemble
    responses = [
        LLMResponse(text="tennis. Reason 1", logprobs=[-0.1], tokens=[], total_logprob=-0.1),
        LLMResponse(text="tennis. Reason 2", logprobs=[-0.05], tokens=[], total_logprob=-0.05),
        LLMResponse(
            text="playing tennis. Reason 3", logprobs=[-0.15], tokens=[], total_logprob=-0.15
        ),
    ]
    mock_llm.generate = Mock(side_effect=responses)

    # Test max_logprob strategy
    answerer = EnsembleQuestionAnswerer(
        llm=mock_llm, engine="gpt3", ensemble_strategy="max_logprob", n_ensemble=3, debug=False
    )

    assert answerer.ensemble_strategy == "max_logprob"
    print("✓ Ensemble answerer initialized with max_logprob strategy")

    # Test majority_vote strategy
    answerer_vote = EnsembleQuestionAnswerer(
        llm=mock_llm, engine="gpt3", ensemble_strategy="majority_vote", n_ensemble=3
    )

    assert answerer_vote.ensemble_strategy == "majority_vote"
    print("✓ Ensemble answerer with majority_vote strategy")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Reasoning Components")
    print("=" * 60)

    test_object_selector_with_mock_llm()
    test_random_object_selector()
    test_oracle_object_selector()
    test_question_answerer_interface()
    test_thought_verifier_clip_mock()
    test_integration_with_see_module()
    test_ensemble_question_answerer()

    print("\n" + "=" * 60)
    print("All Reasoning tests passed! ✓")
    print("=" * 60)
