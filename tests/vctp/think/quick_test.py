"""
Quick smoke test for Think module - minimal dependencies
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def quick_test():
    """Quick test that modules can be imported"""
    print("Quick Smoke Test - Checking Think Module Imports...")
    print(f"Project root: {project_root}")
    print("-" * 60)

    # Test LLM imports
    try:
        from vctp.think.llm import (
            BaseLLMAdapter,
            LLMResponse,
            LLMConfig,
            create_llm_adapter,
        )

        print("✓ LLM adapter modules imported")
    except Exception as e:
        print(f"✗ LLM import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test Prompts imports
    try:
        from vctp.think.prompts import (
            ObjectSelectionPromptBuilder,
            QuestionAnsweringPromptBuilder,
            FewShotExamplesManager,
            process_answer,
        )

        print("✓ Prompt engineering modules imported")
    except Exception as e:
        print(f"✗ Prompts import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test Context imports
    try:
        from vctp.think.context import (
            SimilarityRetriever,
            ContextManager,
            InteractiveContextManager,
        )

        print("✓ Context retrieval modules imported")
    except Exception as e:
        print(f"✗ Context import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test Reasoning imports
    try:
        from vctp.think.reasoning import (
            ObjectSelector,
            QuestionAnswerer,
            ThoughtVerifier,
        )

        print("✓ Reasoning components imported")
    except Exception as e:
        print(f"✗ Reasoning import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test Interactive imports
    try:
        from vctp.think.interactive import (
            LLMAttentionStrategy,
            InteractiveAttention,
            InteractiveLoop,
        )

        print("✓ Interactive attention modules imported")
    except Exception as e:
        print(f"✗ Interactive import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test Reasoner import
    try:
        from vctp.think.reasoner import VisualCoTReasoner

        print("✓ Main reasoner imported")
    except Exception as e:
        print(f"✗ Reasoner import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Quick functionality test - Prompt building
    try:
        from vctp.think.prompts import ObjectSelectionPromptBuilder

        builder = ObjectSelectionPromptBuilder(engine="gpt3")
        prompt = builder.build(
            question="What is in the image?",
            object_list=["person", "dog", "car"],
            examples=[{"question": "What animal?", "selected_object": "dog"}],
        )
        assert "person" in prompt
        assert "dog" in prompt
        print("✓ Basic prompt building works")
        print(f"  Sample prompt length: {len(prompt)} chars")
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test answer processing
    try:
        from vctp.think.prompts import process_answer

        answer = process_answer("The person is playing tennis.")
        assert "person" in answer
        print("✓ Answer processing works")
        print(f"  Processed: '{answer}'")
    except Exception as e:
        print(f"✗ Answer processing failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("-" * 60)
    print("✓ All quick tests passed!")
    return True


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
