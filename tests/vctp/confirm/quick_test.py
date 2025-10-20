"""
Quick smoke test for Confirm module - minimal dependencies
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def quick_test():
    """Quick test that modules can be imported"""
    print("Quick Smoke Test - Checking Confirm Module Imports...")
    print(f"Project root: {project_root}")
    print("-" * 60)

    # Test main confirmer imports
    try:
        from vctp.confirm import (
            NoOpConfirmer,
            VisualConsistencyConfirmer,
            AnswerConsistencyConfirmer,
        )

        print("✓ Main confirmer modules imported")
    except Exception as e:
        print(f"✗ Confirmer import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test verifier imports
    try:
        from vctp.confirm.verifiers import (
            CLIPThoughtVerifier,
            BLIP2ThoughtVerifier,
            ChoiceAnswerVerifier,
        )

        print("✓ Verifier modules imported")
    except Exception as e:
        print(f"✗ Verifier import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Test scorer imports
    try:
        from vctp.confirm.scorers import clip_scorer, rule_based, blip2_scorer

        print("✓ Scorer modules imported")
    except Exception as e:
        print(f"✗ Scorer import failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Quick functionality test - NoOpConfirmer
    try:
        from vctp.confirm import NoOpConfirmer
        from vctp.core.types import ReasoningOutput, EvidenceBundle

        confirmer = NoOpConfirmer()

        # Mock data
        candidate = ReasoningOutput(
            candidate_answer="tennis",
            cot_rationale="The person is playing tennis.",
            used_concepts=["tennis", "racket", "court"],  # ← Field bắt buộc
        )
        # Line 77-85 trong quick_test.py
        evidence = EvidenceBundle(
            image_id="123",
            global_caption="A person on a tennis court",
            detected_objects=[],
            attributes={},
            relations=[],
            clip_image_embed=None,  # ← Đơn giản nhất
            region_captions=None,
        )

        result = confirmer.run(
            question="What sport is being played?",
            candidate=candidate,
            evidence=evidence,
        )

        assert result.is_confirmed == True
        print("✓ NoOpConfirmer works")
        print(f"  Result: confirmed={result.is_confirmed}, score={result.score}")
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("-" * 60)
    print("✓ All quick tests passed!")
    return True


if __name__ == "__main__":
    success = quick_test()
    exit(0 if success else 1)
