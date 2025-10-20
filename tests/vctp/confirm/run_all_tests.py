"""
Run all confirmation module tests
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import traceback


def run_all_tests():
    """Run all confirmation tests"""
    print("\n" + "=" * 70)
    print("RUNNING ALL CONFIRMATION MODULE TESTS")
    print("=" * 70)

    all_passed = True

    # 1. Quick smoke test
    print("\n[1/2] Running quick smoke test...")
    print("-" * 70)
    try:
        from quick_test import quick_test

        if not quick_test():
            all_passed = False
            print("❌ Quick test FAILED")
        else:
            print("✅ Quick test PASSED")
    except Exception as e:
        print(f"❌ Quick test FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # 2. Confirmation component tests
    print("\n[2/2] Running confirmation component tests...")
    print("-" * 70)
    try:
        from test_confirmation import (
            test_visual_consistency_clip_method,
            test_visual_consistency_oracle_method,
            test_visual_consistency_random_method,
            test_answer_consistency_confirmer,
            test_answer_consistency_with_correction,
            test_integration_with_think_module,
            test_no_rationale_case,
            test_no_choices_case,
            test_confirmer_comparison,
        )

        test_visual_consistency_clip_method()
        test_visual_consistency_oracle_method()
        test_visual_consistency_random_method()
        test_answer_consistency_confirmer()
        test_answer_consistency_with_correction()
        test_integration_with_think_module()
        test_no_rationale_case()
        test_no_choices_case()
        test_confirmer_comparison()

        print("✅ Confirmation tests PASSED")
    except Exception as e:
        print(f"❌ Confirmation tests FAILED with exception: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        return 0
    else:
        print("❌ SOME TESTS FAILED")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
