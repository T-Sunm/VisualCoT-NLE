"""
Run all mock tests for the Think module
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[3]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

import traceback


def run_test_file(test_name, test_module_name):
    """Run a single test file"""
    print(f"\n{'=' * 70}")
    print(f"Running: {test_name}")
    print("=" * 70)

    try:
        # Import and run the test module
        test_module = __import__(test_module_name, fromlist=[""])

        # Get all test functions
        test_functions = [
            getattr(test_module, name)
            for name in dir(test_module)
            if name.startswith("test_") and callable(getattr(test_module, name))
        ]

        # Run each test function
        for test_func in test_functions:
            test_func()

        print(f"\n✓ {test_name} PASSED")
        return True
    except Exception as e:
        print(f"\n✗ {test_name} FAILED")
        print(f"Error: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("VISUAL COT THINK MODULE - MOCK TEST SUITE")
    print("=" * 70)

    results = {}

    # Test 1: Prompts
    results["prompts"] = run_test_file("Prompt Engineering", "test_prompts")

    # Test 2: Reasoning
    results["reasoning"] = run_test_file("Reasoning Components", "test_reasoning")

    # Test 3: Interactive
    results["interactive"] = run_test_file("Interactive Attention", "test_interactive")

    # Test 4: Integration
    results["integration"] = run_test_file("See-Think Integration", "test_integration")

    # Print summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} test suites passed")
    print("=" * 70)

    if passed == total:
        print("\n🎉 All tests passed! Think module refactoring is correct!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test suite(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
