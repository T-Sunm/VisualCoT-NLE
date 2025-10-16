# Mock Tests for SEE Module

This directory contains mock tests for the `vctp.see` module and its sub-modules. These tests verify the refactored logic without requiring actual models or large datasets.

## How to Run Tests

There are two primary ways to run the tests:

### 1. From within the test directory

Navigate to this directory from the project root and run the scripts directly.

```bash
# Navigate from project root (E:\AIO\Project\VisualCoT\)
cd tests/vctp/see
```

**Quick Smoke Test:**
Checks if all modules can be imported correctly.
```bash
python quick_test.py
```

**Individual Test Suites:**
Run tests for a specific component.
```bash
python test_scene_graph.py
python test_features.py
python test_perception.py
```

**Run All Tests:**
Executes all test suites and provides a summary.
```bash
python run_all_tests.py
```

### 2. From the Project Root

You can also run the tests as modules from the project's root directory. This is often a more robust way to handle Python's import system.

```bash
# Ensure you are in the project root (E:\AIO\Project\VisualCoT\)

# Quick test
python -m tests.vctp.see.quick_test

# All tests
python -m tests.vctp.see.run_all_tests
```
