

# Mock Tests for THINK Module

This directory contains mock tests for the `vctp.think` module and its sub-modules. These tests verify the refactored logic without requiring actual LLM API keys, models, or large datasets.

## Overview

The Think module implements the reasoning components of Visual Chain-of-Thought, including:
- **LLM Adapters**: OpenAI GPT-3, ChatGPT, OPT, LLaMA
- **Prompt Engineering**: Building few-shot prompts for object selection and QA
- **Context Retrieval**: CLIP-based similarity search for examples
- **Reasoning Components**: Object selection, question answering, thought verification
- **Interactive Attention**: Multi-round iterative refinement

## Test Structure

```
tests/vctp/think/
├── quick_test.py          # Quick smoke test - check imports
├── test_prompts.py        # Test prompt engineering components
├── test_reasoning.py      # Test reasoning components (mocked LLMs)
├── test_interactive.py    # Test interactive attention mechanism
├── test_integration.py    # Full See->Think pipeline integration test
├── run_all_tests.py       # Run all test suites
└── README.md             # This file
```

## How to Run Tests

### 1. Quick Smoke Test

Checks if all modules can be imported correctly.

```bash
# From this directory
python quick_test.py

# From project root
python -m tests.vctp.think.quick_test
```

### 2. Individual Test Suites

Run tests for a specific component.

```bash
# Prompt engineering
python test_prompts.py

# Reasoning components
python test_reasoning.py

# Interactive attention
python test_interactive.py

# Full integration
python test_integration.py
```

### 3. Run All Tests

Executes all test suites and provides a summary.

```bash
# From this directory
python run_all_tests.py

# From project root
python -m tests.vctp.think.run_all_tests
```

## Test Coverage

### Prompt Engineering (`test_prompts.py`)

✅ **ObjectSelectionPromptBuilder**
- GPT-3 and Chat prompt formats
- Few-shot examples integration
- Object formatting with attributes and captions

✅ **QuestionAnsweringPromptBuilder**
- Chain-of-thought prompts
- Multiple choice format
- Context and scene graph integration

✅ **FewShotExamplesManager**
- Random example selection
- Example formatting
- Batch processing

✅ **Answer Formatters**
- Answer processing and cleaning
- Extract answer and rationale from LLM responses
- VQA-style accuracy computation

✅ **BLIP2PromptBuilder**
- Global and local caption prompts
- Thought verification prompts
- Follow-up question generation

### Reasoning Components (`test_reasoning.py`)

✅ **ObjectSelector**
- Interface with mocked LLM
- Prompt building for object selection
- Multiple engine support (GPT-3, Chat, OPT)

✅ **RandomObjectSelector**
- Random selection for ablation studies
- Integration with See module outputs

✅ **OracleObjectSelector**
- Oracle-based selection using ground-truth scores
- Score-based object ranking

✅ **QuestionAnswerer**
- Interface with mocked LLM
- Chain-of-thought reasoning
- Ensemble strategies (max_logprob, majority_vote, weighted_vote)

✅ **ThoughtVerifier**
- CLIP-based verification (mocked)
- BLIP2-based verification (mocked)
- Thought filtering by similarity

✅ **Integration with See Module**
- Processing scene graph attributes
- Formatting detected objects
- Scene graph text generation

### Interactive Attention (`test_interactive.py`)

✅ **AttentionStrategy**
- RandomAttentionStrategy
- OracleAttentionStrategy
- AllRegionsAttentionStrategy

✅ **InteractiveAttention**
- Multi-round execution
- Convergence detection
- Thought accumulation
- Round summary generation

✅ **InteractiveLoop**
- Single sample processing
- Full See → Think pipeline
- Accuracy computation
- Integration with all components

✅ **LLMAttentionStrategy**
- Interface testing with mocked components
- Context manager integration
- Few-shot example retrieval

✅ **End-to-End Pipeline**
- See module output → Object selection → Reasoning
- Complete Visual CoT flow simulation

### Integration Tests (`test_integration.py`)

✅ **Full Visual CoT Pipeline**
- Step-by-step simulation of main_aokvqa.py flow
- See module scene graph → Interactive selection → Q&A → Verification
- Multi-round iterative refinement
- Complete result formatting

✅ **Ablation Modes**
- Random attention strategy
- Oracle attention strategy (upper bound)
- All-regions mode (no iteration)

✅ **See-Think Integration**
- Processing scene graph attributes from See module
- Object selection from detected objects
- Scene graph text formatting
- Question answering with visual evidence
- Final result aggregation

## Mock Data Format

### Scene Graph Attributes (from See module)

```python
scene_graph_attrs = [
    [confidence, object_name, [attributes], caption, ocr_text],
    [0.95, "person", ["standing", "wearing hat"], "A person standing", ""],
    [0.90, "tennis racket", ["green"], "A green tennis racket", ""],
    [0.85, "ball", ["yellow"], "A yellow ball", ""],
]
```

### Oracle Scores (for ablation)

```python
oracle_scores = {
    "image_id<->question_id": {
        "object_name": relevance_score,
        "person": 0.3,
        "tennis racket": 0.9,
        "ball": 0.7,
    }
}
```

### Few-Shot Examples

```python
examples = [
    {
        "context": "Global image caption",
        "question": "What is the person doing?",
        "answer": "playing tennis",
        "rationale": "The person holds a tennis racket.",
    }
]
```

## Mocking Strategy

### LLM Responses

Tests use `unittest.mock.Mock` to simulate LLM responses:

```python
from unittest.mock import Mock
from vctp.think.llm import LLMResponse

mock_llm = Mock()
mock_response = LLMResponse(
    text="tennis. The person is holding a racket.",
    logprobs=[-0.1, -0.2],
    tokens=["tennis", "."],
    total_logprob=-0.3
)
mock_llm.generate = Mock(return_value=mock_response)
```

### CLIP Models

CLIP models are mocked to avoid loading large weights:

```python
mock_clip_model = Mock()
mock_clip_processor = Mock()

verifier = ThoughtVerifier(
    use_clip=True,
    clip_model=mock_clip_model,
    clip_processor=mock_clip_processor
)
```

### Context Managers

Context managers are mocked for example retrieval:

```python
mock_context = Mock()
mock_context.get_examples_with_object_selection = Mock(
    return_value=(["example_key"], [{"object": 0.9}])
)
```

## What These Tests Verify

✅ **Module Structure**: All components can be imported correctly
✅ **Interface Compliance**: Components follow expected interfaces
✅ **Prompt Building**: Templates are correctly formatted
✅ **Data Flow**: See → Think pipeline works correctly
✅ **Integration**: Components work together seamlessly
✅ **Ablation Support**: Random, Oracle, and other ablation modes work
✅ **Multi-round Logic**: Interactive attention rounds execute properly

## What These Tests DON'T Verify

❌ **LLM API Calls**: Requires real API keys (use integration tests)
❌ **Model Inference**: Requires actual models (OPT, LLaMA, CLIP)
❌ **Accuracy**: Requires full dataset and evaluation
❌ **Performance**: Speed and efficiency benchmarks

## Integration with See Module

Tests simulate the flow from `main_aokvqa.py`:

```python
# 1. See module produces scene graph
scene_graph_attrs = [
    [0.95, "person", ["standing"], "A person", ""],
    [0.90, "tennis racket", ["green"], "A racket", ""],
]

# 2. Interactive object selection
strategy = LLMAttentionStrategy(...)
selected_idx = strategy.select_object(
    question="What is the person doing?",
    objects=scene_graph_attrs
)

# 3. Format scene graph for reasoning
scene_graph_text = format_scene_graph(scene_graph_attrs[selected_idx])

# 4. Question answering
answerer = QuestionAnswerer(llm=llm)
answer, rationale, confidence = answerer.answer(
    question="What is the person doing?",
    context="A person on court.",
    scene_graph_text=scene_graph_text
)
```

## Expected Output

When all tests pass:

```
======================================================================
VISUAL COT THINK MODULE - MOCK TEST SUITE
======================================================================

Running: Prompt Engineering
...
✓ Prompt Engineering PASSED

Running: Reasoning Components
...
✓ Reasoning Components PASSED

Running: Interactive Attention
...
✓ Interactive Attention PASSED

======================================================================
TEST SUMMARY
======================================================================
✓ PASS: prompts
✓ PASS: reasoning
✓ PASS: interactive
======================================================================
Results: 3/3 test suites passed
======================================================================

🎉 All tests passed! Think module refactoring is correct!
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're running from the correct directory:

```bash
# Add project root to PYTHONPATH
export PYTHONPATH="/path/to/VisualCoT:$PYTHONPATH"

# Or run as module from project root
cd /path/to/VisualCoT
python -m tests.vctp.think.run_all_tests
```

### Mock Failures

Some tests may show warnings about missing dependencies (transformers, tiktoken):
- This is expected for tokenizer-dependent tests
- These are gracefully skipped with informative messages
- Core functionality is still tested

### Test Incomplete

If a test module is incomplete, you may see:
```
✗ FAILED: Module 'X' not found
```

Ensure all test files are present in the directory.

## Adding New Tests

To add tests for new components:

1. Create a new test file: `test_<component>.py`
2. Follow the existing test structure
3. Use mocks for external dependencies
4. Update `run_all_tests.py` to include your test
5. Document in this README

Example test structure:

```python
def test_new_component():
    """Test description"""
    from vctp.think.module import NewComponent
    
    print("\n" + "=" * 60)
    print("Testing New Component")
    print("=" * 60)
    
    # Setup
    component = NewComponent(...)
    
    # Test
    result = component.do_something()
    
    # Assert
    assert result is not None
    print("✓ New component works")
```

## See Also

- `tests/vctp/see/README.md` - Tests for See (perception) module
- `src/vctp/think/README.md` - Think module documentation
- `VisualCoT/main_aokvqa.py` - Original implementation reference

