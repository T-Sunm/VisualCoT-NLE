# Confirmation Module Tests

Mock tests for the `confirm` module, which verifies reasoning outputs against visual evidence.

## Test Structure
`
tests/vctp/confirm/
├── quick_test.py # Smoke test for imports
├── test_confirmation.py # Component tests with mocks
├── run_all_tests.py # Run all tests
└── README.md # This file

`

## Running Tests

### All Tests
```bash
cd tests/vctp/confirm
python run_all_tests.py
```

### Quick Smoke Test
```bash
python quick_test.py
```

### Individual Component Tests
```bash
python test_confirmation.py
```

## Test Coverage

### 1. Quick Smoke Test (`quick_test.py`)
- ✓ Import main confirmer classes
- ✓ Import verifier classes
- ✓ Import scorer modules
- ✓ Basic NoOpConfirmer functionality

### 2. Confirmation Component Tests (`test_confirmation.py`)

#### Visual Consistency Tests
- `test_visual_consistency_clip_method()` - CLIP-based verification (lines 1096-1116)
- `test_visual_consistency_oracle_method()` - Oracle rationale (lines 1123-1126)
- `test_visual_consistency_random_method()` - Random rationale (lines 1117-1122)

#### Answer Consistency Tests
- `test_answer_consistency_confirmer()` - Direct choice matching (lines 1149-1159)
- `test_answer_consistency_with_correction()` - CLIP similarity correction

#### Integration Tests
- `test_integration_with_think_module()` - Process ReasoningOutput from Think
- `test_no_rationale_case()` - Handle empty rationales
- `test_no_choices_case()` - Handle missing choices
- `test_confirmer_comparison()` - Compare different strategies

## Mock Strategy

These tests use **mocks** to avoid external dependencies:

### What's Mocked
- ✓ CLIP models (transformers)
- ✓ BLIP2 models (LAVIS)
- ✓ Large model weights
- ✓ API calls

### What's Real
- ✓ Confirmer interfaces and logic
- ✓ Data structures (ReasoningOutput, EvidenceBundle, ConfirmationOutput)
- ✓ Strategy patterns
- ✓ Integration points

## Test Data

Tests use synthetic data matching the structure from `main_aokvqa.py`:

```python
# Reasoning output from Think module
candidate = ReasoningOutput(
    candidate_answer="playing tennis",
    cot_rationale="The person is holding a racket. There is a yellow ball.",
    confidence=0.85
)

# Evidence from See module
evidence = EvidenceBundle(
    image_id="123",
    global_caption="A person on a tennis court",
    detected_objects=[...],
    attributes={...},
    relations=[...],
    clip_image_embed=np.array([...])
)
```

## Mapping to main_aokvqa.py

| Test | Lines in main_aokvqa.py | Confirmer | Method |
|------|------------------------|-----------|--------|
| `test_visual_consistency_clip_method` | 1096-1116 | `VisualConsistencyConfirmer` | `"clip"` |
| `test_visual_consistency_oracle_method` | 1123-1126 | `VisualConsistencyConfirmer` | `"oracle"` |
| `test_visual_consistency_random_method` | 1117-1122 | `VisualConsistencyConfirmer` | `"random"` |
| `test_answer_consistency_confirmer` | 1149-1159 | `AnswerConsistencyConfirmer` | - |

## Expected Output
python test_confirmation.py