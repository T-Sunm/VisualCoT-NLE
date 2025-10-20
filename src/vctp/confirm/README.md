# Confirmation Module

Xác minh đầu ra reasoning với visual evidence, dựa trên logic từ `VisualCoT/main_aokvqa.py`.

## Tổng quan

Module `confirm` cung cấp 2 chiến lược confirmation chính:

### 1. Visual Consistency (lines 1080-1136 trong main_aokvqa.py)
Xác minh thoughts/rationales với visual evidence.

### 2. Answer Consistency (lines 1149-1159 trong main_aokvqa.py)
Xác minh answer với available choices.

---

## Kiến trúc

```
confirm/
├── confirmer.py          # Main confirmation classes
│   ├── NoOpConfirmer
│   ├── VisualConsistencyConfirmer  # Strategy 1
│   └── AnswerConsistencyConfirmer  # Strategy 2
├── verifiers/            # Low-level verifiers
│   ├── thought_verifier.py
│   │   ├── CLIPThoughtVerifier
│   │   └── BLIP2ThoughtVerifier
│   └── answer_verifier.py
│       └── ChoiceAnswerVerifier
└── scorers/              # Scoring functions
    ├── clip_scorer.py
    ├── blip2_scorer.py
    └── rule_based.py
```

---

## Sử dụng

### 1. Visual Consistency Confirmation

Xác minh thoughts với visual evidence bằng nhiều phương pháp:

#### Method: CLIP (lines 1096-1116)
```python
from vctp.confirm import VisualConsistencyConfirmer

# Initialize confirmer
confirmer = VisualConsistencyConfirmer(
    method="clip",
    verify_threshold=0.3,  # Similarity threshold
    debug=True
)

# Run confirmation
result = confirmer.run(
    question=question,
    candidate=reasoning_output,
    evidence=evidence_bundle
)

print(f"Confirmed: {result.is_confirmed}")
print(f"Score: {result.score}")
print(f"Rationale: {result.rationale}")
```

#### Method: BLIP2 (lines 1081-1094)
```python
from vctp.see.captions import BLIP2Captioner
from vctp.confirm import VisualConsistencyConfirmer

# Initialize BLIP2 captioner
blip2_captioner = BLIP2Captioner(model_type="pretrain_flant5xxl")

# Initialize confirmer
confirmer = VisualConsistencyConfirmer(
    method="blip2",
    blip2_captioner=blip2_captioner,
    debug=True
)

# Run confirmation
result = confirmer.run(
    question=question,
    candidate=reasoning_output,
    evidence=evidence_bundle,
    image_path="/path/to/image.jpg"
)
```

#### Method: Oracle (lines 1123-1126, ablation)
```python
confirmer = VisualConsistencyConfirmer(
    method="oracle",
    rationale_dict=ground_truth_rationales,  # Dict[key, List[str]]
    debug=True
)

result = confirmer.run(
    question=question,
    candidate=reasoning_output,
    evidence=evidence_bundle,
    query_key="image_id<->question_id"  # Key to lookup ground-truth
)
```

#### Method: Random (lines 1117-1122, ablation)
```python
confirmer = VisualConsistencyConfirmer(
    method="random",
    rationale_dict=train_rationales,  # Dict[key, List[str]]
    debug=True
)

result = confirmer.run(
    question=question,
    candidate=reasoning_output,
    evidence=evidence_bundle
)
```

---

### 2. Answer Consistency Confirmation

Xác minh answer với available choices (lines 1149-1159):

```python
from vctp.confirm import AnswerConsistencyConfirmer

# Initialize confirmer
confirmer = AnswerConsistencyConfirmer(
    correct_answer=True,  # Auto-correct to closest choice
    debug=True
)

# Run confirmation
result = confirmer.run(
    question=question,
    candidate=reasoning_output,
    evidence=evidence_bundle,
    choices=["apple", "banana", "orange"]
)

print(f"Confirmed: {result.is_confirmed}")
print(f"Score: {result.score}")
print(f"Rationale: {result.rationale}")
```

**Logic:**
- Nếu predicted answer nằm trong choices → confirmed
- Nếu không → dùng CLIP text similarity tìm closest match
- Nếu `correct_answer=True` → auto-correct answer

---

## Chi tiết Implementation

### VisualConsistencyConfirmer

Hỗ trợ 4 phương pháp:

| Method | Description | Lines in main_aokvqa.py | Use Case |
|--------|-------------|------------------------|----------|
| `clip` | CLIP image-text similarity | 1096-1116 | Main verification |
| `blip2` | BLIP2 VQA | 1081-1094 | Advanced verification |
| `oracle` | Ground-truth rationale | 1123-1126 | Ablation study (upper bound) |
| `random` | Random rationale | 1117-1122 | Ablation study (baseline) |

**Parameters:**
- `method`: Verification method
- `verify_threshold`: CLIP similarity threshold (default: 0.0)
- `blip2_captioner`: BLIP2Captioner instance (for method="blip2")
- `rationale_dict`: Dict of rationales (for method="oracle" or "random")
- `device`: Device to run on
- `debug`: Enable debug output

**Returns:**
- `is_confirmed`: Whether thoughts are confirmed
- `score`: Average similarity/confidence score
- `rationale`: Description of verification result

---

### AnswerConsistencyConfirmer

Xác minh answer với choices bằng CLIP text similarity.

**Parameters:**
- `correct_answer`: Auto-correct to closest choice if not found (default: True)
- `device`: Device to run on
- `debug`: Enable debug output

**Returns:**
- `is_confirmed`: Whether answer is valid
- `score`: Similarity score with closest choice
- `rationale`: Description of verification result

---

## Integration với Pipeline

```python
from vctp.core.pipeline import VisualCoTPipeline
from vctp.confirm import VisualConsistencyConfirmer, AnswerConsistencyConfirmer

# Create confirmation modules
visual_confirmer = VisualConsistencyConfirmer(
    method="clip",
    verify_threshold=0.3
)

answer_confirmer = AnswerConsistencyConfirmer(
    correct_answer=True
)

# Add to pipeline
pipeline = VisualCoTPipeline(
    see_module=see_module,
    think_module=think_module,
    confirm_module=visual_confirmer  # Or answer_confirmer
)

# Run pipeline
result = pipeline.run(
    image_path="/path/to/image.jpg",
    question="What is the person doing?",
    choices=["eating", "sleeping", "running"]
)
```

---

## Mapping với main_aokvqa.py

| Code trong main_aokvqa.py | Class trong Module | Method |
|---------------------------|-------------------|--------|
| Lines 1096-1116 (CLIP verify) | `VisualConsistencyConfirmer` | `method="clip"` |
| Lines 1081-1094 (BLIP2 verify) | `VisualConsistencyConfirmer` | `method="blip2"` |
| Lines 1123-1126 (Oracle rationale) | `VisualConsistencyConfirmer` | `method="oracle"` |
| Lines 1117-1122 (Random rationale) | `VisualConsistencyConfirmer` | `method="random"` |
| Lines 1149-1159 (Choice verify) | `AnswerConsistencyConfirmer` | - |

---

## Testing

```bash
# Run all confirmation tests
cd tests/vctp/confirm
python run_all_tests.py

# Run specific test
python test_visual_consistency.py
python test_answer_consistency.py
```

---

## Notes

1. **Visual Consistency** là chiến lược chính để filter thoughts không consistent với image
2. **Answer Consistency** dùng để ensure answer nằm trong valid choices
3. Oracle và Random methods chỉ dùng cho ablation studies
4. CLIP method nhanh hơn BLIP2 nhưng BLIP2 accurate hơn
5. Cần pre-computed CLIP image embeddings cho performance tốt

---

## Future Work

- [ ] Thêm confidence calibration
- [ ] Support multi-modal fusion (CLIP + BLIP2)
- [ ] Adaptive threshold selection
- [ ] Cache verification results
