# Core Reasoning Components

This module contains the core reasoning components for Visual Chain-of-Thought.

## Components

### 1. ObjectSelector (`object_selector.py`)

Interactive object selection for the "See" phase.

**Features:**
- LLM-based object selection using few-shot prompts
- Support for multiple engines (GPT-3, ChatGPT, OPT, LLaMA)
- Logit bias for constrained decoding
- Random and oracle selectors for ablation studies

**Usage:**
```python
from vctp.think.reasoning import ObjectSelector
from vctp.think.llm import create_llm_adapter

# Create LLM and selector
llm = create_llm_adapter(engine="gpt3", api_keys=["sk-..."])
selector = ObjectSelector(llm=llm, engine="gpt3")

# Select object
objects = [
    [0.95, "person", ["standing"], "A person on court"],
    [0.90, "tennis racket", ["green"], "Tennis racket"],
    [0.85, "ball", ["flying"], "Yellow ball"],
]

examples = [
    {"question": "What sport?", "selected_object": "tennis racket"}
]

selected_idx = selector.select_object(
    question="What is the person doing?",
    objects=objects,
    examples=examples
)

print(f"Selected: {objects[selected_idx][1]}")
```

**Variants:**
```python
# Random selector (ablation)
from vctp.think.reasoning import RandomObjectSelector
selector = RandomObjectSelector()

# Oracle selector (ablation)
from vctp.think.reasoning import OracleObjectSelector
selector = OracleObjectSelector(oracle_attend_dict=oracle_scores)
```

### 2. QuestionAnswerer (`question_answerer.py`)

Question answering with chain-of-thought reasoning.

**Features:**
- Chain-of-thought reasoning
- Multiple choice support
- Ensemble methods (max logprob, majority vote, weighted vote)
- Iterative refinement with accumulated thoughts
- Support for all LLM engines

**Usage:**
```python
from vctp.think.reasoning import QuestionAnswerer

answerer = QuestionAnswerer(
    llm=llm,
    engine="gpt3",
    chain_of_thoughts=True,
    n_ensemble=5
)

# Answer question
answer, rationale, confidence = answerer.answer(
    question="What is the person doing?",
    context="A person on a tennis court.",
    scene_graph_text="person is holding racket.\nball is flying.",
    choices=None,  # or ["playing", "running", ...]
    examples=[...],  # Few-shot examples
    thoughts=["The person holds a racket."]  # Previous thoughts
)

print(f"Answer: {answer}")
print(f"Rationale: {rationale}")
print(f"Confidence: {confidence}")
```

**Ensemble Strategies:**
```python
from vctp.think.reasoning import EnsembleQuestionAnswerer

# Max logprob (default)
answerer = EnsembleQuestionAnswerer(
    llm=llm,
    n_ensemble=5,
    ensemble_strategy="max_logprob"
)

# Majority vote
answerer = EnsembleQuestionAnswerer(
    llm=llm,
    n_ensemble=5,
    ensemble_strategy="majority_vote"
)

# Weighted vote by logprobs
answerer = EnsembleQuestionAnswerer(
    llm=llm,
    n_ensemble=5,
    ensemble_strategy="weighted_vote"
)
```

### 3. ThoughtVerifier (`thought_verifier.py`)

Verify thoughts/reasoning against image content.

**Features:**
- CLIP-based verification (similarity with image embedding)
- BLIP2-based verification (ask BLIP2 to verify)
- Threshold-based filtering
- Oracle and random verifiers for ablation

**Usage with CLIP:**
```python
from vctp.think.reasoning import ThoughtVerifier

verifier = ThoughtVerifier(
    use_clip=True,
    threshold=0.0  # Similarity threshold
)

# Verify thoughts
thoughts = "The person is playing tennis. The ball is yellow."
filtered, all_thoughts, scores = verifier.verify_thoughts(
    thoughts=thoughts,
    image_embedding=image_clip_embedding
)

print(f"Filtered: {filtered}")
print(f"Scores: {scores}")
```

**Usage with BLIP2:**
```python
from vctp.see.captions import BLIP2Captioner

blip2 = BLIP2Captioner(...)

verifier = ThoughtVerifier(
    use_blip2=True,
    blip2_captioner=blip2
)

filtered, all_thoughts, _ = verifier.verify_thoughts(
    thoughts=thoughts,
    image_path="path/to/image.jpg"
)
```

**Ablation Variants:**
```python
# Oracle verifier (use ground-truth rationales)
from vctp.think.reasoning import OracleThoughtVerifier
verifier = OracleThoughtVerifier(rationale_dict=train_rationales)

# Random verifier (use random rationales)
from vctp.think.reasoning import RandomThoughtVerifier
verifier = RandomThoughtVerifier(rationale_dict=train_rationales)
```

## Integration Example

### Full Visual CoT Pipeline

```python
from vctp.think.reasoning import ObjectSelector, QuestionAnswerer, ThoughtVerifier
from vctp.think.llm import create_llm_adapter
from vctp.think.context import InteractiveContextManager
from vctp.think.prompts import FewShotExamplesManager

# Setup
llm = create_llm_adapter(engine="gpt3", api_keys=["sk-..."])
context_mgr = InteractiveContextManager(...)
examples_mgr = FewShotExamplesManager(...)

# Initialize components
object_selector = ObjectSelector(llm=llm, engine="gpt3")
answerer = QuestionAnswerer(llm=llm, chain_of_thoughts=True, n_ensemble=5)
verifier = ThoughtVerifier(use_clip=True, threshold=0.0)

# Interactive rounds
question = "What is the person doing?"
objects = [...]  # From scene graph
accumulated_thoughts = []

for round_i in range(3):
    # 1. Select relevant object
    selection_examples = context_mgr.get_examples_with_object_selection(...)
    
    selected_idx = object_selector.select_object(
        question=question,
        objects=objects,
        examples=selection_examples
    )
    
    selected_object = objects[selected_idx]
    print(f"Round {round_i+1}: Looking at {selected_object[1]}")
    
    # 2. Answer question with selected evidence
    qa_examples = examples_mgr.format_examples_batch(...)
    
    answer, rationale, confidence = answerer.answer(
        question=question,
        context="Global caption",
        scene_graph_text=f"{selected_object[1]} is {', '.join(selected_object[2])}",
        examples=qa_examples,
        thoughts=accumulated_thoughts
    )
    
    print(f"Answer: {answer}")
    print(f"Rationale: {rationale}")
    
    # 3. Verify thoughts
    verified_thought, all_thought, scores = verifier.verify_thoughts(
        thoughts=rationale,
        image_embedding=clip_image_emb
    )
    
    # 4. Accumulate thoughts for next round
    if verified_thought:
        accumulated_thoughts.append(verified_thought)
    
    # 5. Remove selected object for next round
    objects = objects[:selected_idx] + objects[selected_idx+1:]
    
    if len(objects) == 0:
        break

print(f"Final Answer: {answer}")
print(f"All Thoughts: {accumulated_thoughts}")
```

### Using the Main Reasoner

```python
from vctp.think.reasoner import VisualCoTReasoner
from vctp.core.types import EvidenceBundle, DetectedObject

# Initialize reasoner
reasoner = VisualCoTReasoner(
    llm=llm,
    examples_manager=examples_mgr,
    chain_of_thoughts=True,
    n_ensemble=5,
    use_thought_verification=True,
    thought_verifier=verifier
)

# Create evidence bundle
evidence = EvidenceBundle(
    detected_objects=[
        DetectedObject(label="person", confidence=0.95, attributes=["standing"]),
        DetectedObject(label="tennis racket", confidence=0.90, attributes=["green"]),
    ],
    global_caption="A person on a tennis court",
)

# Run reasoning
result = reasoner.run(
    evidence=evidence,
    question="What is the person doing?",
    context_caption="A person on a tennis court",
    example_keys=["123<->456", "789<->012"],
    reference_answer=["playing tennis", "tennis"]
)

print(f"Answer: {result.candidate_answer}")
print(f"Rationale: {result.cot_rationale}")
print(f"Confidence: {result.confidence}")
print(f"Accuracy: {result.metadata['accuracy']}")
```

## Ablation Studies

The module supports various ablation studies:

### 1. Random Object Selection
```python
from vctp.think.reasoning import RandomObjectSelector
selector = RandomObjectSelector()
```

### 2. Oracle Object Selection
```python
from vctp.think.reasoning import OracleObjectSelector
selector = OracleObjectSelector(oracle_attend_dict=oracle_scores)
```

### 3. Remove Visual Evidence
```python
reasoner = VisualCoTReasoner(
    llm=llm,
    examples_manager=examples_mgr,
    ablation_visual=True  # Remove scene graph
)
```

### 4. Remove Reasoning
```python
reasoner = VisualCoTReasoner(
    llm=llm,
    examples_manager=examples_mgr,
    ablation_reason=True  # No accumulated thoughts
)
```

### 5. Oracle Rationales
```python
from vctp.think.reasoning import OracleThoughtVerifier
verifier = OracleThoughtVerifier(rationale_dict=ground_truth_rationales)
```

### 6. Random Rationales
```python
from vctp.think.reasoning import RandomThoughtVerifier
verifier = RandomThoughtVerifier(rationale_dict=train_rationales)
```

## Performance Tips

1. **Ensemble size**: Balance accuracy vs. cost
   - `n_ensemble=1`: Fast, less accurate
   - `n_ensemble=5`: Good trade-off
   - `n_ensemble=10`: Best accuracy, slow

2. **Thought verification**: Enable for better accuracy
   - CLIP verification is fast
   - BLIP2 verification is more accurate but slower

3. **Iterative rounds**: More rounds = better but slower
   - 1 round: Fast baseline
   - 3 rounds: Default, good performance
   - 5 rounds: Diminishing returns

4. **Caching**: Cache LLM responses for repeated queries

5. **Batch processing**: Process multiple questions in parallel

