# Interactive Attention Module

This module implements the interactive attention mechanism for Visual Chain-of-Thought, enabling iterative refinement through multiple See-Think rounds.

## Overview

Visual CoT performs reasoning through **interactive attention rounds**:

```
Round 1: Question → See (select object) → Think (reason) → Partial answer
Round 2: Question + Previous thoughts → See (select new object) → Think → Better answer
Round 3: Question + All thoughts → See (select another) → Think → Final answer
```

## Components

### 1. AttentionStrategy (`attention_strategy.py`)

Different strategies for selecting which object to attend to.

**Available Strategies:**

#### LLMAttentionStrategy (Default)
Uses LLM with few-shot prompting to select most relevant object.

```python
from vctp.think.interactive import LLMAttentionStrategy
from vctp.think.reasoning import ObjectSelector
from vctp.think.context import InteractiveContextManager

# Setup
object_selector = ObjectSelector(llm=llm, engine="gpt3")
context_mgr = InteractiveContextManager(...)

strategy = LLMAttentionStrategy(
    object_selector=object_selector,
    context_manager=context_mgr,
    n_shot=8
)

# Select object
idx = strategy.select_object(
    question="What is the person doing?",
    objects=[[0.95, "person", ...], [0.90, "racket", ...]],
    query_key="123<->456"
)
```

#### RandomAttentionStrategy (Ablation)
Randomly selects objects.

```python
from vctp.think.interactive import RandomAttentionStrategy

strategy = RandomAttentionStrategy()
idx = strategy.select_object(question="...", objects=[...])
```

#### OracleAttentionStrategy (Ablation)
Uses ground-truth object relevance scores.

```python
from vctp.think.interactive import OracleAttentionStrategy

oracle_scores = {
    "123<->456": {"person": 0.9, "racket": 0.7, "ball": 0.5}
}

strategy = OracleAttentionStrategy(oracle_attend_dict=oracle_scores)
idx = strategy.select_object(
    question="...",
    objects=[...],
    query_key="123<->456"
)
```

#### AllRegionsAttentionStrategy
Attends to all regions at once (no iterative selection).

```python
from vctp.think.interactive import AllRegionsAttentionStrategy

strategy = AllRegionsAttentionStrategy()
# Returns -1 to indicate all regions should be used
```

### 2. InteractiveAttention (`interactive_attention.py`)

Orchestrates multiple attention rounds.

**Features:**
- Iterative object selection
- Thought accumulation across rounds
- Convergence detection
- BLIP2 integration for local captions
- All-regions mode support

**Usage:**

```python
from vctp.think.interactive import InteractiveAttention, LLMAttentionStrategy

# Setup strategy
strategy = LLMAttentionStrategy(...)

# Create interactive attention
interactive = InteractiveAttention(
    attention_strategy=strategy,
    max_rounds=3,
    stop_on_convergence=True,
    use_blip2=False,
    debug=False
)

# Define reasoning callback
def reasoning_callback(selected_objects, accumulated_thoughts):
    # Your reasoning logic here
    answer, rationale, confidence = question_answerer.answer(...)
    return {
        "answer": answer,
        "rationale": rationale,
        "confidence": confidence
    }

# Run rounds
round_results, accumulated_thoughts = interactive.run_rounds(
    question="What is the person doing?",
    objects=[[0.95, "person", ...], ...],
    reasoning_callback=reasoning_callback,
    query_key="123<->456"
)

# Get summary
summary = interactive.get_round_summary(round_results, accumulated_thoughts)
print(f"Final Answer: {summary['answer']}")
print(f"Rounds: {summary['rounds']}")
```

**Round Results Format:**
```python
round_results = [
    {
        "answer": "playing tennis",
        "rationale": "The person holds a racket.",
        "confidence": 0.85,
        "selected_objects": ["tennis racket"]
    },
    {
        "answer": "playing tennis",
        "rationale": "There is a yellow ball.",
        "confidence": 0.92,
        "selected_objects": ["tennis racket", "ball"]
    },
    # ... more rounds
]
```

### 3. InteractiveLoop (`interactive_loop.py`)

Full end-to-end pipeline combining See, Select, and Think.

**Features:**
- Complete Visual CoT pipeline
- Context retrieval integration
- Example management
- Thought verification
- Batch processing
- Automatic saving
- Accuracy computation

**Single Sample Usage:**

```python
from vctp.think.interactive import InteractiveLoop

# Setup components
interactive_loop = InteractiveLoop(
    interactive_attention=interactive,
    question_answerer=question_answerer,
    context_manager=context_mgr,
    examples_manager=examples_mgr,
    thought_verifier=thought_verifier,
    n_shot_qa=16,
    n_ensemble=5,
    chain_of_thoughts=True,
    debug=False
)

# Run single sample
result = interactive_loop.run_single_sample(
    query_key="123<->456",
    question="What is the person doing?",
    objects=[[0.95, "person", ["standing"]], ...],
    global_caption="A person on a tennis court.",
    choices=None,  # or ["playing", "running", ...]
    reference_answer=["playing tennis", "tennis"],
    image_embedding=clip_image_emb,
    image_path="path/to/image.jpg"
)

print(f"Answer: {result['answer']}")
print(f"Rationale: {result['rationale']}")
print(f"Accuracy: {result['accuracy']}")
print(f"Rounds: {result['rounds']}")
```

**Batch Processing:**

```python
# Prepare samples
samples = [
    {
        "query_key": "123<->456",
        "question": "What is the person doing?",
        "objects": [[0.95, "person", ...], ...],
        "global_caption": "A person on court.",
        "reference_answer": ["playing tennis"],
        "image_embedding": img_emb_1,
    },
    {
        "query_key": "789<->012",
        "question": "What color is the ball?",
        "objects": [[0.90, "ball", ...], ...],
        "global_caption": "A tennis ball.",
        "reference_answer": ["yellow"],
        "image_embedding": img_emb_2,
    },
    # ... more samples
]

# Run batch
results = interactive_loop.run_batch(
    samples=samples,
    save_path="outputs/results.json",
    save_every=10  # Save every 10 samples
)

# Compute overall accuracy
overall_acc = sum(r["accuracy"] for r in results) / len(results)
print(f"Overall Accuracy: {overall_acc * 100:.2f}%")
```

## Complete Example

### Full Visual CoT Pipeline

```python
from vctp.think import (
    create_llm_adapter,
    QuestionAnswerer,
    ObjectSelector,
    ThoughtVerifier,
    FewShotExamplesManager,
)
from vctp.think.interactive import (
    InteractiveLoop,
    InteractiveAttention,
    LLMAttentionStrategy,
)
from vctp.think.context import InteractiveContextManager

# 1. Setup LLM
llm = create_llm_adapter(
    engine="gpt3",
    api_keys=["sk-..."],
    temperature=0.0
)

# 2. Setup Context Manager
context_mgr = InteractiveContextManager(
    similarity_path="data/coco_clip_new",
    sg_dir="data/scene_graphs",
    sg_attr_dir="data/scene_graphs_attr",
    train_questions=train_q_dict,
    train_answers=train_a_dict,
    train_rationales=train_r_dict,
)
context_mgr.load_features(metric="imagequestion")

# 3. Setup Examples Manager
examples_mgr = FewShotExamplesManager(
    train_questions=train_q_dict,
    train_answers=train_a_dict,
    train_rationales=train_r_dict,
    train_choices=train_c_dict,
)

# 4. Setup Components
object_selector = ObjectSelector(llm=llm, engine="gpt3")
question_answerer = QuestionAnswerer(
    llm=llm,
    chain_of_thoughts=True,
    n_ensemble=5
)
thought_verifier = ThoughtVerifier(use_clip=True, threshold=0.0)

# 5. Setup Attention Strategy
attention_strategy = LLMAttentionStrategy(
    object_selector=object_selector,
    context_manager=context_mgr,
    n_shot=8
)

# 6. Setup Interactive Attention
interactive_attention = InteractiveAttention(
    attention_strategy=attention_strategy,
    max_rounds=3,
    stop_on_convergence=True,
    debug=True
)

# 7. Setup Interactive Loop
interactive_loop = InteractiveLoop(
    interactive_attention=interactive_attention,
    question_answerer=question_answerer,
    context_manager=context_mgr,
    examples_manager=examples_mgr,
    thought_verifier=thought_verifier,
    n_shot_qa=16,
    n_ensemble=5,
    chain_of_thoughts=True,
    debug=True
)

# 8. Run on samples
result = interactive_loop.run_single_sample(
    query_key="123<->456",
    question="What is the person doing?",
    objects=[[0.95, "person", ["standing"]], [0.90, "tennis racket", ["green"]]],
    global_caption="A person on a tennis court.",
    reference_answer=["playing tennis", "tennis"]
)

print(f"Final Answer: {result['answer']}")
print(f"Rationale: {result['rationale']}")
print(f"Accuracy: {result['accuracy'] * 100:.1f}%")
print(f"Rounds: {result['rounds']}")
```

## Ablation Studies

### 1. Random Object Selection

```python
from vctp.think.interactive import RandomAttentionStrategy

strategy = RandomAttentionStrategy()
interactive = InteractiveAttention(
    attention_strategy=strategy,
    max_rounds=3
)
```

### 2. Oracle Object Selection

```python
from vctp.think.interactive import OracleAttentionStrategy

strategy = OracleAttentionStrategy(oracle_attend_dict=oracle_scores)
interactive = InteractiveAttention(
    attention_strategy=strategy,
    max_rounds=3
)
```

### 3. All Regions (No Iteration)

```python
from vctp.think.interactive import AllRegionsAttentionStrategy

strategy = AllRegionsAttentionStrategy()
interactive = InteractiveAttention(
    attention_strategy=strategy,
    max_rounds=1  # Only 1 round needed
)
```

### 4. Remove Visual Evidence

```python
interactive_loop = InteractiveLoop(
    ...,
    ablation_visual=True  # Scene graph will be empty
)
```

### 5. Remove Reasoning

```python
interactive_loop = InteractiveLoop(
    ...,
    ablation_reason=True  # No accumulated thoughts
)
```

### 6. Single Round (No Iteration)

```python
from vctp.think.interactive import SingleRoundAttention

interactive = SingleRoundAttention(
    attention_strategy=strategy,
    debug=False
)
```

## Performance Tips

1. **Number of Rounds**: Balance accuracy vs. cost
   - 1 round: Fast baseline
   - 3 rounds: Default, good performance
   - 5+ rounds: Diminishing returns

2. **Convergence Detection**: Enable to save costs
   ```python
   interactive = InteractiveAttention(
       ...,
       stop_on_convergence=True  # Stop if answer doesn't change
   )
   ```

3. **Batch Processing**: Process multiple samples efficiently
   ```python
   results = interactive_loop.run_batch(
       samples=samples,
       save_every=10  # Save intermediate results
   )
   ```

4. **Caching**: Cache few-shot examples
   ```python
   # Load features once
   context_mgr.load_features()
   
   # Reuse for all queries
   for sample in samples:
       result = interactive_loop.run_single_sample(**sample)
   ```

## Output Format

### Single Sample Result

```python
{
    "key": "123<->456",
    "question": "What is the person doing?",
    "answer": "playing tennis",
    "rationale": "There is a yellow ball.",
    "all_thoughts": [
        "The person holds a racket.",
        "There is a yellow ball."
    ],
    "confidence": 0.92,
    "accuracy": 1.0,
    "rounds": 2,
    "round_results": [
        {"answer": "playing tennis", "rationale": "...", ...},
        {"answer": "playing tennis", "rationale": "...", ...}
    ],
    "global_caption": "A person on a tennis court.",
    "selected_objects": [
        ["tennis racket"],
        ["tennis racket", "ball"]
    ]
}
```

### Batch Results

Saved to two files:
1. `results.json`: Full results with all metadata
2. `results_format.json`: Format for evaluation scripts

```json
[
    {
        "question_id": "456",
        "answer": "playing tennis",
        "rationale": ["The person holds a racket.", "There is a yellow ball."]
    }
]
```

## Integration with See Module

```python
from vctp.see import SceneGraphBuilder, BLIP2Captioner

# Setup perception
sg_builder = SceneGraphBuilder(...)
blip2 = BLIP2Captioner(...)

# Get objects from scene graph
objects = sg_builder.extract_objects_from_file("path/to/sg.json")

# Get global caption
global_caption = blip2.query_global_caption(
    image_path="path/to/image.jpg",
    question=question
)

# Run interactive loop
result = interactive_loop.run_single_sample(
    query_key=key,
    question=question,
    objects=objects,
    global_caption=global_caption,
    image_path="path/to/image.jpg"
)
```

