# Prompt Engineering Module

This module provides prompt templates, builders, and utilities for Visual Chain-of-Thought reasoning.

## Components

### 1. Templates (`templates.py`)

Pre-defined prompt templates for:
- **Object Selection**: Interactive object selection prompts
- **Question Answering**: QA with chain-of-thought reasoning
- **BLIP2**: Vision-language model prompts

### 2. Builders (`builders.py`)

Prompt builders that construct few-shot prompts:

- **ObjectSelectionPromptBuilder**: Build prompts for selecting relevant objects
- **QuestionAnsweringPromptBuilder**: Build QA prompts with CoT
- **BLIP2PromptBuilder**: Build BLIP2-specific prompts

### 3. Formatters (`formatters.py`)

Utilities for formatting and parsing:
- `process_answer()`: Clean and normalize answers
- `extract_answer_and_rationale()`: Parse LLM responses
- `compute_vqa_score()`: Compute VQA-style accuracy
- `filter_thoughts_by_similarity()`: Filter reasoning steps

### 4. Examples Manager (`examples.py`)

Manage few-shot examples for in-context learning:

- **FewShotExamplesManager**: Manage QA examples
- **ObjectSelectionExamplesManager**: Manage object selection examples

## Usage Examples

### Object Selection

```python
from vctp.think.prompts import ObjectSelectionPromptBuilder

builder = ObjectSelectionPromptBuilder(engine="chat")

# Build prompt with few-shot examples
prompt = builder.build(
    question="What is the person doing?",
    object_list=["person", "tennis racket", "ball", "court"],
    examples=[
        {
            "question": "What sport is being played?",
            "selected_object": "tennis racket"
        }
    ]
)
```

### Question Answering with Chain-of-Thought

```python
from vctp.think.prompts import QuestionAnsweringPromptBuilder

builder = QuestionAnsweringPromptBuilder(
    engine="gpt3",
    chain_of_thoughts=True
)

prompt = builder.build(
    question="What is the person doing?",
    context="A tennis player on a clay court.",
    scene_graph_text="person is holding racket. ball is flying.",
    examples=[
        {
            "question": "What sport is this?",
            "context": "An outdoor court with lines.",
            "answer": "tennis",
            "rationale": "The player holds a racket and there is a ball."
        }
    ]
)
```

### Answer Processing

```python
from vctp.think.prompts import (
    process_answer,
    extract_answer_and_rationale,
    compute_vqa_score
)

# Clean answer
answer = process_answer("The person is playing tennis.")
# -> "person playing tennis"

# Extract from LLM response
response = "tennis. The player is holding a racket on the court."
answer, rationale = extract_answer_and_rationale(response, chain_of_thoughts=True)
# answer: "tennis"
# rationale: "The player is holding a racket on the court."

# Compute score
score = compute_vqa_score("tennis", ["tennis", "tennis", "playing tennis"])
# -> 0.6 (2/3 * 0.3 = 0.6, capped at min(1.0, count*0.3))
```

### Few-Shot Examples Management

```python
from vctp.think.prompts import FewShotExamplesManager

manager = FewShotExamplesManager(
    train_questions={"123<->456": "What color is the sky?"},
    train_answers={"123<->456": ["blue", "light blue"]},
    train_rationales={"123<->456": ["The sky appears blue during daytime."]},
    train_captions={123: ["An outdoor scene with clear sky."]}
)

# Get random examples
example_keys = manager.get_random_examples(n_shot=8)

# Format examples for prompt
examples = manager.format_examples_batch(
    example_keys,
    include_rationale=True,
    include_choices=False
)
```

## Integration with LLM Adapters

```python
from vctp.think.llm import create_llm_adapter
from vctp.think.prompts import QuestionAnsweringPromptBuilder, process_answer

# Create LLM adapter
llm = create_llm_adapter(engine="gpt3", api_keys=["sk-..."])

# Build prompt
builder = QuestionAnsweringPromptBuilder(engine="gpt3", chain_of_thoughts=True)
prompt = builder.build(
    question="What is the person doing?",
    context="A person on a tennis court.",
    scene_graph_text="person holding racket",
    examples=[...]
)

# Generate answer
response = llm.generate(prompt, max_tokens=50)

# Parse result
answer, rationale = extract_answer_and_rationale(response.text)
print(f"Answer: {answer}")
print(f"Reasoning: {rationale}")
```

## Prompt Flow in Visual CoT

1. **Object Selection** (Interactive Phase):
   ```
   Question → ObjectSelectionPromptBuilder → LLM → Selected Object
   ```

2. **Question Answering** (Reasoning Phase):
   ```
   Context + Scene Graph + Question → QuestionAnsweringPromptBuilder → LLM → Answer + Rationale
   ```

3. **Iterative Refinement**:
   ```
   Previous Thoughts → Filter by Similarity → Update Context → Re-generate
   ```

