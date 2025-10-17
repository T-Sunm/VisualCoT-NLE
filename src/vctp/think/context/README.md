# Context Retrieval Module

This module handles context retrieval for few-shot in-context learning in Visual CoT.

## Components

### 1. SimilarityRetriever (`similarity_retriever.py`)

Retrieves similar examples using pre-computed CLIP features.

**Features:**
- Load CLIP image and text features
- Support both A-OKVQA and OK-VQA datasets
- Multiple similarity metrics: question-only, image+question
- Efficient numpy-based similarity computation

**Usage:**
```python
from vctp.think.context import SimilarityRetriever

retriever = SimilarityRetriever(
    similarity_path="path/to/clip/features",
    dataset_name="aokvqa",
    split="val"
)

# Load features
retriever.load_features(metric="imagequestion")

# Get similar examples
similar_keys = retriever.get_similar_examples(
    query_key="123<->456",
    metric="imagequestion",
    n_shot=16
)
```

### 2. ObjectSimilarityComputer (`object_similarity.py`)

Computes object-level similarity for interactive selection.

**Features:**
- Load objects from scene graphs
- Compute rationale-based similarity (keyword matching)
- Compute answer-based similarity (CLIP text embeddings)
- Rank objects by relevance

**Usage:**
```python
from vctp.think.context import ObjectSimilarityComputer

computer = ObjectSimilarityComputer(
    sg_dir="path/to/scene_graphs",
    sg_attr_dir="path/to/scene_graphs_attr",
    train_questions=train_q_dict,
    train_answers=train_a_dict,
    train_rationales=train_r_dict
)

# Compute object similarity
obj_list, conf_list, sim_dict = computer.compute_object_similarity(
    example_key="123<->456",
    metric="answer"  # or "rationale"
)

# Get most relevant object
best_obj = computer.get_most_relevant_object(
    example_key="123<->456",
    metric="answer"
)
```

### 3. ContextManager (`context_manager.py`)

Unified manager for both QA and interactive context retrieval.

**Features:**
- Manage both similarity retrieval and object similarity
- Get QA context examples (standard few-shot)
- Get interactive context examples (with object selection)
- Validate examples
- Filter valid examples

**Usage:**
```python
from vctp.think.context import ContextManager

manager = ContextManager(
    similarity_path="path/to/clip/features",
    sg_dir="path/to/scene_graphs",
    sg_attr_dir="path/to/scene_graphs_attr",
    train_questions=train_q_dict,
    train_answers=train_a_dict,
    train_rationales=train_r_dict,
    dataset_name="aokvqa",
    split="val"
)

# Load features
manager.load_features(metric="imagequestion")

# Get QA context examples
qa_examples = manager.get_qa_context_examples(
    query_key="123<->456",
    metric="imagequestion",
    n_shot=16
)

# Get interactive context examples with object selections
interactive_examples, obj_sims = manager.get_interactive_context_examples(
    query_key="123<->456",
    metric="imagequestion",
    n_shot=8,
    object_sim_metric="answer"
)
```

### 4. InteractiveContextManager

Specialized manager for interactive object selection phase.

**Usage:**
```python
from vctp.think.context import InteractiveContextManager

manager = InteractiveContextManager(
    similarity_path="path/to/clip/features",
    sg_dir="path/to/scene_graphs",
    sg_attr_dir="path/to/scene_graphs_attr",
    train_questions=train_q_dict,
    train_answers=train_a_dict
)

manager.load_features(metric="imagequestion")

# Get formatted examples for object selection prompts
examples = manager.get_examples_with_object_selection(
    query_key="123<->456",
    n_shot=8
)

# examples = [
#     {
#         "question": "What is the person doing?",
#         "selected_object": "tennis racket",
#         "example_key": "789<->012"
#     },
#     ...
# ]
```

## Similarity Metrics

### Question Similarity
- Uses only text features from CLIP
- Computes cosine similarity between question embeddings
- Fast but ignores visual content

### Image+Question Similarity
- Combines image and question features
- `similarity = question_similarity + image_similarity`
- Better performance as it considers both modalities

### Object Similarity Metrics

#### Rationale-based (keyword matching)
```python
computer.compute_object_similarity(key, metric="rationale")
```
- Counts object mentions in rationales
- Fast but simple
- Works without CLIP

#### Answer-based (CLIP similarity)
```python
computer.compute_object_similarity(key, metric="answer")
```
- Uses CLIP to compute semantic similarity
- Compares object names with answer text
- More accurate but requires GPU

## Integration with Other Modules

### With Prompts Module

```python
from vctp.think.context import ContextManager
from vctp.think.prompts import FewShotExamplesManager, ObjectSelectionPromptBuilder

# Setup context retrieval
context_mgr = ContextManager(...)
context_mgr.load_features()

# Get similar examples
example_keys = context_mgr.get_qa_context_examples(
    query_key="123<->456",
    n_shot=16
)

# Format examples for prompts
examples_mgr = FewShotExamplesManager(...)
examples = examples_mgr.format_examples_batch(example_keys)

# Build prompt
builder = ObjectSelectionPromptBuilder(engine="gpt3")
prompt = builder.build(
    question="What is happening?",
    object_list=["person", "racket", "ball"],
    examples=examples
)
```

### Full Pipeline Example

```python
from vctp.think.context import InteractiveContextManager
from vctp.think.prompts import ObjectSelectionPromptBuilder
from vctp.think.llm import create_llm_adapter

# 1. Setup context manager
context_mgr = InteractiveContextManager(
    similarity_path="coco_clip_new",
    sg_dir="scene_graphs",
    sg_attr_dir="scene_graphs_attr",
    train_questions=train_q,
    train_answers=train_a
)
context_mgr.load_features(metric="imagequestion")

# 2. Get few-shot examples with object selections
examples = context_mgr.get_examples_with_object_selection(
    query_key="123<->456",
    n_shot=8
)

# 3. Build prompt
builder = ObjectSelectionPromptBuilder(engine="gpt3")
prompt = builder.build(
    question="What is the person doing?",
    object_list=["person", "tennis racket", "ball", "court"],
    examples=examples
)

# 4. Generate with LLM
llm = create_llm_adapter(engine="gpt3", api_keys=["sk-..."])
response = llm.generate(prompt)

# 5. Parse selected object
selected_idx = # ... parse from response
selected_object = object_list[selected_idx]
```

## Data Requirements

### Pre-computed CLIP Features

The module expects pre-computed CLIP features in the following structure:

```
similarity_path/
├── aokvqa_qa_line2sample_idx_train2017.json
├── aokvqa_qa_line2sample_idx_val2017.json
├── coco_clip_vitb16_train2017_aokvqa_question.npy
├── coco_clip_vitb16_val2017_aokvqa_question.npy
├── coco_clip_vitb16_train2017_aokvqa_convertedidx_image.npy
└── coco_clip_vitb16_val2017_aokvqa_convertedidx_image.npy
```

Generate these using the preprocessing scripts in `VisualCoT/preprocess/`.

### Scene Graphs

Scene graphs should be in JSON format:

```json
[
  [
    {
      "rect": [x1, y1, x2, y2],
      "class": "person",
      "conf": 0.95,
      "attr": ["standing", "wearing red"]
    },
    ...
  ]
]
```

## Performance Tips

1. **Load features once**: Call `load_features()` during initialization, not per query
2. **Batch retrieval**: Retrieve multiple queries together if possible
3. **Cache results**: Cache similarity scores for frequently used queries
4. **Use question-only for speed**: If visual similarity is not critical
5. **Pre-filter objects**: Use confidence threshold to reduce object list size

