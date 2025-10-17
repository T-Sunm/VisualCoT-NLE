# Think Module - Visual Chain-of-Thought Reasoning

The `think/` module implements the reasoning and inference components for Visual CoT, including LLM adapters, prompt engineering, context retrieval, and core reasoning logic.

## Module Structure

```
think/
├── llm/                    # LLM Adapters
│   ├── base_adapter.py        # Base LLM interface
│   ├── types.py               # Data types
│   ├── openai_adapter.py      # GPT-3, ChatGPT
│   ├── opt_adapter.py         # Meta OPT models
│   ├── llama_adapter.py       # Meta LLaMA models
│   └── factory.py             # LLM factory
│
├── prompts/                # Prompt Engineering
│   ├── templates.py           # Prompt templates
│   ├── builders.py            # Prompt builders
│   ├── formatters.py          # Answer formatters
│   └── examples.py            # Examples manager
│
├── context/                # Context Retrieval
│   ├── similarity_retriever.py    # CLIP-based retrieval
│   ├── object_similarity.py       # Object similarity
│   └── context_manager.py         # Unified manager
│
├── reasoning/              # Core Reasoning
│   ├── object_selector.py         # Object selection
│   ├── question_answerer.py      # QA with CoT
│   └── thought_verifier.py        # Thought verification
│
├── interactive/            # Interactive Attention
│   ├── attention_strategy.py     # Attention strategies
│   ├── interactive_attention.py  # Multi-round attention
│   └── interactive_loop.py        # Full pipeline
│
└── reasoner.py             # Main Reasoner
```

## Quick Start

### 1. Basic Question Answering

```python
from vctp.think import (
    create_llm_adapter,
    QuestionAnswerer,
    FewShotExamplesManager
)

# Setup LLM
llm = create_llm_adapter(
    engine="gpt3",
    api_keys=["sk-..."],
    temperature=0.0
)

# Setup examples
examples_mgr = FewShotExamplesManager(
    train_questions=train_q_dict,
    train_answers=train_a_dict,
    train_rationales=train_r_dict
)

# Answer questions
answerer = QuestionAnswerer(
    llm=llm,
    chain_of_thoughts=True,
    n_ensemble=5
)

answer, rationale, confidence = answerer.answer(
    question="What is the person doing?",
    context="A person on a tennis court.",
    scene_graph_text="person holding racket",
    examples=examples_mgr.format_examples_batch([...])
)
```

### 2. Interactive Object Selection

```python
from vctp.think import ObjectSelector, InteractiveContextManager

# Setup context manager
context_mgr = InteractiveContextManager(
    similarity_path="coco_clip_new",
    sg_dir="scene_graphs",
    sg_attr_dir="scene_graphs_attr",
    train_questions=train_q,
    train_answers=train_a
)
context_mgr.load_features()

# Select object
selector = ObjectSelector(llm=llm, engine="gpt3")

# Get examples
examples = context_mgr.get_examples_with_object_selection(
    query_key="123<->456",
    n_shot=8
)

# Select from objects
selected_idx = selector.select_object(
    question="What is the person doing?",
    objects=[[0.95, "person", ...], [0.90, "racket", ...]],
    examples=examples
)
```

### 3. Interactive Visual CoT Pipeline

```python
from vctp.think import (
    create_llm_adapter,
    InteractiveLoop,
    InteractiveAttention,
    LLMAttentionStrategy,
    ObjectSelector,
    QuestionAnswerer,
    ThoughtVerifier,
    InteractiveContextManager,
    FewShotExamplesManager,
)

# Setup
llm = create_llm_adapter(engine="gpt3", api_keys=["sk-..."])

# Context and examples
context_mgr = InteractiveContextManager(...)
context_mgr.load_features(metric="imagequestion")
examples_mgr = FewShotExamplesManager(...)

# Components
object_selector = ObjectSelector(llm=llm, engine="gpt3")
question_answerer = QuestionAnswerer(llm=llm, chain_of_thoughts=True, n_ensemble=5)
thought_verifier = ThoughtVerifier(use_clip=True)

# Attention strategy
attention_strategy = LLMAttentionStrategy(
    object_selector=object_selector,
    context_manager=context_mgr,
    n_shot=8
)

# Interactive attention
interactive_attention = InteractiveAttention(
    attention_strategy=attention_strategy,
    max_rounds=3,
    stop_on_convergence=True
)

# Full pipeline
interactive_loop = InteractiveLoop(
    interactive_attention=interactive_attention,
    question_answerer=question_answerer,
    context_manager=context_mgr,
    examples_manager=examples_mgr,
    thought_verifier=thought_verifier,
    n_shot_qa=16,
    n_ensemble=5,
    chain_of_thoughts=True
)

# Run on a sample
result = interactive_loop.run_single_sample(
    query_key="123<->456",
    question="What is the person doing?",
    objects=[[0.95, "person", ["standing"]], [0.90, "tennis racket", ["green"]]],
    global_caption="A person on a tennis court.",
    reference_answer=["playing tennis", "tennis"]
)

print(f"Answer: {result['answer']}")
print(f"Rationale: {result['rationale']}")
print(f"Rounds: {result['rounds']}")
print(f"Accuracy: {result['accuracy']}")
```

### 4. Full Visual CoT with VisualCoTReasoner

```python
from vctp.think import VisualCoTReasoner, ContextManager
from vctp.core.types import EvidenceBundle, DetectedObject

# Setup
llm = create_llm_adapter(engine="gpt3", api_keys=["sk-..."])
context_mgr = ContextManager(...)
context_mgr.load_features()

examples_mgr = FewShotExamplesManager(...)

# Create reasoner
reasoner = VisualCoTReasoner(
    llm=llm,
    examples_manager=examples_mgr,
    chain_of_thoughts=True,
    n_ensemble=5,
    use_thought_verification=True
)

# Get context examples
example_keys = context_mgr.get_qa_context_examples(
    query_key="123<->456",
    n_shot=16
)

# Create evidence
evidence = EvidenceBundle(
    detected_objects=[
        DetectedObject(label="person", confidence=0.95),
        DetectedObject(label="tennis racket", confidence=0.90),
    ],
    global_caption="A person on court"
)

# Run reasoning
result = reasoner.run(
    evidence=evidence,
    question="What is the person doing?",
    context_caption="A person on a tennis court",
    example_keys=example_keys
)

print(f"Answer: {result.candidate_answer}")
print(f"Rationale: {result.cot_rationale}")
```

## Core Concepts

### 1. Chain-of-Thought Reasoning

Visual CoT uses chain-of-thought prompting to generate step-by-step reasoning:

```
Question: What is the person doing?
Context: A person on court. person is holding racket. ball is flying.

Answer: playing tennis. The person holds a tennis racket and there is a ball, indicating they are playing tennis.
         ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
         Answer          Reasoning/Rationale
```

### 2. Interactive Object Selection

The system iteratively selects relevant objects to look at:

```
Round 1: Select "tennis racket" → Answer: playing tennis
Round 2: Select "ball" → Answer: playing tennis (more confident)
Round 3: Select "court" → Answer: playing tennis (most confident)
```

### 3. Thought Verification

Verify generated thoughts against image using CLIP or BLIP2:

```
Generated: "The person is playing tennis. The sky is blue."
             ✓ matches image (keep)     ✗ doesn't match (filter)
Verified: "The person is playing tennis."
```

### 4. Few-Shot In-Context Learning

Retrieve similar examples using CLIP similarity:

```
Query: "What sport is being played?" + image
  ↓ CLIP similarity
Examples:
  1. "What is the person doing?" → "playing tennis" (sim: 0.92)
  2. "What equipment is used?" → "tennis racket" (sim: 0.88)
  3. ...
```

## Components Overview

### LLM Adapters (`llm/`)

Unified interface for different language models:

- **OpenAI**: GPT-3, ChatGPT (API-based)
- **OPT**: Meta's OPT models (local)
- **LLaMA**: Meta's LLaMA models (local)

All adapters support:
- Text generation
- Logit bias for constrained decoding
- Logprobs extraction
- API key rotation
- Auto-retry on failures

### Prompts (`prompts/`)

Comprehensive prompt engineering utilities:

- **Templates**: Pre-defined prompts for object selection, QA
- **Builders**: Construct few-shot prompts dynamically
- **Formatters**: Parse and clean LLM outputs
- **Examples Manager**: Manage and format training examples

### Context (`context/`)

Similarity-based example retrieval:

- **SimilarityRetriever**: Retrieve examples using CLIP features
- **ObjectSimilarityComputer**: Compute object-answer similarity
- **ContextManager**: Unified interface for both phases

### Reasoning (`reasoning/`)

Core reasoning components:

- **ObjectSelector**: Select relevant objects interactively
- **QuestionAnswerer**: Answer with chain-of-thought
- **ThoughtVerifier**: Verify reasoning with image

### Interactive (`interactive/`)

Interactive attention mechanism:

- **AttentionStrategy**: Different object selection strategies (LLM, Random, Oracle)
- **InteractiveAttention**: Multi-round attention orchestration
- **InteractiveLoop**: Complete See-Think pipeline with iteration

### Reasoner (`reasoner.py`)

Main orchestrator that combines all components:

- Manages iterative refinement
- Handles ensemble methods
- Supports ablation studies
- Integrates with perception module

## Configuration Examples

### GPT-3 with Ensemble

```python
from vctp.think import create_llm_adapter, QuestionAnswerer

llm = create_llm_adapter(
    engine="gpt3",
    engine_name="text-davinci-003",
    api_keys=["sk-..."],
    temperature=0.0,
    max_tokens=41
)

answerer = QuestionAnswerer(
    llm=llm,
    chain_of_thoughts=True,
    n_ensemble=5  # 5 generations, pick best
)
```

### Local OPT-66B

```python
llm = create_llm_adapter(
    engine="opt",
    engine_name="facebook/opt-66b",
    device="auto"  # Multi-GPU
)
```

### With CLIP Verification

```python
from vctp.think import ThoughtVerifier

verifier = ThoughtVerifier(
    use_clip=True,
    threshold=0.0  # Filter thoughts below threshold
)

reasoner = VisualCoTReasoner(
    llm=llm,
    examples_manager=examples_mgr,
    use_thought_verification=True,
    thought_verifier=verifier
)
```

## Ablation Studies

The module supports various ablation experiments:

```python
# Random object selection
from vctp.think import RandomObjectSelector
selector = RandomObjectSelector()

# Oracle object selection
from vctp.think import OracleObjectSelector
selector = OracleObjectSelector(oracle_attend_dict=scores)

# Remove visual evidence
reasoner = VisualCoTReasoner(..., ablation_visual=True)

# Remove reasoning
reasoner = VisualCoTReasoner(..., ablation_reason=True)

# Oracle rationales
from vctp.think import OracleThoughtVerifier
verifier = OracleThoughtVerifier(rationale_dict=gt_rationales)
```

## Performance Optimization

### 1. Caching
```python
# Cache LLM responses
response_cache = {}
key = (prompt, max_tokens, temperature)
if key in response_cache:
    response = response_cache[key]
else:
    response = llm.generate(prompt)
    response_cache[key] = response
```

### 2. Batch Processing
```python
# Process multiple questions in parallel
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(reasoner.run, evidence, question)
        for evidence, question in zip(evidences, questions)
    ]
    results = [f.result() for f in futures]
```

### 3. Feature Preloading
```python
# Load CLIP features once
context_mgr.load_features(metric="imagequestion")

# Reuse for all queries
for query_key in queries:
    examples = context_mgr.get_qa_context_examples(query_key)
```

## Debugging

Enable debug mode for detailed logging:

```python
reasoner = VisualCoTReasoner(
    llm=llm,
    examples_manager=examples_mgr,
    debug=True  # Print prompts and intermediate results
)

selector = ObjectSelector(llm=llm, debug=True)
answerer = QuestionAnswerer(llm=llm, debug=True)
verifier = ThoughtVerifier(use_clip=True, debug=True)
```

## See Also

- `llm/README.md` - LLM adapter details
- `prompts/README.md` - Prompt engineering guide
- `context/README.md` - Context retrieval guide
- `reasoning/README.md` - Core reasoning components

