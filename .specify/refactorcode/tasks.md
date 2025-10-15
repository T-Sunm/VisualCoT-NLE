---
description: "Executable task checklist for VisualCoT refactor"
---

# Tasks: VisualCoT Refactor (See–Think–Confirm)

Input: `.specify/refactorcode/specs.md`, `.specify/refactorcode/plan.md`
Prerequisites: Constitution, Spec, Plan ready

## Phase 1: Setup & Scaffolding

- [ ] Create directories per plan:
  - [ ] Create `configs/datasets/`, `configs/models/`, `configs/pipelines/`, `configs/experiments/`
  - [ ] Create `data/raw/`, `data/processed/`, `data/artifacts/`
  - [ ] Create `src/vctp/core/`, `src/vctp/see/`, `src/vctp/think/`, `src/vctp/confirm/`
  - [ ] Create `scripts/`, `tests/`, `docs/`
- [ ] Add `__init__.py` to all `src/vctp/**/` directories
- [ ] Create `requirements.txt` with core libs (placeholders):
  - [ ] Add `torch`, `transformers`, `tqdm`, `numpy`, `pillow`, `pydantic`/`omegaconf`, `pyyaml`
- [ ] Create `environment.yml` mirroring `requirements.txt`
- [ ] Create `pyproject.toml` (optional) and `setup.cfg` with linters (flake8, isort, black) basic config
- [ ] Create `.env.example` with placeholders (e.g., `CUDA_VISIBLE_DEVICES`, `WANDB_DISABLED`)
- [ ] Create `README.md` skeleton (project goal, quickstart install)
- [ ] Create `LICENSE` (choose MIT/Apache-2.0)
- [ ] Bootstrap core stubs with docstrings:
  - [ ] Create `src/vctp/core/types.py` (empty dataclass stubs)
  - [ ] Create `src/vctp/core/interfaces.py` (empty ABC stubs)
  - [ ] Create `src/vctp/core/registry.py` (empty registry dict + TODO functions)
  - [ ] Create `src/vctp/core/pipeline.py` (class stub `VCTPPipeline`)
  - [ ] Create `src/vctp/core/config.py` (loader stub)
- [ ] Create `src/vctp/core/utils.py` with `set_global_seed(seed)` and `get_logger(name)` placeholders
- [ ] Create `scripts/download_data.sh` with comments on where to place datasets

## Phase 2: Data & Config Pipeline

- [ ] Implement config schemas/loader `src/vctp/core/config.py`:
  - [ ] Define `DatasetConfig`, `ModelConfig`, `PipelineConfig`, `ExperimentConfig` (pydantic or omegaconf)
  - [ ] Implement `load_experiment_config(path)` that resolves/merges referenced YAMLs
- [ ] Add sample YAMLs:
  - [ ] `configs/datasets/aokvqa.yaml`, `configs/datasets/okvqa.yaml`
  - [ ] `configs/models/llm_opt_1.3b.yaml`, `configs/models/llm_llama_7b.yaml`
  - [ ] `configs/models/clip_vit_l14.yaml`, `configs/models/detector_groundingdino.yaml`
  - [ ] `configs/pipelines/vctp_default.yaml`, `configs/pipelines/vctp_self_consistency.yaml`
  - [ ] `configs/experiments/aokvqa_baseline.yaml`, `configs/experiments/okvqa_cot_clipconfirm.yaml`
- [ ] Implement dataset loader `src/vctp/data/loader.py`:
  - [ ] Create function `build_dataset(dataset_cfg, split)` returning iterable dicts
  - [ ] Implement A-OKVQA wrapper reading images/questions, yielding `image_id, image_path, question, question_id`
  - [ ] Implement OKVQA wrapper similarly
- [ ] Add preprocess CLI `src/cli/preprocess.py`:
  - [ ] Parse `--config` (experiment)
  - [ ] Build dataset and emit placeholder caches to `data/processed/<dataset>/`
  - [ ] Write metadata JSONL with per-sample paths
- [ ] Add smoke test `tests/test_dataset_smoke.py` to iterate 3 samples from each dataset

## Phase 3: Core Modules Implementation

- [ ] Flesh out datamodels `src/vctp/core/types.py`:
  - [ ] Add `DetectedObject`, `EvidenceBundle`, `ReasoningOutput`, `ConfirmationOutput`
- [ ] Flesh out interfaces `src/vctp/core/interfaces.py`:
  - [ ] Implement ABCs: `PerceptionModule.run`, `ReasoningModule.run`, `ConfirmationModule.run`
- [ ] SEE implementation:
  - [ ] Create `src/vctp/see/perception.py` with class `Perception(PerceptionModule)`
  - [ ] Create `src/vctp/see/features/clip_extractor.py` (stub extraction using CLIP)
  - [ ] Create `src/vctp/see/captions/blip_captioner.py` (stub caption)
  - [ ] Create `src/vctp/see/detectors/groundingdino.py` (stub detector)
  - [ ] Create `src/vctp/see/graphs/scene_graph_builder.py` (optional stub)
- [ ] THINK implementation:
  - [ ] Create `src/vctp/think/reasoner.py` with class `Reasoner(ReasoningModule)`
  - [ ] Create `src/vctp/think/llm/opt_adapter.py` (common interface)
  - [ ] Create `src/vctp/think/llm/llama_adapter.py`
  - [ ] Add `src/vctp/think/prompts/base_prompt.txt`
- [ ] CONFIRM implementation:
  - [ ] Create `src/vctp/confirm/confirmer.py` with class `Confirmer(ConfirmationModule)`
  - [ ] Create `src/vctp/confirm/scorers/clip_scorer.py`
  - [ ] Create `src/vctp/confirm/scorers/rule_based.py`
- [ ] Add unit tests:
  - [ ] `tests/see/test_perception_minimal.py` (mocks)
  - [ ] `tests/think/test_reasoner_prompt.py`
  - [ ] `tests/confirm/test_confirmer_clip.py`

## Phase 4: Pipeline Integration & Scripting

- [ ] Implement registry `src/vctp/core/registry.py`:
  - [ ] Add registries for see/think/confirm modules and factories
  - [ ] Map YAML names → classes
- [ ] Implement pipeline `src/vctp/core/pipeline.py`:
  - [ ] Add `VCTPPipeline.__init__(see, think, confirm)`
  - [ ] Add `run(sample)` → returns dict with answer, scores, rationale
  - [ ] Add `run_dataset(dataset)` → list of dicts, saves JSONL to output_dir
- [ ] Add CLI `src/cli/evaluate.py`:
  - [ ] Parse `--config`
  - [ ] Load experiment config, build modules via registry, run pipeline
  - [ ] Save `results.jsonl`, `metrics.csv`, `config_resolved.yaml`
- [ ] Add CLI `src/cli/inference.py`:
  - [ ] Parse `--image`, `--question` or `--input_jsonl`
  - [ ] Load config, run pipeline on inputs, print/save outputs
- [ ] Add smoke test `tests/test_pipeline_smoke.py`

## Phase 5: Testing, Documentation & Finalization

- [ ] Expand tests:
  - [ ] `tests/test_core_types.py` for dataclasses
  - [ ] `tests/test_core_config.py` for config loader/validation
  - [ ] Additional scorer tests
- [ ] Setup CI (optional):
  - [ ] Add GitHub Actions workflow for `lint` and `tests` (CPU only)
- [ ] Documentation:
  - [ ] Create `docs/architecture.md` with diagrams and sequence
  - [ ] Create `docs/reproduce.md` with exact commands (download → preprocess → evaluate)
  - [ ] Create `docs/modules.md` explaining registry and how to add modules
  - [ ] Update `README.md` quickstart and structure section
- [ ] Code quality:
  - [ ] Add docstrings and type hints to all public APIs
  - [ ] Configure `flake8`, `black`, `isort` in `setup.cfg`; run and fix issues
- [ ] Cleanup & release:
  - [ ] Remove deprecated scripts and references
  - [ ] Tag baseline release `v0.1.0` and include checksums/metrics


