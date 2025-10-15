## Spec: Refactor VisualCoT (See–Think–Confirm) into a Reproducible Research Repo

**Status**: Draft  
**Scope**: Align with `.specify/constitution.md` and the Visual CoT for KVQA paper to deliver a modular, reproducible, and extensible repository structured around See–Think–Confirm (VCTP).

### 1) Cấu trúc thư mục (Directory Structure)

Mục tiêu: Phân tách rõ ràng module, cấu hình, dữ liệu, script, và kiểm thử. Các đường dẫn ở dưới mang tính đề xuất và có thể tinh chỉnh khi hiện thực.

```text
repo-root/
  README.md
  LICENSE
  .env.example
  requirements.txt
  environment.yml
  pyproject.toml
  setup.cfg
  configs/
    datasets/
      aokvqa.yaml
      okvqa.yaml
    models/
      llm_opt_1.3b.yaml
      llm_llama_7b.yaml
      clip_vit_l14.yaml
      detector_groundingdino.yaml
    pipelines/
      vctp_default.yaml
      vctp_self_consistency.yaml
    experiments/
      aokvqa_baseline.yaml
      okvqa_cot_clipconfirm.yaml
  data/
    raw/            # hướng dẫn: dataset gốc (A-OKVQA/OKVQA)
    processed/      # cache tiền xử lý (features, scene graphs, captions)
    artifacts/      # checkpoints, tokenizer, index, retrieval stores
  src/
    vctp/
      core/
        types.py        # dataclasses: EvidenceBundle, ReasoningOutput, ConfirmationOutput
        interfaces.py   # ABCs: PerceptionModule, ReasoningModule, ConfirmationModule
        registry.py     # registry/factory cho module theo tên trong YAML
        pipeline.py     # VCTPPipeline orchestration
        utils.py        # logging, seeding, io helpers
        config.py       # pydantic/omegaconf schemas
      see/
        perception.py   # Orchestrate detector/feature/caption extraction
        detectors/
          groundingdino.py
          detic.py
        features/
          clip_extractor.py
          vit_extractor.py
        captions/
          blip_captioner.py
        graphs/
          scene_graph_builder.py
      think/
        reasoner.py     # Orchestrate LLM prompting/CoT strategies
        prompts/
          base_prompt.txt
          deliberate_prompt.txt
        llm/
          opt_adapter.py
          llama_adapter.py
      confirm/
        confirmer.py    # Orchestrate verification/scoring
        scorers/
          clip_scorer.py
          retrieval_consistency.py
          rule_based.py
    cli/
      preprocess.py   # build features, captions, graphs from configs
      evaluate.py     # run pipeline and evaluate metrics
      train.py        # optional: fine-tune/adapt LLM or components
  scripts/
    download_data.sh
    preprocess.sh
    evaluate.sh
    train.sh
  notebooks/
    exploration.ipynb
    error_analysis.ipynb
  tests/
    test_core_types.py
    test_pipeline_smoke.py
    see/
      test_perception_minimal.py
    think/
      test_reasoner_prompt.py
    confirm/
      test_confirmer_clip.py
  docs/
    architecture.md
    reproduce.md
    modules.md
```

Ghi chú:
- `src/vctp/core` là lớp nền: interfaces, datamodels, registry, pipeline, config schema. Các module See/Think/Confirm chỉ phụ thuộc `core`.
- `configs/` chia theo `datasets/`, `models/`, `pipelines/`, `experiments/` giúp tái sử dụng và tái tạo.
- `scripts/` dùng CLI không tương tác, đọc cấu hình từ `configs/` và biến môi trường.

### 2) Thiết kế các Module chính (See–Think–Confirm)

Các interface và datamodels tham chiếu từ `src/vctp/core/interfaces.py` và `src/vctp/core/types.py`.

```python
# types.py (phác thảo)
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

@dataclass
class DetectedObject:
    name: str
    bbox: List[float]   # [x1, y1, x2, y2]
    score: float

@dataclass
class EvidenceBundle:
    image_id: str
    global_caption: Optional[str]
    detected_objects: List[DetectedObject]
    attributes: Dict[str, Any]
    relations: List[Dict[str, Any]]
    clip_image_embed: Optional[List[float]]
    region_captions: Optional[List[str]]

@dataclass
class ReasoningOutput:
    candidate_answer: str
    cot_rationale: str
    used_concepts: List[str]

@dataclass
class ConfirmationOutput:
    is_confirmed: bool
    score: float
    rationale: str
```

```python
# interfaces.py (phác thảo)
from abc import ABC, abstractmethod
from typing import Any, Dict
from .types import EvidenceBundle, ReasoningOutput, ConfirmationOutput

class PerceptionModule(ABC):
    @abstractmethod
    def run(self, image_path: str, question: str, **kwargs: Dict[str, Any]) -> EvidenceBundle:
        ...

class ReasoningModule(ABC):
    @abstractmethod
    def run(self, evidence: EvidenceBundle, question: str, **kwargs: Dict[str, Any]) -> ReasoningOutput:
        ...

class ConfirmationModule(ABC):
    @abstractmethod
    def run(self, question: str, candidate: ReasoningOutput, evidence: EvidenceBundle, **kwargs: Dict[str, Any]) -> ConfirmationOutput:
        ...
```

Chi tiết trách nhiệm và I/O:

- See Module (`PerceptionModule`)
  - **Chức năng**: Nhận ảnh + câu hỏi, chạy detector, trích xuất features (CLIP/ViT), caption tổng quan, scene graph; tổng hợp thành `EvidenceBundle`.
  - **Đầu vào**: `image_path`, `question`, `configs.models.detector|features|captioner`.
  - **Đầu ra**: `EvidenceBundle` chứa đối tượng, caption, embeddings, attributes/relations, region captions (nếu có).

- Think Module (`ReasoningModule`)
  - **Chức năng**: Dùng LLM và prompt CoT để chọn/lọc concepts từ `EvidenceBundle`, sinh rationale và câu trả lời ứng viên.
  - **Đầu vào**: `EvidenceBundle`, `question`, `configs.models.llm`, `configs.pipelines.prompt_template`.
  - **Đầu ra**: `ReasoningOutput(candidate_answer, cot_rationale, used_concepts)`.

- Confirm Module (`ConfirmationModule`)
  - **Chức năng**: Kiểm chứng/ghi điểm câu trả lời ứng viên dựa trên bằng chứng thị giác và/hoặc retrieval; ví dụ dùng CLIP để so khớp answer/rationale với vùng ảnh hoặc caption.
  - **Đầu vào**: `question`, `ReasoningOutput`, `EvidenceBundle`, `configs.models.clip|retriever`.
  - **Đầu ra**: `ConfirmationOutput(is_confirmed, score, rationale)`.

### 3) Quản lý Cấu hình (Configuration Management)

Sử dụng YAML cho cấu hình có khả năng kết hợp. `src/vctp/core/config.py` chịu trách nhiệm load/validate (pydantic/omegaconf). Các lớp cấu hình: `DatasetConfig`, `ModelConfig`, `PipelineConfig`, `ExperimentConfig`.

Ví dụ `configs/pipelines/vctp_default.yaml`:

```yaml
pipeline:
  name: vctp
  see:
    detector: groundingdino
    features: clip_vit_l14
    captioner: blip
    scene_graph: true
  think:
    llm: opt_1.3b
    prompt: base_prompt
    strategy: self_consistency
    num_samples: 5
  confirm:
    scorer: clip_scorer
    retrieval_consistency: false
    threshold: 0.45
```

Ví dụ `configs/experiments/aokvqa_baseline.yaml`:

```yaml
experiment:
  seed: 42
  dataset: aokvqa
  split: val
  batch_size: 8
  num_workers: 4
  output_dir: runs/aokvqa_baseline
  pipeline_config: configs/pipelines/vctp_default.yaml
models:
  llm: configs/models/llm_opt_1.3b.yaml
  clip: configs/models/clip_vit_l14.yaml
  detector: configs/models/detector_groundingdino.yaml
datasets:
  aokvqa: configs/datasets/aokvqa.yaml
```

Ví dụ `configs/datasets/aokvqa.yaml`:

```yaml
dataset:
  name: aokvqa
  root: data/raw/A-OKVQA
  images: images
  annotations: annotations
  processed: data/processed/aokvqa
```

### 4) Luồng thực thi (Execution Flow)

Ví dụ người dùng chạy `scripts/evaluate.sh` (hoặc CLI Python): đọc config → dựng module → tải dữ liệu → chạy VCTP → lưu kết quả và logs.

Pseudocode (`src/vctp/core/pipeline.py` orchestration):

```python
def run_pipeline(experiment_cfg):
    set_global_seed(experiment_cfg.experiment.seed)
    dataset = build_dataset(experiment_cfg.datasets[experiment_cfg.experiment.dataset], split=experiment_cfg.experiment.split)
    pipeline_cfg = load_yaml(experiment_cfg.experiment.pipeline_config)

    see = build_module('see', pipeline_cfg.pipeline.see, models_cfg=experiment_cfg.models)
    think = build_module('think', pipeline_cfg.pipeline.think, models_cfg=experiment_cfg.models)
    confirm = build_module('confirm', pipeline_cfg.pipeline.confirm, models_cfg=experiment_cfg.models)

    results = []
    for sample in dataset:
        evidence = see.run(sample.image_path, sample.question)
        reasoning = think.run(evidence, sample.question)
        confirmation = confirm.run(sample.question, reasoning, evidence)
        results.append({
            'image_id': sample.image_id,
            'question_id': sample.question_id,
            'answer': reasoning.candidate_answer,
            'confirmed': confirmation.is_confirmed,
            'score': confirmation.score,
            'rationale': reasoning.cot_rationale,
        })

    save_jsonl(results, output_dir=experiment_cfg.experiment.output_dir)
    return results
```

CLI ví dụ (`src/cli/evaluate.py`):

```python
import argparse
from vctp.core.config import load_experiment_config
from vctp.core.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to experiment YAML')
    args = parser.parse_args()
    cfg = load_experiment_config(args.config)
    run_pipeline(cfg)

if __name__ == '__main__':
    main()
```

Kết quả & logging:
- Lưu `results.jsonl` và `metrics.csv` vào `output_dir`, kèm `config_resolved.yaml` (cấu hình hợp nhất thực tế).
- Log theo `logging` với mức độ INFO/DEBUG; thêm `--dry-run` để tạo cấu trúc mà không chạy mô hình nặng.

### Acceptance & Success Criteria

- Có thể chạy end-to-end trên một subset nhỏ (smoke test) không cần GPU nặng.
- Thay đổi LLM/detector/CLIP bằng cách sửa YAML mà không sửa code pipeline.
- Tạo lại số liệu cơ bản trên A-OKVQA/OKVQA bằng script và config đi kèm.


