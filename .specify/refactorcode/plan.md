## Implementation Plan: Refactor VisualCoT (See–Think–Confirm)

Branch: `[refactor-visualcot]` | Date: [YYYY-MM-DD] | Spec: `.specify/refactorcode/specs.md`
Inputs: `.specify/refactorcode/constitution.md`, Visual CoT for KVQA paper

### Giai đoạn 1: Khởi tạo Nền tảng (Setup & Scaffolding)

- **Mục tiêu (Objective)**: Thiết lập khung dự án thống nhất theo See–Think–Confirm; đảm bảo môi trường có thể tái tạo và quy ước coding rõ ràng.
- **Deliverables**:
  - Cấu trúc thư mục cơ bản theo spec (có `__init__.py` đầy đủ):
    ```text
    repo-root/
      README.md
      LICENSE
      .env.example
      requirements.txt
      environment.yml
      pyproject.toml
      setup.cfg
      configs/{datasets,models,pipelines,experiments}
      data/{raw,processed,artifacts}
      src/vctp/{core,see,think,confirm}
      scripts/
      tests/
      docs/
    ```
  - Khởi tạo files cấu hình môi trường: `requirements.txt`, `environment.yml`, `pyproject.toml`, `setup.cfg` (PEP8/flake8/isort/black).
  - Tạo `README.md` khởi điểm (mục tiêu, cách cài đặt nhanh) và `LICENSE`.
  - Thiết lập logging tối thiểu và seeding helper trong `src/vctp/core/utils.py`.
  - Chuẩn bị stub files: `src/vctp/core/{types.py,interfaces.py,registry.py,pipeline.py,config.py}` rỗng có docstring.
  - Script khởi tạo mẫu: `scripts/download_data.sh` (chỉ tạo placeholder, hướng dẫn thủ công nếu cần).

### Giai đoạn 2: Xây dựng Luồng Dữ liệu và Cấu hình (Data & Config Pipeline)

- **Mục tiêu (Objective)**: Hiện thực hóa cơ chế nạp/tải dữ liệu OKVQA/A-OKVQA và hệ quản trị cấu hình YAML được validate; thiết lập quy trình tiền xử lý cơ bản/caching.
- **Deliverables**:
  - `src/vctp/core/config.py`: bộ schema và loader (pydantic/omegaconf) cho `DatasetConfig`, `ModelConfig`, `PipelineConfig`, `ExperimentConfig` + hợp nhất cấu hình.
  - `configs/` mẫu:
    - `configs/datasets/{aokvqa.yaml, okvqa.yaml}`
    - `configs/models/{llm_opt_1.3b.yaml, llm_llama_7b.yaml, clip_vit_l14.yaml, detector_groundingdino.yaml}`
    - `configs/pipelines/{vctp_default.yaml, vctp_self_consistency.yaml}`
    - `configs/experiments/{aokvqa_baseline.yaml, okvqa_cot_clipconfirm.yaml}`
  - Module dữ liệu: `src/vctp/data/loader.py` với `build_dataset(cfg, split)` và dataset wrappers (A-OKVQA/OKVQA), trả về iterable có `image_id, image_path, question, question_id`.
  - Tiền xử lý tối thiểu: CLI `scripts/preprocess.py` (hoặc `src/cli/preprocess.py`) để tạo cache `data/processed/` (ví dụ: captions, features placeholder), ghi metadata vào JSONL.
  - Kiểm tra tích hợp nhẹ: `tests/test_dataset_smoke.py` xác nhận có thể đọc một vài mẫu từ mỗi dataset với config tương ứng.

### Giai đoạn 3: Hiện thực hóa các Module Cốt lõi (Core Modules Implementation)

- **Mục tiêu (Objective)**: Cài đặt các module See–Think–Confirm theo interfaces; cung cấp triển khai tham chiếu tối thiểu nhưng chạy được.
- **Deliverables**:
  - `src/vctp/core/types.py`: dataclasses `DetectedObject`, `EvidenceBundle`, `ReasoningOutput`, `ConfirmationOutput` hoàn chỉnh.
  - `src/vctp/core/interfaces.py`: ABC `PerceptionModule`, `ReasoningModule`, `ConfirmationModule` với chữ ký thống nhất.
  - SEE:
    - `src/vctp/see/perception.py`: lớp `Perception` điều phối detector/feature/caption/scene-graph theo config.
    - Triển khai tối thiểu: `features/clip_extractor.py`, `captions/blip_captioner.py` (hoặc mock), `detectors/groundingdino.py` (stub nếu cần), `graphs/scene_graph_builder.py` (tùy chọn, stub).
  - THINK:
    - `src/vctp/think/reasoner.py`: lớp `Reasoner` sinh CoT và answer candidate từ `EvidenceBundle`.
    - Adapters tối thiểu: `llm/opt_adapter.py` và `llm/llama_adapter.py` (API chung), `prompts/base_prompt.txt`.
  - CONFIRM:
    - `src/vctp/confirm/confirmer.py`: lớp `Confirmer` chấm điểm/kiểm chứng answer.
    - Scorer tối thiểu: `scorers/clip_scorer.py` (dựa trên CLIP image/text embedding) + `rule_based.py` (fallback đơn giản).
  - Unit tests cơ bản: `tests/see/test_perception_minimal.py`, `tests/think/test_reasoner_prompt.py`, `tests/confirm/test_confirmer_clip.py` chạy nhanh (mock nếu cần).

### Giai đoạn 4: Tích hợp Pipeline và Hoàn thiện Scripts (Pipeline Integration & Scripting)

- **Mục tiêu (Objective)**: Kết nối SEE→THINK→CONFIRM trong `VCTPPipeline`; cung cấp script hoàn chỉnh để evaluate/infer dựa trên YAML, có lưu kết quả và logs.
- **Deliverables**:
  - `src/vctp/core/pipeline.py`: `VCTPPipeline.run(sample)` + `run_dataset(dataset)` thực thi tuần tự See→Think→Confirm, lưu trace (bằng chứng, prompt, answer, scores).
  - `src/vctp/core/registry.py`: registry/factory ánh xạ tên trong YAML đến lớp triển khai.
  - Scripts có thể chạy trực tiếp:
    - `scripts/evaluate.py`: nhận `--config configs/experiments/xxx.yaml`, chạy pipeline trên split, xuất `results.jsonl`, `metrics.csv`, `config_resolved.yaml`.
    - `scripts/inference.py`: nhận ảnh + câu hỏi hoặc một file JSONL input, trả JSONL các kết quả.
    - (tùy chọn) `scripts/preprocess.py`: dựng cache features/captions/graphs theo config.
  - Smoke test pipeline: `tests/test_pipeline_smoke.py` chạy trên vài mẫu (CPU), xác minh không lỗi và có output files.

### Giai đoạn 5: Kiểm thử, Hoàn thiện Tài liệu và Dọn dẹp (Testing, Documentation & Finalization)

- **Mục tiêu (Objective)**: Đảm bảo chất lượng, dễ dùng, và khả năng tái tạo theo constitution.
- **Deliverables**:
  - Kiểm thử:
    - Unit tests bổ sung cho `core/types`, `core/config`, registry, và scorers; độ bao phủ cơ bản.
    - CI khởi điểm (nếu có): lint + tests trên subset (không GPU).
  - Tài liệu:
    - `docs/architecture.md` (sơ đồ See–Think–Confirm, sequence, interfaces).
    - `docs/reproduce.md` (các lệnh từ `download_data.sh` → `preprocess.py` → `evaluate.py`).
    - `docs/modules.md` (cách thêm module mới qua registry + YAML).
    - `README.md` cập nhật: quickstart, cấu trúc repo, ví dụ lệnh.
  - Chất lượng mã:
    - Docstrings theo chuẩn (Google/NumPy), type hints cho public APIs.
    - Thiết lập `flake8`, `black`, `isort` và fix các vi phạm chính.
  - Dọn dẹp:
    - Loại bỏ script cũ dư thừa, hợp nhất logic còn dùng vào modules mới.
    - Gắn tag release `v0.1.0` (baseline tái tạo được) với checksum kết quả.

---

### Ghi chú thực thi & tiêu chí chấp nhận theo giai đoạn

- Mỗi giai đoạn kết thúc khi: (a) deliverables tồn tại trong repo, (b) smoke test tối thiểu chạy qua, (c) tài liệu gắn kèm đã cập nhật.
- Tất cả tham số và đường dẫn phải đến từ YAML; seed được ấn định trong pipeline; logs ghi ra `output_dir`.
- Cho phép dùng mock/stub ở GĐ3–4 miễn đảm bảo interface ổn định; thay thế bằng triển khai thật ở nhánh phụ nếu cần.


