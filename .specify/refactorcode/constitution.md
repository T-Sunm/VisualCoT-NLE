## Hiến chương dự án: VisualCoT (See–Think–Confirm)

### Tầm nhìn dự án (Project Vision)

- **Mục tiêu**: Xây dựng một repository nghiên cứu sạch, rõ ràng, có khả năng tái tạo hoàn toàn các kết quả trong paper VisualCoT/KVQA, và là nền tảng vững chắc để mở rộng các ý tưởng mới.
- **Kết quả mong đợi**: Một mã nguồn có cấu trúc module theo 3 giai đoạn See–Think–Confirm (VCTP), kèm quy trình dữ liệu, cấu hình, và script tự động hóa training/evaluation; có test tối thiểu và tài liệu đầy đủ để người dùng mới khởi động nhanh.
- **Phạm vi**: Hỗ trợ tối thiểu A-OKVQA/OKVQA; cho phép thay thế LLM (OPT/LLaMA/…), detector/feature extractor (CLIP/ViT/…), và module xác nhận (retrieval/consistency-check) thông qua cấu hình mà không chỉnh sâu logic.

### Các Nguyên tắc Cốt lõi (Core Principles)

#### 1) Tính Module hóa (Modularity)

- **Ba giai đoạn rõ ràng**:
  - **See**: Trích xuất bằng chứng thị giác (đối tượng, thuộc tính, quan hệ, caption, CLIP features, scene graph…).
  - **Think**: Lập luận/chuỗi suy nghĩ (chain-of-thought) để tạo giả thuyết/đáp án từ bằng chứng + câu hỏi.
  - **Confirm**: Xác nhận/điểm số/giải thích lại bằng retrieval, kiểm tra nhất quán, hoặc đối chiếu tri thức.
- **Ranh giới module**: Mỗi module có interface rõ ràng, I/O chuẩn hóa bằng dataclass/type hints, có test đơn vị tối thiểu.
- **Cấu trúc thư mục định hướng module** (đề xuất):
  - `vctp/see/`: perception, detector, feature extractor, scene-graph builder.
  - `vctp/think/`: reasoner/LLM adapters, prompt templates, CoT strategies.
  - `vctp/confirm/`: verifier/scorer, retrieval augmentors, consistency checks.
  - `vctp/core/`: interfaces (ABC), datamodels, registry, utils chung.
  - `configs/`: YAML cho pipeline/module/dataset/training/eval.
  - `scripts/`: preprocess/train/eval/infer/pack-results (CLI không tương tác).
  - `tests/`: unit tests cho từng module và pipeline nhỏ.
  - `docs/`: hướng dẫn, kiến trúc, sơ đồ pipeline, quickstart.
- **Interfaces bắt buộc** (đề xuất, dạng Python ABC):
  - `PerceptionModule.run(image, question, **kwargs) -> EvidenceBundle`
  - `ReasoningModule.run(evidence, question, **kwargs) -> ReasoningOutput`
  - `ConfirmationModule.run(question, candidate, evidence, **kwargs) -> ConfirmationOutput`
- **Orchestrator**: `VCTPPipeline` đọc `configs/` để ghép module; lưu full trace (bằng chứng, prompt, giả thuyết, điểm xác nhận) phục vụ debug/giải thích.

#### 2) Khả năng Tái tạo (Reproducibility)

- **Môi trường**: Cung cấp `requirements.txt` và/hoặc `environment.yml` (pin phiên bản), kèm hướng dẫn CUDA/cuDNN.
- **Hạt giống ngẫu nhiên**: Ấn định seed cho `python/random`, `numpy`, `torch` (và các lib liên quan) trong toàn pipeline.
- **Dữ liệu & Tiền xử lý**: Script không tương tác trong `scripts/` để tải, kiểm tra checksum, tiền xử lý (ví dụ: trích xuất CLIP/scene graph) và ghi cache theo phiên bản.
- **Cấu hình thí nghiệm**: Mọi tham số (model, prompt, hyperparam, dataset split) phải đến từ `configs/`. Tuyệt đối tránh hard-code.
- **Theo dõi & Lưu kết quả**: Lưu log chuẩn `logging`, CSV/JSONL kết quả, và tùy chọn tích hợp tracker (ví dụ: W&B) có thể bật/tắt qua config.
- **Bản công bố**: `docs/reproduce.md` mô tả chính xác lệnh tái tạo số liệu báo cáo (dataset, checkpoint, seed, config cụ thể).

#### 3) Khả năng Mở rộng (Extensibility)

- **Thiết kế hướng đối tượng + cấu hình**: Module mới chỉ cần kế thừa interface và đăng ký vào registry; chọn qua `configs/` mà không đổi pipeline.
- **Registry/Factory**: `vctp/core/registry.py` ánh xạ tên module (string trong YAML) → lớp triển khai.
- **Tách tài nguyên**: Adapter hóa LLM (OPT/LLaMA/…), detector (GroundingDINO/Detic/…), vector store/retriever; I/O thống nhất.
- **Kiến trúc cấu hình**: Dùng YAML + pydantic/omegaconf/hydra để validate và nhập cấu hình (mặc định + override theo thí nghiệm).
- **Ràng buộc phụ thuộc**: Module chỉ phụ thuộc `vctp/core` và thư viện ngoài cần thiết; tránh phụ thuộc chéo giữa `see/think/confirm`.

#### 4) Sự Rõ ràng và Dễ đọc (Clarity and Readability)

- **Chuẩn mã**: Tuân thủ PEP 8, dùng type hints đầy đủ cho public APIs; hạn chế `Any`, tránh try/except thừa.
- **Docstrings**: Bắt buộc cho lớp/hàm quan trọng theo chuẩn Google/NumPy; mô tả I/O, side effects, exceptions.
- **Tên & cấu trúc**: Tên biến/hàm mang nghĩa, tránh viết tắt; dùng guard clause, độ lồng ≤ 3 mức.
- **Logging & lỗi**: Dùng `logging` thay vì `print`; thông báo lỗi hữu ích và có hành động khắc phục.
- **Kiểm thử**: Tối thiểu có test cho từng interface và path chính của pipeline; test chạy nhanh, không cần GPU khi có thể (mock/stub).

### Đối tượng mục tiêu (Target Audience)

- **Nhà nghiên cứu AI/ML**: Muốn đọc nhanh kiến trúc, thay thế module, chạy benchmark và mở rộng ý tưởng.
- **Sinh viên/Học viên**: Cần tài liệu rõ ràng, lệnh tái tạo có thể chạy được, và ví dụ tối giản.
- **Tác giả tương lai (chính tôi)**: Dễ quay lại, hiểu logic ngay, chạy lại kết quả cũ và thử biến thể mới qua cấu hình.

### Kết nối với VisualCoT/KVQA

- Repo hiện thực hóa VisualCoT theo ba giai đoạn See–Think–Confirm; tương thích A-OKVQA/OKVQA.
- Bằng chứng (See) có thể gồm: đối tượng/thuộc tính/quan hệ, caption, CLIP features, scene graph, tri thức liên quan.
- Lập luận (Think) dùng CoT/prompt chiến lược (zero/few-shot, self-consistency, deliberate…) với adapter LLM.
- Xác nhận (Confirm) chấm điểm/lọc giả thuyết qua retrieval hoặc kiểm tra nhất quán; lưu giải thích giúp phân tích lỗi.

### Định nghĩa “Hoàn thành” (Definition of Done)

- Có `configs/`, `scripts/`, và hướng dẫn giúp tái tạo số liệu chính trên A-OKVQA/OKVQA bằng một lệnh mỗi bước.
- Các module See/Think/Confirm có interface ổn định, ví dụ triển khai tham chiếu, và test cơ bản.
- Log/kết quả được lưu có cấu trúc; tài liệu (docs) đủ để người mới chạy end-to-end trong vài bước rõ ràng.


