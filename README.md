# VisualCoT for ViQVQA (See–Think–Confirm)

## 🚀 Quick Setup (One-time Installation)

This guide assumes you are working in the `~/workspace/VisualCoT-NLE` directory.

### 1. Activate Environment
```bash
source /home/research/my_envs/vllm_env/bin/activate
```
*Your prompt should change to `(vllm_env)`.*

### 2. Install vLLM
```bash
uv pip install vllm --torch-backend=auto
```

### 3. Install Dependencies

**VisualCoT-NLE Dependencies:**
```bash
uv add -r requirements.txt
```

**Describe-Anything (DAM) - External:**
```bash
cd external/describe-anything
pip install -v .
cd ../..
```

---

## 🛠️ Usage Guide (Runtime)

⚠️ **Quan trọng**: Cần mở **3 terminal riêng biệt** để chạy các server và pipeline.

---

### Terminal 1: Start vLLM Server (LLM API)

```bash
# Activate môi trường vLLM
source /home/research/my_envs/vllm_env/bin/activate

# Start vLLM với Vintern-1B (cho tiếng Việt)
vllm serve 5CD-AI/Vintern-1B-v3_5 \
    --port 1234 \
    --dtype auto \
    --gpu-memory-utilization 0.5 \
    --max-model-len 2048 \
    --trust-remote-code
```
> Server sẽ chạy tại `http://localhost:1234`

---

### Terminal 2: Start DAM Server (Describe-Anything Model)

```bash
# Activate môi trường vLLM (hoặc môi trường riêng nếu có)
source /home/research/my_envs/vllm_env/bin/activate

# Di chuyển vào thư mục DAM
cd ~/workspace/VisualCoT-NLE/external/describe-anything

# Start DAM server
python dam_server.py \
    --model-path nvidia/DAM-3B \
    --conv-mode v1 \
    --prompt-mode focal_prompt \
    --port 8000
```
> Server sẽ chạy tại `http://localhost:8000`

---

### Terminal 3: Run ViQVQA Pipeline

```bash
# Activate môi trường chính của project
source /home/research/my_envs/vllm_env/bin/activate

# Di chuyển vào thư mục project
cd ~/workspace/VisualCoT-NLE

# Chạy pipeline
python src/pipeline.py \
    --config configs/experiments/vivqax_baseline.yaml \
    --limit 300 \
    --output results/vivqax_results.json
```

---

## 📊 Kiểm tra kết quả

Sau khi chạy xong, kết quả được lưu tại:
```
results/vivqax_results.json
```

---

## 📂 Data Preparation

### Automated Download

#### 1. Download COCO 2014 Images (train + val)
```bash
bash scripts/download_data.sh
```
This will download and extract COCO 2014 images to `data/raw/coco/images/`.

#### 2. Download ViVQA-X Annotations (train + val + test)
```bash
python scripts/download_vivqax.py
```
This will download ViVQA-X annotations from Hugging Face to `data/raw/vivqa-x/annotations/`.

### Expected Directory Structure

After running both scripts, your data directory should look like:
```
data/
└── raw/
    ├── coco/
    │   └── images/
    │       ├── train2014/          # ~82,783 images
    │       └── val2014/            # ~40,504 images
    ├── vivqa-x/
    │   └── annotations/
    │       ├── train.json          # Training annotations
    │       ├── val.json            # Validation annotations
    │       └── test.json           # Test annotations
    └── scene-graph/                # Pre-computed scene graphs (if needed)
```


---

## 📄 Citation
[Visual Chain-of-Thought Prompting for Knowledge-based Visual Reasoning](https://arxiv.org/abs/2301.05226)
```bibtex
@article{chen2023see,
  title={Visual Chain-of-Thought Prompting for Knowledge-based Visual Reasoning},
  author={Chen, Zhenfang and Zhou, Qinhong and Shen, Yikang and Hong, Yining and Sun, Zhiqing and Gutfreund, Dan and Gan, Chuang},
  journal={arXiv preprint arXiv:2301.05226},
  year={2023}
}
```