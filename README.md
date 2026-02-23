# Scene Dynamic Field (SDF) — ICLR 2026

Official implementation of the paper **"Scene Dynamic Field"** (ICLR 2026).

---

## Directory Structure

```
Scene-Dynamic-Field/
├── benchmark/                  # Benchmark evaluation
│   ├── models/                 # Model wrapper classes (one file per model)
│   │   ├── api.py              # Cloud API wrappers
│   │   ├── InternVL2_5.py
│   │   ├── InternVideo2_5.py
│   │   ├── llava_interleave.py
│   │   ├── minimax.py
│   │   ├── mPLUG_Owl3.py
│   │   ├── Qwen2_5.py
│   │   ├── Qwen2_5_VL.py
│   │   ├── Qwen2_VL.py
│   │   └── VideoChat2.py
│   ├── data/                   # Benchmark configuration files
│   │   ├── NFS_stride_2.json   # Next-frame selection, stride-2 split
│   │   ├── NFS_stride_4.json   # Next-frame selection, stride-4 split
│   │   ├── TCV_stride_2.json   # Temporal consistency verification, stride-2
│   │   ├── TCV_stride_4.json   # Temporal consistency verification, stride-4
│   │   ├── pics/               # ← Place benchmark images here
│   │   └── videos/             # ← Place benchmark videos here
│   ├── benchmark.py            # Main evaluation script
│   ├── dataset.py              # VideoDataset / VideoDataset_TCV classes
│   ├── logger.py               # BenchmarkLogger — saves per-sample results to JSON
│   ├── evaluate.py             # Compute accuracy from a saved results file
│   └── download.py             # Pre-download model weights from HuggingFace
└── README.md
```

---

## Prerequisites

Install dependencies based on your GPU environment:

```bash
pip install torch torchvision transformers decord opencv-python tqdm
# Model-specific extras
pip install qwen-vl-utils          # Qwen2-VL / Qwen2.5-VL
pip install modelscope             # mPLUG-Owl3
pip install openai                 # MiniMax API
```

---

## Data Setup

1. Place benchmark **video files** in `benchmark/data/videos/`.  
   The filenames and relative paths must match the `"path"` field in the JSON configs.
2. If your dataset uses still images, place them in `benchmark/data/pics/`.

---

## Quick Start

### 1. (Optional) Pre-download model weights

```bash
cd benchmark
python download.py --model Qwen2_5_VL --parameters 7B
python download.py --model InternVL2_5 --parameters 8B
```

### 2. Run evaluation

Run from the `benchmark/` directory:

```bash
cd benchmark

# Next-frame selection benchmark with Qwen2.5-VL 7B, stride-4 split
python benchmark.py \
    --model Qwen2_5_VL \
    --type NFS \
    --json_file data/NFS_stride_4.json \
    --input_frames 8 \
    --parameters 7B

# Temporal consistency verification with InternVL2.5 8B
python benchmark.py \
    --model InternVL2_5 \
    --type TCV \
    --json_file data/TCV_stride_4.json \
    --input_frames 8 \
    --parameters 8B
```

Results are automatically saved to `benchmark/logs/` as JSON files.

### 3. Compute accuracy from a saved results file

```bash
python evaluate.py --results logs/benchmark_Qwen2_5_VL_NFS_8_<timestamp>.json
```

### 4. Using cloud API models

Set the `siliconflow_token` environment variable before running:

```bash
export siliconflow_token=<your_siliconflow_api_key>
python benchmark.py --model Qwen2_VL_api --type NFS --json_file data/NFS_stride_4.json
```

For MiniMax, pass your API key directly in `benchmark.py` (`MiniMax("your_api_key")`).

---

## Supported Models

| Model | Task | Size variants |
|---|---|---|
| Qwen2-VL | NFS / TCV | 7B, 72B |
| Qwen2.5-VL | NFS / TCV | 3B, 7B |
| Qwen2.5 (text-only) | NFS / TCV | 7B, 72B |
| InternVL2.5 | NFS / TCV | 2B, 8B, 26B |
| InternVideo2.5 | NFS / TCV | 8B |
| mPLUG-Owl3 | NFS / TCV | 2B, 7B |
| VideoChat2 | NFS / TCV | — |
| LLaVA-Interleave | NFS / TCV | — |
| MiniMax (API) | NFS / TCV | — |
| Qwen2-VL (API) | NFS / TCV | 72B |
| InternVL2 (API) | NFS / TCV | 26B |
---

```

