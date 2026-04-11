# Scene Dynamic Field (SDF) — ICLR 2026

Official implementation of the paper **"Scene Dynamic Field"** (ICLR 2026).

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

1. Download the benchmark **video files** from Hugging Face:

```bash
pip install huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='andyc03/SDF-Videos', repo_type='dataset', local_dir='data/videos')"
```

   Alternatively, you can clone the dataset with Git LFS:

```bash
cd data
apt-get install git-lfs && git lfs install
git clone https://huggingface.co/datasets/andyc03/SDF-Videos videos
```

   The filenames and relative paths must match the `"path"` field in the JSON configs.
   
---

## Quick Start

### 1. (Optional) Pre-download model weights

```bash
python download.py --model Qwen2_5_VL --parameters 7B
python download.py --model InternVL2_5 --parameters 8B
```

### 2. Run evaluation

Run from the project root directory:

```bash
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

Results are automatically saved to `logs/` as JSON files.

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

## Citation

```bibtex
@article{li2026beyond,
  title={Beyond Static Vision: Scene Dynamic Field Unlocks Intuitive Physics Understanding in Multi-modal Large Language Models},
  author={Li, Nanxi and Wang, Xiang and Chen, Yuanjie and Zhang, Haode and Li, Hong and Li, Yong-Lu},
  journal={arXiv preprint arXiv:2604.03302},
  year={2026}
}
```
