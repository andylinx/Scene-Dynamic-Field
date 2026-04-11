"""
download.py — Pre-download model weights from HuggingFace Hub.

Usage:
    python download.py --model Qwen2_5_VL --parameters 7B
    python download.py --model InternVL2_5 --parameters 2B

This caches weights locally so that benchmark.py runs without network access.
"""

import argparse
from transformers import AutoTokenizer, AutoModel, AutoProcessor

MODEL_IDS = {
    "Qwen2_VL":       {"7B": "Qwen/Qwen2-VL-7B-Instruct",   "72B": "Qwen/Qwen2-VL-72B-Instruct"},
    "Qwen2_5_VL":     {"3B": "Qwen/Qwen2.5-VL-3B-Instruct", "7B":  "Qwen/Qwen2.5-VL-7B-Instruct"},
    "InternVL2_5":    {"2B": "OpenGVLab/InternVL2_5-2B",     "8B":  "OpenGVLab/InternVL2_5-8B",
                       "26B": "OpenGVLab/InternVL2-26B"},
    "InternVideo2_5": {"default": "OpenGVLab/InternVideo2_5-Chat-8B"},
    "mPLUG_Owl3":     {"7B": "iic/mPLUG-Owl3-7B-241101",    "2B":  "iic/mPLUG-Owl3-2B-241101"},
}


def download_model(model_name: str, parameters: str) -> None:
    if model_name not in MODEL_IDS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_IDS)}")

    param_map = MODEL_IDS[model_name]
    model_id  = param_map.get(parameters) or param_map.get("default")
    if model_id is None:
        raise ValueError(f"Parameter size '{parameters}' not available for '{model_name}'. "
                         f"Options: {list(param_map)}")

    print(f"Downloading {model_name} ({parameters}) from '{model_id}' ...")
    AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    AutoModel.from_pretrained(model_id, trust_remote_code=True)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download model weights from HuggingFace Hub.")
    parser.add_argument("--model",      type=str, required=True,
                        choices=list(MODEL_IDS), help="Model to download.")
    parser.add_argument("--parameters", type=str, default="7B",
                        help="Parameter size variant (e.g. 7B, 2B).")
    args = parser.parse_args()
    download_model(args.model, args.parameters)
