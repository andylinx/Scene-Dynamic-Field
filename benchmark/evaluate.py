"""
evaluate.py — Compute accuracy from a saved benchmark results JSON.

Usage:
    python evaluate.py --results logs/benchmark_Qwen2_5_VL_NFS_8_<timestamp>.json
"""

import argparse
import json


def compute_accuracy(results_path: str) -> None:
    with open(results_path, "r") as f:
        data = json.load(f)

    metrics = data.get("metrics", {})
    total   = metrics.get("total_samples", 0)
    correct = metrics.get("correct_count", 0)

    if total == 0:
        print("No samples found in results file.")
        return

    accuracy = correct / total
    print(f"Model         : {data.get('model', 'N/A')}")
    print(f"Benchmark type: {data.get('benchmark_type', 'N/A')}")
    print(f"Input frames  : {data.get('input_frames', 'N/A')}")
    print(f"Parameters    : {data.get('parameter_size', 'N/A')}")
    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.4f}  ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute accuracy from a benchmark results JSON file.")
    parser.add_argument("--results", type=str, required=True,
                        help="Path to the benchmark results JSON file (saved in logs/ by benchmark.py).")
    args = parser.parse_args()
    compute_accuracy(args.results)
