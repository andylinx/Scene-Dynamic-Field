import json
import os
from datetime import datetime
from typing import Dict, Any
import logging

class BenchmarkLogger():
    def __init__(self, model_name: str, benchmark_type: str, input_frames: int, json_file: str, parameter_size: str = "7B"):
        self.results = {
            "model": model_name,
            "parameter_size": parameter_size,
            "benchmark_type": benchmark_type,
            "input_frames": input_frames,
            "json_file": json_file,
            "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            "samples": [],
            "metrics": {
                "total_samples": 0,
                "correct_count": 0,
                "accuracy": 0.0
            }
        }
        self.logger = logging.getLogger(__name__)
        self.info = self.logger.info
        self.warning = self.logger.warning
        
        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)
        
        # Create filename based on parameters
        self.filename = f"logs/benchmark_{model_name}_{benchmark_type}_{input_frames}_{self.results['timestamp']}.json"

    def log_sample(self, sample_idx: int, prediction: str, target: str, is_correct: bool, response: str = None):
        """Log individual sample results"""
        sample_result = {
            "sample_idx": sample_idx,
            "prediction": prediction,
            "target": target,
            "is_correct": is_correct
        }
        
        if response != None:
            sample_result["response"] = response
            
        self.results["samples"].append(sample_result)
        
        # Update metrics
        self.results["metrics"]["total_samples"] += 1
        if is_correct:
            self.results["metrics"]["correct_count"] += 1
        self.results["metrics"]["accuracy"] = (
            self.results["metrics"]["correct_count"] / 
            self.results["metrics"]["total_samples"]
        )

    def save(self):
        """Save results to JSON file"""
        with open(self.filename, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to {self.filename}")

    def print_summary(self):
        """Print summary of results"""
        print("\n=== Benchmark Summary ===")
        print(f"Model: {self.results['model']}")
        print(f"Parameter Size: {self.results['parameter_size']}")
        print(f"Benchmark Type: {self.results['benchmark_type']}")
        print(f"Input Frames: {self.results['input_frames']}")
        print(f"JSON File: {self.results['json_file']}")
        print(f"Total Samples: {self.results['metrics']['total_samples']}")
        print(f"Correct Predictions: {self.results['metrics']['correct_count']}")
        print(f"Accuracy: {self.results['metrics']['accuracy']:.4f}") 