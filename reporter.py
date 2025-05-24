import os
import json
from typing import Dict

RESULTS_DIR = "results"


def load_all_results(results_dir=RESULTS_DIR):
    all_results_path = os.path.join(results_dir, "all_models_sensitivity.json")
    if not os.path.exists(all_results_path):
        raise FileNotFoundError(
            f"{all_results_path} not found. Run quantizer.py first."
        )
    with open(all_results_path, "r") as f:
        all_results = json.load(f)
    return all_results


def print_summary(all_results: Dict):
    print("\nSummary of Model Sensitivity and Quantization Results:")
    print("-" * 80)
    for model_name, res in all_results.items():
        print(f"Model: {model_name}")
        print(f"  Original Perplexity:      {res.get('original_perplexity', 'N/A')}")
        print(f"  Quantized Perplexity:     {res.get('quantized_perplexity', 'N/A')}")
        print(f"  Original Model Size (MB): {res.get('original_model_size_mb', 'N/A')}")
        print(
            f"  Quantized Model Size (MB):{res.get('quantized_model_size_mb', 'N/A')}"
        )
        print(f"  Actual Avg Bits:          {res.get('actual_avg_bits', 'N/A')}")
        print(f"  Bit Distribution:         {res.get('bit_distribution', 'N/A')}")
        print("-")


def main():
    all_results = load_all_results()
    print_summary(all_results)


if __name__ == "__main__":
    main()
