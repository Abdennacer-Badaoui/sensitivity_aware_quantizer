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


def run_full_analysis_report(
    sensitivity_scores,
    mp_config,
    original_ppl,
    quantized_ppl,
    original_size,
    quantized_size,
    target_avg_bits,
    metric,
):
    """Run complete analysis including sensitivity analysis, mixed precision configuration, and evaluation."""
    print("=" * 70)
    print("LAYER-WISE SENSITIVITY ANALYSIS")
    print("=" * 70)
    print(f"\nLayer Sensitivity Scores ({metric}, higher = more sensitive):")
    print("-" * 70)
    sorted_scores = sorted(sensitivity_scores.items(), key=lambda x: x[1], reverse=True)
    for layer_name, score in sorted_scores:
        print(f"{layer_name:<45} {score:.8f}")
    print(f"\nMixed Precision Configuration (target: {target_avg_bits:.1f} bits avg):")
    print("-" * 70)
    bit_counts = {}
    for layer_name, bits in mp_config.items():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1
        sensitivity = sensitivity_scores[layer_name]
        print(f"{layer_name:<45} {bits:2d} bits (sensitivity: {sensitivity:.8f})")
    print(f"\nBit Distribution:")
    for bits in sorted(bit_counts.keys(), reverse=True):
        print(f"  {bits:2d} bits: {bit_counts[bits]:2d} layers")
    actual_avg_bits = sum(mp_config.values()) / len(mp_config)
    print(f"  Actual average: {actual_avg_bits:.2f} bits")
    print(f"\nModel Evaluation:")
    print("-" * 70)
    try:
        print(f"Original Model Size:            {original_size:.2f} MB")
        print(f"Original Model Perplexity:      {original_ppl:.4f}")
    except Exception as e:
        print(f"Error computing original model metrics: {e}")
        original_ppl = float("inf")
        original_size = 0
        print(f"Original Model Size:            Failed to compute")
        print(f"Original Model Perplexity:      Failed to compute")
    try:
        print(f"Quantized Model Size:           {quantized_size:.2f} MB")
        print(f"Mixed Precision Perplexity:     {quantized_ppl:.4f}")
        if original_ppl != float("inf") and quantized_ppl != float("inf"):
            degradation = quantized_ppl - original_ppl
            rel_degradation = (quantized_ppl / original_ppl - 1) * 100
            size_reduction = ((original_size - quantized_size) / original_size) * 100
            print(f"Perplexity Degradation:         {degradation:.4f}")
            print(f"Relative Degradation:           {rel_degradation:.2f}%")
            print(f"Size Reduction:                 {size_reduction:.2f}%")
        else:
            print(f"Degradation analysis:           Not available")
    except Exception as e:
        print(f"Error computing quantized model metrics: {e}")
        quantized_ppl = float("inf")
        quantized_size = 0
        print(f"Quantized Model Size:           Failed to compute")
        print(f"Mixed Precision Perplexity:     Failed to compute")
    results = {
        "sensitivity_scores": sensitivity_scores,
        "mixed_precision_config": mp_config,
        "original_perplexity": original_ppl if original_ppl != float("inf") else None,
        "quantized_perplexity": quantized_ppl
        if quantized_ppl != float("inf")
        else None,
        "original_model_size_mb": original_size,
        "quantized_model_size_mb": quantized_size,
        "target_avg_bits": target_avg_bits,
        "actual_avg_bits": actual_avg_bits,
        "bit_distribution": bit_counts,
    }
    return results


def main():
    all_results = load_all_results()
    print_summary(all_results)


if __name__ == "__main__":
    main()
