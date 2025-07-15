# Reporting function for sensitivity-aware quantization analysis.


def run_full_analysis_report(
    sensitivity_scores,
    mode,
    mp_config,
    original_ppl,
    quantized_ppl,
    original_size,
    quantized_size,
):
    """Run complete analysis including sensitivity analysis, mixed precision configuration, and evaluation."""

    print("=" * 70)
    print("Summary of Layer Sensitivity Analysis")
    print("=" * 70)

    bit_counts = {}
    for layer_name, bits in mp_config.items():
        bit_counts[bits] = bit_counts.get(bits, 0) + 1

    print(f"\nBit Distribution:")
    for bits in sorted(bit_counts.keys(), reverse=True):
        print(f"  {bits:2d} bits: {bit_counts[bits]:2d} {'layers' if mode == 'per_layer' else 'blocks'}")

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
        "bit_distribution": bit_counts,
    }

    return results
