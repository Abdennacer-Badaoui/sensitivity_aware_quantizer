import os
import json
import matplotlib.pyplot as plt
from layer_sensitivity_analyzer import LayerSensitivityAnalyzer
from model_utils import get_model_size_mb

def run_analysis_for_models(
    results_dir,
    models,
    calibration_data=None,
    eval_data=None,
    calibration_num_samples=100,
    eval_num_samples=100,
    batch_size=32,
    target_avg_bits=12.0,
    sensitivity_method="divergence",
):
    """
    Run sensitivity analysis for a list of models and save results.
    """
    all_results = {}
    for model_name in models:
        print(f"\n{'='*30}\nAnalyzing {model_name}\n{'='*30}")
        try:
            analyzer = LayerSensitivityAnalyzer(
                model_name=model_name,
                calibration_data=calibration_data,
                eval_data=eval_data,
                calibration_num_samples=calibration_num_samples,
                eval_num_samples=eval_num_samples,
                batch_size=batch_size,
            )

            # Run analysis
            results = analyzer.run_full_analysis(
                target_avg_bits=target_avg_bits, sensitivity_method=sensitivity_method
            )

            # Create results dictionary
            model_results = {
                "model_name": model_name,
                "original_perplexity": results.get("original_perplexity"),
                "quantized_perplexity": results.get("quantized_perplexity"),
                "original_model_size_mb": results.get("original_model_size_mb"),
                "quantized_model_size_mb": results.get("quantized_model_size_mb"),
                "sensitivity_scores": results.get("sensitivity_scores"),
                "mixed_precision_config": results.get("mixed_precision_config"),
                "bit_distribution": results.get("bit_distribution"),
                "target_avg_bits": target_avg_bits,
                "actual_avg_bits": results.get("actual_avg_bits"),
            }

            # Clean up quantized model from results
            if "quantized_model" in results:
                del results["quantized_model"]

            all_results[model_name] = model_results

            # Save individual results
            out_path = os.path.join(
                results_dir, f"{model_name.replace('/', '_')}_sensitivity.json"
            )
            save_json(model_results, out_path)

        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    # Save all results
    if all_results:
        save_json(all_results, os.path.join(results_dir, "all_models_sensitivity.json"))
    return all_results


def plot_comparisons(plots_dir, all_results, target_avg_bits, sensitivity_method):
    """
    Plot model size and perplexity comparisons for all analyzed models.
    """
    if not all_results:
        print("No results to plot!")
        return

    os.makedirs(plots_dir, exist_ok=True)
    model_names = list(all_results.keys())

    # Extract data with error checking
    orig_ppl = []
    quant_ppl = []
    orig_size = []
    quant_size = []

    for model_name in model_names:
        results = all_results[model_name]
        orig_ppl.append(results.get("original_perplexity", 0))
        quant_ppl.append(results.get("quantized_perplexity", 0))
        orig_size.append(results.get("original_model_size_mb", 0))
        quant_size.append(results.get("quantized_model_size_mb", 0) or 0)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    x = range(len(model_names))
    width = 0.35

    # Plot 1: Model sizes
    ax1.bar(
        [i - width / 2 for i in x],
        orig_size,
        width,
        label="Original Size (MB)",
        color="tab:blue",
        alpha=0.7,
    )
    ax1.bar(
        [i + width / 2 for i in x],
        quant_size,
        width,
        label="Quantized Size (MB)",
        color="tab:cyan",
        alpha=0.7,
    )
    ax1.set_ylabel("Model Size (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax1.set_title(f"Model Size Comparison (target avg bits: {target_avg_bits})")

    # Add size reduction percentage
    for i, (orig, quant) in enumerate(zip(orig_size, quant_size)):
        if orig > 0 and quant > 0:
            reduction = ((orig - quant) / orig) * 100
            ax1.text(
                i, max(orig, quant) + 1, f"{reduction:.1f}%", ha="center", va="bottom"
            )

    # Plot 2: Perplexity
    ax2.bar(
        [i - width / 2 for i in x],
        orig_ppl,
        width,
        label="Original Perplexity",
        color="tab:red",
        alpha=0.7,
    )
    ax2.bar(
        [i + width / 2 for i in x],
        quant_ppl,
        width,
        label="Quantized Perplexity",
        color="tab:orange",
        alpha=0.7,
    )
    ax2.set_ylabel("Perplexity")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.legend(loc="upper left")
    ax2.set_title(f"Model Perplexity Comparison (target avg bits: {target_avg_bits})")

    # Add perplexity degradation percentage
    for i, (orig, quant) in enumerate(zip(orig_ppl, quant_ppl)):
        if orig > 0 and quant > 0:
            degradation = (quant / orig - 1) * 100
            ax2.text(
                i,
                max(orig, quant) + 1,
                f"+{degradation:.1f}%",
                ha="center",
                va="bottom",
            )

    ax1.set_ylim(0, max(max(orig_size), max(quant_size)) * 1.2)
    ax2.set_ylim(0, max(max(orig_ppl), max(quant_ppl)) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"model_comparison_{target_avg_bits}_{sensitivity_method}.png"))
    plt.close()


def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)
