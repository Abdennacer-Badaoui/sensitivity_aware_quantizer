import os
import json
import matplotlib.pyplot as plt
from layer_sensitivity_analyzer import LayerSensitivityAnalyzer


def run_analysis_for_models(
    results_dir,
    models,
    calibration_data=None,
    eval_data=None,
    calibration_num_samples=100,
    eval_num_samples=100,
    batch_size=32,
    sensitivity_method="divergence",
    config_strategy="adaptive_threshold",
    use_iterative=False,
    max_perplexity_increase=0.1,
    layers_per_iteration=3,
    max_iterations=50,
):
    """Run sensitivity analysis for a list of models and save results."""
    print(f"\n{'<'*150}\n{' '*50}Mixed Precision Quantization Analysis\n{'>'*150}")
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*60}\nAnalyzing {model_name}\n{'='*60}")
        try:
            analyzer = LayerSensitivityAnalyzer(
                model_name=model_name,
                calibration_data=calibration_data,
                eval_data=eval_data,
                calibration_num_samples=calibration_num_samples,
                eval_num_samples=eval_num_samples,
                batch_size=batch_size,
                sensitivity_method=sensitivity_method,
                config_strategy=config_strategy,
                use_iterative=use_iterative,
                max_perplexity_increase=max_perplexity_increase,
                layers_per_iteration=layers_per_iteration,
                max_iterations=max_iterations,
            )

            results = analyzer.run_full_analysis()

            model_results = {
                "model_name": model_name,
                "sensitivity_method": sensitivity_method,
                "calibration_num_samples": calibration_num_samples,
                "config_strategy": config_strategy,
                "use_iterative": use_iterative,
                "layers_per_iteration": layers_per_iteration,
                "max_iterations": max_iterations,
                "max_perplexity_increase": max_perplexity_increase,
                "original_perplexity": results.get("original_perplexity"),
                "quantized_perplexity": results.get("quantized_perplexity"),
                "original_model_size_mb": results.get("original_model_size_mb"),
                "quantized_model_size_mb": results.get("quantized_model_size_mb"),
                "sensitivity_scores": results.get("sensitivity_scores"),
                "mixed_precision_config": results.get("mixed_precision_config"),
                "benchmark_results": results.get("benchmark_results", {}),
            }

            # Clean up quantized model from results
            if "quantized_model" in results:
                del results["quantized_model"]

            all_results[model_name] = model_results

            # Save individual results
            out_path = os.path.join(
                results_dir,
                f"{model_name.replace('/', '_')}_{sensitivity_method}_{calibration_num_samples}_{config_strategy}_{use_iterative}.json",
            )
            save_json(model_results, out_path)

        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    return all_results


def plot_comparisons(
    plots_dir,
    all_results,
    sensitivity_method,
    calibration_num_samples,
    config_strategy,
    use_iterative,
    max_perplexity_increase,
):
    """Plot model size, MMLU accuracy, and perplexity comparisons."""
    if not all_results:
        print("No results to plot!")
        return

    os.makedirs(plots_dir, exist_ok=True)
    model_names = list(all_results.keys())

    # Create figure with subplots for model size, MMLU accuracy, and perplexity
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    width = 0.35
    x = range(len(model_names))

    # Extract data
    orig_size = []
    quant_size = []
    orig_mmlu = []
    quant_mmlu = []
    orig_perplexity = []
    quant_perplexity = []

    for model_name in model_names:
        results = all_results[model_name]
        orig_size.append(results.get("original_model_size_mb", 0))
        quant_size.append(results.get("quantized_model_size_mb", 0) or 0)
        
        # Get MMLU accuracies
        if "benchmark_results" in results:
            orig_acc = results["benchmark_results"]["original"]["mean_accuracy"] if results["benchmark_results"]["original"] else 0
            quant_acc = results["benchmark_results"]["quantized"]["mean_accuracy"] if results["benchmark_results"]["quantized"] else 0
            orig_mmlu.append(orig_acc * 100)  # Convert to percentage
            quant_mmlu.append(quant_acc * 100)  # Convert to percentage
        else:
            orig_mmlu.append(0)
            quant_mmlu.append(0)
            
        # Get perplexity scores
        orig_perplexity.append(results.get("original_perplexity", 0))
        quant_perplexity.append(results.get("quantized_perplexity", 0))

    # Plot 1: Model sizes
    ax = axes[0]
    ax.bar(
        [i - width / 2 for i in x],
        orig_size,
        width,
        label="Original Size",
        color="tab:blue",
        alpha=0.7,
    )
    ax.bar(
        [i + width / 2 for i in x],
        quant_size,
        width,
        label="Quantized Size",
        color="tab:cyan",
        alpha=0.7,
    )
    ax.set_ylabel("Model Size (MB)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    max_y = max(max(orig_size), max(quant_size)) * 1.2
    for i, (orig, quant) in enumerate(zip(orig_size, quant_size)):
        if orig > 0 and quant > 0:
            reduction = ((orig - quant) / orig) * 100
            ax.text(
                i,
                max(orig, quant) + max_y * 0.05,
                f"↓{reduction:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
            )

    ax.set_ylim(0, max_y)
    ax.legend(loc="upper right")
    ax.set_title("Model Size Comparison")

    # Plot 2: MMLU Accuracy
    ax = axes[1]
    ax.bar(
        [i - width / 2 for i in x],
        orig_mmlu,
        width,
        label="Original MMLU",
        color="tab:green",
        alpha=0.7,
    )
    ax.bar(
        [i + width / 2 for i in x],
        quant_mmlu,
        width,
        label="Quantized MMLU",
        color="tab:olive",
        alpha=0.7,
    )
    ax.set_ylabel("MMLU Accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    max_y_mmlu = max(max(orig_mmlu), max(quant_mmlu)) * 1.2
    for i, (orig, quant) in enumerate(zip(orig_mmlu, quant_mmlu)):
        if orig > 0 and quant > 0:
            change = quant - orig
            ax.text(
                i,
                max(orig, quant) + max_y_mmlu * 0.05,
                f"{'↑' if change >= 0 else '↓'}{abs(change):.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
                color="green" if change >= 0 else "red",
            )

    ax.set_ylim(0, max_y_mmlu)
    ax.legend(loc="upper right")
    ax.set_title("MMLU Accuracy Comparison")

    # Plot 3: Perplexity
    ax = axes[2]
    ax.bar(
        [i - width / 2 for i in x],
        orig_perplexity,
        width,
        label="Original Perplexity",
        color="tab:red",
        alpha=0.7,
    )
    ax.bar(
        [i + width / 2 for i in x],
        quant_perplexity,
        width,
        label="Quantized Perplexity",
        color="tab:orange",
        alpha=0.7,
    )
    ax.set_ylabel("Perplexity")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")

    max_y_ppl = max(max(orig_perplexity), max(quant_perplexity)) * 1.2
    for i, (orig, quant) in enumerate(zip(orig_perplexity, quant_perplexity)):
        if orig > 0 and quant > 0:
            change = ((quant - orig) / orig) * 100
            ax.text(
                i,
                max(orig, quant) + max_y_ppl * 0.05,
                f"{'↑' if change >= 0 else '↓'}{abs(change):.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                weight="bold",
                color="red" if change > 0 else "green",
            )

    ax.set_ylim(0, max_y_ppl)
    ax.legend(loc="upper right")
    ax.set_title("Perplexity Comparison")

    # Add overall title
    fig.suptitle(
        f"Model Analysis Results\n"
        f"Method: {sensitivity_method} with {calibration_num_samples} samples | Config Strategy: {config_strategy} | "
        f"Iterative: {use_iterative} | Max PPL Increase: {max_perplexity_increase}",
        fontsize=12,
        y=1.05,
    )

    plt.tight_layout()
    
    # Save plot
    filename = f"comprehensive_analysis_{sensitivity_method}_{config_strategy}_{use_iterative}.png"
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\nComprehensive analysis plot saved to {os.path.join(plots_dir, filename)}")


def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """Load data from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)
