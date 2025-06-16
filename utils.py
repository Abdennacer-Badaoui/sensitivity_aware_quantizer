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
    config_strategy="aggressive",
    use_iterative=False,
    max_perplexity_increase=0.1,
    layers_per_iteration=3,
    max_iterations=50,
):
    """
    Run sensitivity analysis for a list of models and save results.
    """
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
            )

            # Run analysis
            results = analyzer.run_full_analysis(
                sensitivity_method=sensitivity_method,
                config_strategy=config_strategy,
                use_iterative=use_iterative,
                max_perplexity_increase=max_perplexity_increase,
                layers_per_iteration=layers_per_iteration,
                max_iterations=max_iterations,
            )

            # Create results dictionary
            model_results = {
                "model_name": model_name,
                "sensitivity_method": sensitivity_method,
                "config_strategy": config_strategy,
                "use_iterative": use_iterative,
                "max_perplexity_increase": max_perplexity_increase,
                "original_perplexity": results.get("original_perplexity"),
                "quantized_perplexity": results.get("quantized_perplexity"),
                "original_model_size_mb": results.get("original_model_size_mb"),
                "quantized_model_size_mb": results.get("quantized_model_size_mb"),
                "sensitivity_scores": results.get("sensitivity_scores"),
                "mixed_precision_config": results.get("mixed_precision_config"),
                "bit_distribution": results.get("bit_distribution"),
                "benchmark_results": results.get("benchmark_results", {}),  # Include benchmark results
            }

            # Clean up quantized model from results
            if "quantized_model" in results:
                del results["quantized_model"]

            all_results[model_name] = model_results

            # Save individual results
            out_path = os.path.join(
                results_dir,
                f"{model_name.replace('/', '_')}_{sensitivity_method}_{config_strategy}_{use_iterative}.json",
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


def get_benchmark_tasks_and_metrics(all_results):
    """Extract all unique benchmark tasks and their metrics from results."""
    tasks_metrics = {}
    
    for model_results in all_results.values():
        if "benchmark_results" in model_results:
            for model_type in ["original", "quantized"]:
                if model_type in model_results["benchmark_results"]:
                    for task_name, task_results in model_results["benchmark_results"][model_type].items():
                        if task_name not in tasks_metrics:
                            tasks_metrics[task_name] = set()
                        
                        # Collect all metric names for this task
                        for metric_name in task_results.keys():
                            if isinstance(task_results[metric_name], (int, float)):
                                tasks_metrics[task_name].add(metric_name)
    
    # Convert sets to lists for easier handling
    return {task: list(metrics) for task, metrics in tasks_metrics.items()}


def get_metric_display_info(metric_name):
    """Get display information for different metric types."""
    metric_name_lower = metric_name.lower()
    
    # Define metric categories and their properties
    accuracy_metrics = ['accuracy', 'acc', 'exact_match', 'em', 'correct', 'score']
    loss_metrics = ['loss', 'perplexity', 'ppl', 'error', 'mse', 'rmse']
    f1_metrics = ['f1', 'f1_score', 'macro_f1', 'micro_f1']
    bleu_metrics = ['bleu', 'bleu_score', 'rouge']
    correlation_metrics = ['matthews_correlation', 'mcc', 'pearson', 'spearman', 'correlation', 'corr']
    
    if any(corr_metric in metric_name_lower for corr_metric in correlation_metrics):
        return {
            'ylabel': 'Correlation Coefficient',
            'title_suffix': 'Correlation',
            'multiplier': 1,
            'format': '.3f',
            'higher_better': True,
            'range': (-1, 1),  # Special handling for -1 to 1 range
            'zero_centered': True
        }
    elif any(acc_metric in metric_name_lower for acc_metric in accuracy_metrics):
        return {
            'ylabel': 'Accuracy (%)',
            'title_suffix': 'Accuracy',
            'multiplier': 100 if 'accuracy' in metric_name_lower and metric_name_lower != 'accuracy' else 1,
            'format': '.1f',
            'higher_better': True,
            'range': None,
            'zero_centered': False
        }
    elif any(loss_metric in metric_name_lower for loss_metric in loss_metrics):
        return {
            'ylabel': metric_name.replace('_', ' ').title(),
            'title_suffix': metric_name.replace('_', ' ').title(),
            'multiplier': 1,
            'format': '.3f',
            'higher_better': False,
            'range': None,
            'zero_centered': False
        }
    elif any(f1_metric in metric_name_lower for f1_metric in f1_metrics):
        return {
            'ylabel': 'F1 Score',
            'title_suffix': 'F1 Score',
            'multiplier': 100,
            'format': '.1f',
            'higher_better': True,
            'range': None,
            'zero_centered': False
        }
    elif any(bleu_metric in metric_name_lower for bleu_metric in bleu_metrics):
        return {
            'ylabel': metric_name.upper() + ' Score',
            'title_suffix': metric_name.upper() + ' Score',
            'multiplier': 100,
            'format': '.1f',
            'higher_better': True,
            'range': None,
            'zero_centered': False
        }
    else:
        return {
            'ylabel': metric_name.replace('_', ' ').title(),
            'title_suffix': metric_name.replace('_', ' ').title(),
            'multiplier': 1,
            'format': '.3f',
            'higher_better': True,
            'range': None,
            'zero_centered': False
        }


def plot_comparisons(
    plots_dir,
    all_results,
    sensitivity_method,
    config_strategy,
    use_iterative,
    max_perplexity_increase,
):
    """
    Plot model size, perplexity, and individual benchmark task comparisons for all analyzed models.
    Enhanced to handle correlation metrics with -1 to 1 range.
    """
    if not all_results:
        print("No results to plot!")
        return

    os.makedirs(plots_dir, exist_ok=True)
    model_names = list(all_results.keys())

    # Get all benchmark tasks and their metrics
    tasks_metrics = get_benchmark_tasks_and_metrics(all_results)
    
    # Calculate total number of plots needed
    num_benchmark_plots = sum(len(metrics) for metrics in tasks_metrics.values())
    total_plots = 2 + num_benchmark_plots  # Size + Perplexity + Benchmark plots
    
    # Create figure with dynamic subplot layout
    if total_plots <= 6:
        cols = 2
        rows = (total_plots + 1) // 2
    else:
        cols = 3
        rows = (total_plots + 2) // 3
    
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows))
    
    # Handle case where we have only one row or column
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    x = range(len(model_names))
    width = 0.35
    plot_idx = 0

    # Extract basic data with error checking
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

    # Plot 1: Model sizes
    ax = axes[plot_idx]
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
    
    # Calculate max y-value for this plot
    max_y = max(max(orig_size), max(quant_size)) * 1.2
    
    # Add size reduction percentage
    for i, (orig, quant) in enumerate(zip(orig_size, quant_size)):
        if orig > 0 and quant > 0:
            reduction = ((orig - quant) / orig) * 100
            ax.text(
                i, max(orig, quant) + max_y * 0.05, f"↓{reduction:.1f}%", 
                ha="center", va="bottom", fontsize=9, weight='bold'
            )
    
    # Set ylim after adding text to ensure everything fits
    ax.set_ylim(0, max_y)
    
    # Place legend outside the plot if there's overlap
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.set_title("Model Size Comparison")
    plot_idx += 1

    # Plot 2: Perplexity
    ax = axes[plot_idx]
    ax.bar(
        [i - width / 2 for i in x],
        orig_ppl,
        width,
        label="Original Perplexity",
        color="tab:red",
        alpha=0.7,
    )
    ax.bar(
        [i + width / 2 for i in x],
        quant_ppl,
        width,
        label="Quantized Perplexity",
        color="tab:orange",
        alpha=0.7,
    )
    ax.set_ylabel("Perplexity")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    
    # Calculate max y-value for this plot
    max_y_ppl = max(max(orig_ppl), max(quant_ppl)) * 1.2
    
    # Add perplexity degradation percentage
    for i, (orig, quant) in enumerate(zip(orig_ppl, quant_ppl)):
        if orig > 0 and quant > 0:
            degradation = (quant / orig - 1) * 100
            ax.text(
                i,
                max(orig, quant) + max_y_ppl * 0.05,
                f"↑{degradation:.1f}%",
                ha="center",
                va="bottom",
                fontsize=9,
                weight='bold'
            )
    
    # Set ylim after adding text to ensure everything fits
    ax.set_ylim(0, max_y_ppl)
    
    # Place legend outside the plot if there's overlap
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    ax.set_title("Perplexity Comparison")
    plot_idx += 1

    # Plot individual benchmark tasks and metrics
    # Use consistent colors for all benchmark plots
    benchmark_colors = {
        'original': 'tab:green',
        'quantized': 'tab:olive'
    }
    
    for task_idx, (task_name, metrics) in enumerate(tasks_metrics.items()):
        for metric_idx, metric_name in enumerate(metrics):
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            
            # Extract data for this specific task and metric
            orig_scores = []
            quant_scores = []
            
            for model_name in model_names:
                results = all_results[model_name]
                
                # Get original score
                orig_score = 0
                if ("benchmark_results" in results and 
                    "original" in results["benchmark_results"] and
                    task_name in results["benchmark_results"]["original"] and
                    metric_name in results["benchmark_results"]["original"][task_name]):
                    orig_score = results["benchmark_results"]["original"][task_name][metric_name]
                orig_scores.append(orig_score)
                
                # Get quantized score
                quant_score = 0
                if ("benchmark_results" in results and 
                    "quantized" in results["benchmark_results"] and
                    task_name in results["benchmark_results"]["quantized"] and
                    metric_name in results["benchmark_results"]["quantized"][task_name]):
                    quant_score = results["benchmark_results"]["quantized"][task_name][metric_name]
                quant_scores.append(quant_score)
            
            # Get metric display information
            metric_info = get_metric_display_info(metric_name)
            
            # Apply multiplier for percentage metrics
            orig_scores_display = [score * metric_info['multiplier'] for score in orig_scores]
            quant_scores_display = [score * metric_info['multiplier'] for score in quant_scores]
            
            # Plot with consistent colors
            ax.bar(
                [i - width / 2 for i in x],
                orig_scores_display,
                width,
                label="Original",
                color=benchmark_colors['original'],
                alpha=0.7,
            )
            ax.bar(
                [i + width / 2 for i in x],
                quant_scores_display,
                width,
                label="Quantized",
                color=benchmark_colors['quantized'],
                alpha=0.7,
            )
            
            ax.set_ylabel(metric_info['ylabel'])
            ax.set_xticks(x)
            ax.set_xticklabels(model_names, rotation=45, ha="right")
            
            # Create title
            task_display = task_name.replace('_', ' ').title()
            ax.set_title(f"{task_display}: {metric_info['title_suffix']}")
            
            # Calculate max y-value for this plot
            if metric_info.get('zero_centered', False):
                max_abs_val = max(abs(min(min(orig_scores_display), min(quant_scores_display))),
                                 abs(max(max(orig_scores_display), max(quant_scores_display))))
                max_y_bench = min(1.2 * max_abs_val, 1.0)  # Don't exceed -1 to 1 range
                min_y_bench = -max_y_bench
            else:
                max_y_bench = max(max(orig_scores_display), max(quant_scores_display)) * 1.2
                min_y_bench = 0
            
            # Add performance change percentage
            for i, (orig, quant) in enumerate(zip(orig_scores, quant_scores)):
                if orig != 0 and quant != 0:  # Changed condition to handle negative values
                    # For correlation metrics, calculate absolute difference instead of percentage
                    if metric_info.get('zero_centered', False):
                        change = quant - orig  # Absolute difference
                        
                        # Choose arrow and color based on change direction
                        arrow = "↑" if change >= 0 else "↓"
                        color = "green" if change >= 0 else "red"
                        
                        # Position text above the highest bar
                        text_y = max(orig_scores_display[i], quant_scores_display[i]) + (max_y_bench - min_y_bench) * 0.05
                        
                        ax.text(
                            i,
                            text_y,
                            f"{arrow}{abs(change):.3f}",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            weight='bold',
                            color=color
                        )
                    else:
                        # Regular percentage calculation for other metrics
                        change = ((quant / orig - 1) * 100)
                        
                        # Choose arrow based on whether higher is better
                        if metric_info['higher_better']:
                            arrow = "↑" if change >= 0 else "↓"
                            color = "green" if change >= 0 else "red"
                        else:
                            arrow = "↓" if change >= 0 else "↑"
                            color = "red" if change >= 0 else "green"
                        
                        ax.text(
                            i,
                            max(orig_scores_display[i], quant_scores_display[i]) + max_y_bench * 0.05,
                            f"{arrow}{abs(change):.1f}%",
                            ha="center",
                            va="bottom",
                            fontsize=9,
                            weight='bold',
                            color=color
                        )
            
            # Set y-axis limits with special handling for correlation metrics
            if metric_info.get('zero_centered', False):
                ax.set_ylim(min_y_bench, max_y_bench)
                # Add horizontal line at y=0 for reference
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
            else:
                ax.set_ylim(min_y_bench, max_y_bench)
            
            # Place legend outside the plot if there's overlap
            ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            
            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    # Add overall title
    fig.suptitle(
        f"Model Analysis Results\n"
        f"Sensitivity Method: {sensitivity_method} | Config Strategy: {config_strategy} | "
        f"Iterative: {use_iterative} | Max PPL Increase: {max_perplexity_increase}",
        fontsize=16,
        y=0.98
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.93, right=0.85)  # Make room for suptitle and legends
    
    # Save plot
    filename = f"comprehensive_analysis_{sensitivity_method}_{config_strategy}_{use_iterative}.png"
    plt.savefig(os.path.join(plots_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComprehensive analysis plot saved to {os.path.join(plots_dir, filename)}")

def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)