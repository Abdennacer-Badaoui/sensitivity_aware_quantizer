import os
import json
import matplotlib.pyplot as plt
from layer_sensitivity_analyzer import LayerSensitivityAnalyzer
from utils import get_model_size_mb, save_json

MODELS = [
    #"microsoft/DialoGPT-small",
    "gpt2",
    #"distilgpt2",
    # Add more model names as needed
]

RESULTS_DIR = "results"
PLOTS_DIR = "plots"

def run_analysis_for_models(models, target_avg_bits=12.0):
    all_results = {}
    for model_name in models:
        print(f"\n{'='*30}\nAnalyzing {model_name}\n{'='*30}")
        try:
            analyzer = LayerSensitivityAnalyzer(model_name=model_name)
            
            # Run analysis
            results = analyzer.run_full_analysis(target_avg_bits=target_avg_bits)
            
            # Create results dictionary
            model_results = {
                'original_perplexity': results.get('original_perplexity'),
                'quantized_perplexity': results.get('quantized_perplexity'),
                'original_model_size_mb': results.get('original_model_size_mb'),
                'quantized_model_size_mb': results.get('quantized_model_size_mb'),
                'sensitivity_scores': results.get('sensitivity_scores'),
                'mixed_precision_config': results.get('mixed_precision_config'),
                'bit_distribution': results.get('bit_distribution'),
                'actual_avg_bits': results.get('actual_avg_bits')
            }
            
            # Clean up quantized model from results
            if 'quantized_model' in results:
                del results['quantized_model']
            
            all_results[model_name] = model_results
            
            # Save individual results
            out_path = os.path.join(RESULTS_DIR, f"{model_name.replace('/', '_')}_sensitivity.json")
            save_json(model_results, out_path)
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    if all_results:
        save_json(all_results, os.path.join(RESULTS_DIR, "all_models_sensitivity.json"))
    return all_results

def plot_comparisons(all_results):
    if not all_results:
        print("No results to plot!")
        return
        
    os.makedirs(PLOTS_DIR, exist_ok=True)
    model_names = list(all_results.keys())
    
    # Extract data with error checking
    orig_ppl = []
    quant_ppl = []
    orig_size = []
    quant_size = []
    
    for model_name in model_names:
        results = all_results[model_name]
        orig_ppl.append(results.get('original_perplexity', 0))
        quant_ppl.append(results.get('quantized_perplexity', 0))
        orig_size.append(results.get('original_model_size_mb', 0))
        quant_size.append(results.get('quantized_model_size_mb', 0) or 0)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    x = range(len(model_names))
    width = 0.35
    
    # Plot 1: Model sizes
    ax1.bar([i - width/2 for i in x], orig_size, width, label="Original Size (MB)", color="tab:blue", alpha=0.7)
    ax1.bar([i + width/2 for i in x], quant_size, width, label="Quantized Size (MB)", color="tab:cyan", alpha=0.7)
    ax1.set_ylabel("Model Size (MB)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha="right")
    ax1.legend(loc="upper left")
    ax1.set_title("Model Size Comparison")
    
    # Add size reduction percentage
    for i, (orig, quant) in enumerate(zip(orig_size, quant_size)):
        if orig > 0 and quant > 0:
            reduction = ((orig - quant) / orig) * 100
            ax1.text(i, max(orig, quant) + 1, f"{reduction:.1f}%", ha='center', va='bottom')

    # Plot 2: Perplexity (using bars instead of lines)
    ax2.bar([i - width/2 for i in x], orig_ppl, width, label="Original Perplexity", color="tab:red", alpha=0.7)
    ax2.bar([i + width/2 for i in x], quant_ppl, width, label="Quantized Perplexity", color="tab:orange", alpha=0.7)
    ax2.set_ylabel("Perplexity")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=45, ha="right")
    ax2.legend(loc="upper left")
    ax2.set_title("Model Perplexity Comparison")
    
    # Add perplexity degradation percentage
    for i, (orig, quant) in enumerate(zip(orig_ppl, quant_ppl)):
        if orig > 0 and quant > 0:
            degradation = (quant / orig - 1) * 100
            ax2.text(i, max(orig, quant) + 1, f"+{degradation:.1f}%", ha='center', va='bottom')

    # Adjust y-axis limits to ensure bars are visible
    ax1.set_ylim(0, max(max(orig_size), max(quant_size)) * 1.2)
    ax2.set_ylim(0, max(max(orig_ppl), max(quant_ppl)) * 1.2)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"))
    plt.close()

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = run_analysis_for_models(MODELS, target_avg_bits=12.0)
    if all_results:
        plot_comparisons(all_results)
        print(f"\nAnalysis and visualizations complete. See {RESULTS_DIR} and {PLOTS_DIR}.")
    else:
        print("No results were generated. Check the errors above.")

if __name__ == "__main__":
    main()