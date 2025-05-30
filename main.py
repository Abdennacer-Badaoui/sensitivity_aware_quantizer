import os
import matplotlib.pyplot as plt
import argparse
from layer_sensitivity_analyzer import LayerSensitivityAnalyzer
from utils import run_analysis_for_models, plot_comparisons

MODELS = [
    # "microsoft/DialoGPT-small",
    # "gpt2",
    "distilbert/distilgpt2",
    "EleutherAI/gpt-neo-125M",
    "facebook/opt-125M",
    "facebook/opt-355M",
]

RESULTS_DIR = "results"
PLOTS_DIR = "plots"


def parse_args():
    parser = argparse.ArgumentParser(description="Model Quantization Analysis Tool")

    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        help="List of model names to analyze",
    )

    # Data configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Dataset name for calibration and evaluation",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="wikitext-2-raw-v1",
        help="Dataset configuration",
    )
    parser.add_argument(
        "--calibration_samples",
        type=int,
        default=100,
        help="Number of samples for calibration",
    )
    parser.add_argument(
        "--eval_samples", type=int, default=100, help="Number of samples for evaluation"
    )

    # Quantization parameters
    parser.add_argument(
        "--target_bits",
        type=float,
        default=12.0,
        help="Target average bits for mixed precision quantization",
    )

    # Sensitivity analysis methods
    parser.add_argument(
        "--sensitivity_method",
        type=str,
        default="divergence",
        choices=["divergence", "hessian"],
        help="Method for sensitivity analysis",
    )

    # Processing parameters
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for processing"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to use (auto, cuda, or cpu)"
    )

    # Output configuration
    parser.add_argument(
        "--results_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--plots_dir", type=str, default="plots", help="Directory to save plots"
    )

    return parser.parse_args()



def main():
    args = parse_args()

    # Create output directories
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)

    # Run analysis
    all_results = run_analysis_for_models(
        results_dir=RESULTS_DIR,
        models=args.models,
        calibration_data=None,
        eval_data=None,
        calibration_num_samples=args.calibration_samples,
        eval_num_samples=args.eval_samples,
        batch_size=args.batch_size,
        target_avg_bits=args.target_bits,
        sensitivity_method=args.sensitivity_method,
    )

    if all_results:
        plot_comparisons(PLOTS_DIR, all_results, args.target_bits, args.sensitivity_method)
        print(
            f"\nAnalysis and visualizations complete. See {args.results_dir} and {args.plots_dir}."
        )
    else:
        print("No results were generated. Check the errors above.")


if __name__ == "__main__":
    main()
