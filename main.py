import os
import matplotlib.pyplot as plt
import argparse
from utils import run_analysis_for_models, plot_comparisons

MODELS = [
    "facebook/opt-125m",
    "EleutherAI/gpt-neo-125M",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]


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

    # Sensitivity analysis methods
    parser.add_argument(
        "--sensitivity_method",
        type=str,
        default="divergence",
        choices=["divergence", "hessian"],
        help="Method for sensitivity analysis",
    )

    parser.add_argument(
        "--config_strategy",
        type=str,
        default="adaptive_threshold",
        choices=[
            "adaptive_threshold",
            "percentile",
            "exponential",
            "aggressive",
            "conservative",
            "int8_only",
            "int4_only",
        ],
        help="Configuration strategy for quantization",
    )

    parser.add_argument(
        "--use_iterative",
        action="store_true",
        default=False,
        help="Use iterative quantization method (default: False). Use --use_iterative to enable.",
    )

    parser.add_argument(
        "--max_ppl_increase",
        type=float,
        default=0.01,
        help="Maximum allowed increase in perplexity during quantization",
    )

    parser.add_argument(
        "--layers_per_iteration",
        type=int,
        default=3,
        help="Number of layers to process per iteration",
    )

    parser.add_argument(
        "--max_iterations",
        type=int,
        default=100,
        help="Maximum number of iterations for processing",
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run benchmark evaluation on the models (default: False). Use --benchmark to enable.",
    )

    # Processing parameters
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for processing"
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
        results_dir=args.results_dir,
        models=args.models,
        calibration_data=None,
        eval_data=None,
        calibration_num_samples=args.calibration_samples,
        eval_num_samples=args.eval_samples,
        batch_size=args.batch_size,
        sensitivity_method=args.sensitivity_method,
        config_strategy=args.config_strategy,
        use_iterative=args.use_iterative,
        max_perplexity_increase=args.max_ppl_increase,
        layers_per_iteration=args.layers_per_iteration,
        max_iterations=args.max_iterations,
        benchmark=args.benchmark,
    )

    if all_results:
        plot_comparisons(
            args.plots_dir,
            all_results,
            args.sensitivity_method,
            args.calibration_samples,
            args.config_strategy,
            args.use_iterative,
            args.max_ppl_increase,
            args.benchmark,
        )
        print(
            f"\nAnalysis and visualizations complete. See {args.results_dir} and {args.plots_dir}."
        )
    else:
        print("No results were generated. Check the errors above.")


if __name__ == "__main__":
    main()
