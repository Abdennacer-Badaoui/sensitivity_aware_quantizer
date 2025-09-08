# Sensitivity-Aware Quantizer

A sophisticated tool for analyzing and quantizing transformer models using sensitivity-aware mixed precision quantization. This framework intelligently determines the optimal quantization strategy for each layer based on its sensitivity to quantization, maintaining model performance while maximizing compression.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a7ec6f0b-6fd1-44b3-983d-a91baa5dd738" alt="Sensitivity-Aware Quantization Analysis" />
</p>

## 🎯 Key Features

- **Layer-wise Sensitivity Analysis**: Analyzes each layer's sensitivity to quantization using divergence-based or Hessian-based methods
- **Mixed Precision Quantization**: Automatically assigns optimal precision (INT4/INT8/FP16) to each layer based on sensitivity
- **Multiple Quantization Strategies**: Supports various configuration strategies from conservative to aggressive quantization
- **Iterative Quantization**: Optional iterative approach with perplexity control for fine-tuned results
- **Comprehensive Analysis**: Generates detailed reports, visualizations, and performance comparisons
- **Model Agnostic**: Works with any transformer model from Hugging Face

## 🔧 How It Works

1. **Profiling**: Analyzes model behavior on profiling data to establish baseline performance
2. **Sensitivity Computation**: Calculates sensitivity scores for each layer using:
   - **Divergence Method**: Measures output distribution changes (Jensen-Shannon divergence)
   - **Hessian Method**: Estimates curvature using Hessian eigenvalue approximation
3. **Strategy Application**: Applies quantization configuration based on sensitivity scores:
   - High sensitivity layers → Higher precision (FP16)
   - Medium sensitivity layers → INT8 quantization
   - Low sensitivity layers → INT4 quantization
4. **Evaluation**: Measures final model performance (perplexity, model size, compression ratio)

## 📋 Requirements

- Python 3.8+
- PyTorch 1.12+
- CUDA-compatible GPU (recommended)

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/Abdennacer-Badaoui/sensitivity_aware_quantizer.git
cd sensitivity_aware_quantizer

# Install dependencies
pip install -r requirements.txt
```

## 💻 Usage Example

Here's a complete example that analyzes multiple models with divergence-based sensitivity analysis:

```bash
python main.py \
    --models "HuggingFaceTB/SmolLM2-135M" "google/gemma-3-1b-it" \
    --sensitivity_method divergence \
    --config_strategy adaptive_threshold \
    --profiling_samples 100 \
    --eval_samples 100 \
    --batch_size 32 \
    --mode per_layer \
    --results_dir results \
    --plots_dir plots
```

This command will:
1. Load the specified models
2. Perform sensitivity analysis using divergence method
3. Apply adaptive threshold quantization strategy
4. Generate results in JSON format and visualization plots
5. Save everything to the specified directories

## 📊 Command Line Arguments

| Argument | Type | Choices | Default | Description |
|----------|------|---------|---------|-------------|
| `--models` | list | - | `["HuggingFaceTB/SmolLM2-135M", "google/gemma-3-1b-it", "Qwen/Qwen3-4B"]` | List of Hugging Face model names to analyze |
| `--dataset` | str | - | `"wikitext"` | Dataset name for profiling and evaluation |
| `--dataset_config` | str | - | `"wikitext-2-raw-v1"` | Dataset configuration/subset |
| `--profiling_samples` | int | - | `100` | Number of samples used for sensitivity profiling |
| `--eval_samples` | int | - | `100` | Number of samples used for final evaluation |
| `--mode` | str | `per_layer`, `per_block` | `"per_layer"` | Granularity of sensitivity analysis |
| `--sensitivity_method` | str | `divergence`, `hessian` | `"divergence"` | Method for computing layer sensitivity scores |
| `--config_strategy` | str | `adaptive_threshold`, `percentile`, `exponential`, `aggressive`, `conservative`, `int8_only`, `int4_only` | `"adaptive_threshold"` | Strategy for assigning quantization precision |
| `--use_iterative` | flag | - | `False` | Enable iterative quantization with perplexity monitoring |
| `--max_ppl_increase` | float | - | `0.01` | Maximum allowed perplexity increase during iterative quantization |
| `--layers_per_iteration` | int | - | `5` | Number of layers to quantize per iteration (when using iterative mode) |
| `--max_iterations` | int | - | `100` | Maximum iterations for iterative quantization |
| `--batch_size` | int | - | `32` | Batch size for processing samples |
| `--device` | str | `auto`, `cuda`, `cpu` | `"auto"` | Computing device (auto-detects CUDA availability) |
| `--results_dir` | str | - | `"results"` | Directory to save JSON results and analysis reports |
| `--plots_dir` | str | - | `"plots"` | Directory to save visualization plots and charts |

## 🎯 Configuration Strategies

- **`adaptive_threshold`**: Dynamically adjusts thresholds based on sensitivity distribution
- **`percentile`**: Uses percentile-based thresholds for quantization decisions
- **`exponential`**: Applies exponential weighting to sensitivity scores
- **`aggressive`**: Maximizes compression with more INT4 quantization
- **`conservative`**: Prioritizes accuracy with more FP16 precision
- **`int8_only`**: Forces all quantizable layers to INT8 (uniform quantization)
- **`int4_only`**: Forces all quantizable layers to INT4 (maximum compression)

## 📈 Output Files

The tool generates comprehensive analysis results:

### Results Directory (`results/`)
- `{model_name}_analysis.json`: Complete sensitivity analysis with scores and configuration
- `comparison_summary.json`: Cross-model performance comparison
- Analysis reports with detailed statistics

### Plots Directory (`plots/`)
- Model size vs. perplexity comparison charts
- Sensitivity score distributions
- Layer-wise quantization configuration visualizations
- Performance trade-off analysis

### Key Metrics Reported
- **Original vs. Quantized Perplexity**: Model quality preservation
- **Model Size Reduction**: Compression ratio achieved
- **Layer Sensitivity Scores**: Individual layer analysis
- **Quantization Configuration**: Precision assignment per layer


## 🏗️ Architecture

The framework consists of several key components:

- **`LayerSensitivityAnalyzer`**: Core analysis engine
- **`SensitivityMetrics`**: Sensitivity computation methods (divergence, Hessian)
- **`Quantizer`**: Custom quantized layer implementations (W4A16, W8A16)
- **`Reporter`**: Results analysis and visualization
- **`Utils`**: Multi-model processing and plotting utilities

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.



