# Sensitivity Aware Quantizer (WIP)

A tool for analyzing and quantizing transformer models using sensitivity-aware mixed precision quantization.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd sensitivity_aware_quantizer
```

## Usage

The tool provides various command-line options to customize the quantization process. Here are some common usage examples:

### Basic Usage

Run with default settings (uses facebook/opt-125m model with divergence-based method):

```bash
python main.py
```

### Multiple Models Analysis

Analyze multiple models with custom sensitivity method and configuration strategy:

```bash
python main.py \
    --models facebook/opt-125m EleutherAI/gpt-neo-125M \
    --sensitivity_method divergence \
    --config_strategy int4_only
```

### Custom Dataset Configuration

Use a different dataset and adjust sample sizes:

```bash
python main.py \
    --dataset wikitext \
    --dataset_config wikitext-2-raw-v1 \
    --calibration_samples 200 \
    --eval_samples 200
```

### Advanced Configuration Example

Complete example with iterative quantization and perplexity control:

```bash
python main.py \
    --models facebook/opt-125m \
    --sensitivity_method divergence \
    --config_strategy int4_only \
    --use_iterative True \
    --max_ppl_increase 0.1 \
    --batch_size 32 \
    --device cuda
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--models` | List of model names to analyze | `["facebook/opt-125m"]` |
| `--dataset` | Dataset name for calibration and evaluation | `"wikitext"` |
| `--dataset_config` | Dataset configuration | `"wikitext-2-raw-v1"` |
| `--calibration_samples` | Number of samples for calibration | `100` |
| `--eval_samples` | Number of samples for evaluation | `100` |
| `--sensitivity_method` | Method for sensitivity analysis (`divergence`, `hessian`) | `"divergence"` |
| `--config_strategy` | Configuration strategy for quantization (`adaptive_threshold`, `percentile`, `exponential`, `aggressive`, `conservative`, `int4_only`) | `"int4_only"` |
| `--use_iterative` | Use iterative quantization method | `True` |
| `--max_ppl_increase` | Maximum allowed increase in perplexity during quantization | `0.1` |
| `--batch_size` | Batch size for processing | `32` |
| `--device` | Device to use (`auto`, `cuda`, or `cpu`) | `"auto"` |
| `--results_dir` | Directory to save results | `"results"` |
| `--plots_dir` | Directory to save plots | `"plots"` |

## Output

The tool generates:
- Quantization analysis results in JSON format (saved in `results_dir`)
- Comparison plots showing model sizes and perplexity (saved in `plots_dir`)
- Detailed sensitivity scores for each layer
- Mixed precision configuration based on layer sensitivity