# sensitivity_aware_quantizer

Original: FP32 Model
    ↓
Detect Super Weights (FP32 activations)
    ↓
Analyze Sensitivity (Quantize non-super weights)
    ↓
Generate Config (Force ≥16-bit for super weights)
    ↓
Apply Quantization (Preserve super weights in FP32)
    ↓
Evaluate Mixed-Precision Model