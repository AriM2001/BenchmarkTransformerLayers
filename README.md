# BenchmarkTransformerLayers
This script benchmarks, profiles, and analyzes the performance of a machine translation model from Hugging Face Transformers — specifically designed to measure latency, throughput, FLOPs, and GPU memory usage during inference.

1. Create a fresh env and install deps:

pip install torch torchvision torchaudio
pip install transformers datasets

2. Configuration:

| Variable         | Description                    | Default                        |
| ---------------- | ------------------------------ | ------------------------------ |
| `MODEL_NAME`     | Hugging Face model to use      | `"Helsinki-NLP/opus-mt-en-es"` |
| `DATASET`        | Dataset name and language pair | `("opus_books", "en-es")`      |
| `SPLIT`          | Dataset subset for testing     | `"train[:100]"`                |
| `BATCH_SIZE`     | Number of samples per batch    | `32`                           |
| `MAX_NEW_TOKENS` | Max tokens to generate         | `96`                           |
| `NUM_BEAMS`      | Beam width for decoding        | `4`                            |
| `USE_MIXED_PREC` | Enable mixed precision on GPU  | `True`                         |

3. Usage:

Run the script directly: python main.py

It will:

Load the dataset and translation model.
Print device and precision info.
Perform latency, FLOPs, and memory measurements.
Display top operations from the PyTorch profiler.

Example:

Device: cuda  |  Mixed precision: torch.bfloat16

=== Latency (one batch) ===
Batch size: 32 | num_beams=4 | max_new_tokens=96
mean: 0.2481s | p50: 0.2440s | p95: 0.2560s
Throughput (approx gen tokens/sec): 1432.7

=== FLOPs & Throughput ===
Estimated total FLOPs (this batch): 0.521 TFLOPs
Achieved throughput: 2.18 TFLOP/s

=== Memory ===
Peak allocated: 1842.5 MiB
Peak reserved : 2050.3 MiB
Device total  : 24576 MiB
