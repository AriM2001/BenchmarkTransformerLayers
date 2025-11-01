# BenchmarkTransformerLayers
Goal: Benchmark attention/MLP layers on CPU, GPU, and Apple M-series.  Deliverable: Latency, FLOPs utilization, memory bottlenecks.  Grading: Benchmark correctness (30%), Comparison depth (40%), Report (30%).  Reference: pytorch/examplesLinks to an external site.


1. Create a fresh env and install deps:

pip install torch --upgrade
# Optional FLOPs helpers:
pip install fvcore thop


2. Run a small sweep on whatever you have:

python bench_transformer.py --devices cpu \
  --modes attention mlp block \
  --batch-sizes 1 4 --seq-lens 128 1024 \
  --num-heads 8 --head-dim 64 \
  --dtype fp32 bf16 \
  --sdpa-backend auto \
  --cpu-threads 8


3. If you have a GPU or Apple Silicon:

# NVIDIA
python bench_transformer.py --devices cuda --dtype fp32 bf16 --sdpa-backend flash

# Apple Silicon (M-series)
python bench_transformer.py --devices mps --dtype fp32 bf16
