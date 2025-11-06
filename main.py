import os, time, statistics, torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.profiler import profile, ProfilerActivity

MODEL_NAME = "Helsinki-NLP/opus-mt-en-es"
DATASET = ("opus_books", "en-es")      # any OPUS variant works (e.g., "tatoeba", "opus_tedtalks")
SPLIT = "train[:100]"                   # subset for speed; expand as needed
BATCH_SIZE = 32
MAX_NEW_TOKENS = 96
NUM_BEAMS = 4                          # beam search; set to 1 for greedy (faster)
USE_MIXED_PREC = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device).eval()

# Mixed precision dtype (GPU only)
amp_dtype = None
if USE_MIXED_PREC and device.type == "cuda":
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

print(f"Device: {device}  |  Mixed precision: {amp_dtype}")

ds = load_dataset(DATASET[0], DATASET[1], split=SPLIT)
sources = [ex["translation"]["en"] for ex in ds]
references = [ex["translation"]["es"] for ex in ds]

def chunked(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i+n]

@torch.inference_mode()
def generate_batch(texts):
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    autocast_kwargs = dict(device_type=device.type, dtype=amp_dtype, enabled=(amp_dtype is not None))
    with torch.autocast(**autocast_kwargs):
        out = model.generate(
            **inputs,
            num_beams=NUM_BEAMS,
            max_new_tokens=MAX_NEW_TOKENS
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def measure_latency(texts, warmup=3, repeats=10):
    # use one representative batch
    batch = texts[:BATCH_SIZE] if len(texts) >= BATCH_SIZE else texts
    # warmup
    for _ in range(warmup):
        _ = generate_batch(batch)
        if device.type == "cuda": torch.cuda.synchronize()
    # timed runs
    times = []
    for _ in range(repeats):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = generate_batch(batch)
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    p50 = statistics.median(times)
    p95 = sorted(times)[int(0.95 * len(times)) - 1]
    mean = statistics.mean(times)

    # rough token throughput: count chars as proxy or compute token lengths properly
    # here we compute generated tokens count from tokenizer
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        gen = model.generate(**inputs, num_beams=NUM_BEAMS, max_new_tokens=MAX_NEW_TOKENS)
    # generated token counts per sample
    gen_lens = [seq.size(0) for seq in gen]
    total_gen_tokens = sum(gen_lens)

    print("\n=== Latency (one batch) ===")
    print(f"Batch size: {len(batch)} | num_beams={NUM_BEAMS} | max_new_tokens={MAX_NEW_TOKENS}")
    print(f"mean: {mean:.4f}s | p50: {p50:.4f}s | p95: {p95:.4f}s")
    print(f"Throughput (approx gen tokens/sec): {total_gen_tokens/mean:.1f}")

measure_latency(sources)

@torch.inference_mode()
def one_profiled_run(texts):
    batch = texts[:BATCH_SIZE] if len(texts) >= BATCH_SIZE else texts
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True
    ) as prof:
        _ = generate_batch(batch)
        if device.type == "cuda": torch.cuda.synchronize()

    # Sum op FLOPs (some ops may report None; skip those)
    total_flops = 0
    for evt in prof.key_averages():
        if getattr(evt, "flops", None):
            total_flops += evt.flops

    # get elapsed from CUDA time (fallback to wall)
    if device.type == "cuda":
        # Safely access cuda_time_total using getattr with a default value of 0
        cuda_time_ms = sum(getattr(e, "cuda_time_total", 0) for e in prof.key_averages())
        elapsed_s = max(cuda_time_ms / 1e3, 1e-9)
    else:
        # rough wall time fallback
        t0 = time.perf_counter(); _ = generate_batch(batch); t1 = time.perf_counter()
        elapsed_s = t1 - t0

    achieved_tflops = (total_flops / elapsed_s) / 1e12 if total_flops else 0.0

    print("\n=== FLOPs & Throughput ===")
    print(f"Estimated total FLOPs (this batch): {total_flops/1e12:.3f} TFLOPs")
    print(f"Elapsed (s): {elapsed_s:.4f}")
    print(f"Achieved throughput: {achieved_tflops:.2f} TFLOP/s")

    # Optional: If you know your GPU's theoretical TFLOP/s, you can compute utilization:
    # utilization = achieved_tflops / THEORETICAL_TFLOPS
    # print(f\"Utilization vs. peak: {100*utilization:.1f}%\")

    return prof

prof = one_profiled_run(sources)

def measure_peak_memory(texts):
    batch = texts[:BATCH_SIZE] if len(texts) >= BATCH_SIZE else texts
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # warmup run
    _ = generate_batch(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    _ = generate_batch(batch)
    if device.type == "cuda": torch.cuda.synchronize()
    t1 = time.perf_counter()

    print("\n=== Memory ===")
    if device.type == "cuda":
        peak_alloc = torch.cuda.max_memory_allocated() / (1024**2)
        peak_reserved = torch.cuda.max_memory_reserved() / (1024**2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
        print(f"Peak allocated: {peak_alloc:.1f} MiB")
        print(f"Peak reserved : {peak_reserved:.1f} MiB")
        print(f"Device total  : {total:.0f} MiB")
    print(f"Wall time for measured run: {t1 - t0:.4f}s")

measure_peak_memory(sources)

# Top ops by CUDA time
print("\n=== Top ops by CUDA/CPU time ===")
print(prof.key_averages().table(
    sort_by="cuda_time_total" if device.type=="cuda" else "cpu_time_total",
    row_limit=15))

# Top ops by self CUDA memory usage
if device.type == "cuda":
    rows = sorted(prof.key_averages(),
                  key=lambda e: getattr(e, "self_cuda_memory_usage", 0),
                  reverse=True)[:10]
    print("\n=== Top ops by self CUDA memory usage ===")
    for r in rows:
        mem = getattr(r, "self_cuda_memory_usage", 0)
        if mem:
            print(f"{r.key:40s}  self_cuda_mem={mem/1024/1024:.1f} MiB  calls={r.count}")