# 🧠 LLM Quantization Benchmark + Stock Market Summary Generator  
CLI tool for benchmarking large language model quantization modes (FP16, INT8, INT4, Hybrid, Dynamic INT8) while generating **real financial summaries** based on live stock market data.

This script loads **Phi-4-Mini-Instruct (5B)** with various quantization settings using HuggingFace Transformers + BitsAndBytes, evaluates:

- **Latency**
- **Memory usage**
- **Token output length**
- **ROUGE-L similarity** vs FP16 baseline

All tests run on a **single GPU (with CPU/disk offload)** and produce a final comparison table.

---

## ✨ Features

### ✔ Live Stock Data
- Fetches 1-day stock data using **Yahoo Finance (`yfinance`)**
- Auto-builds a structured financial analysis prompt

### ✔ Quantization Modes Included
Five model load modes are benchmarked:

| Mode | Description | Compatibility |
|------|-------------|---------------|
| **FP16 (Baseline)** | Half-precision standard checkpoint | Works with CPU+GPU offload |
| **INT8 (bitsandbytes)** | 8-bit weight quantization | Recommended on RTX 3060 |
| **INT4 (NF4)** | 4-bit quantized weights | May OOM on 12GB GPUs, handled gracefully |
| **Hybrid 4bit-FP16** | 4-bit + FP16 blocks | Good speed/accuracy balance |
| **Dynamic INT8 (CPU)** | PyTorch dynamic quantization | Skipped unless >32GB RAM |

### ✔ Automatic Offloading (Option A)
The script intelligently uses:
- **`device_map="auto"`**
- **CPU offload for FP32 layers**
- **Disk offload (`offload_dir/`)** for oversized tensors

This ensures the model runs even on **RTX 3060 (12GB)**.

### ✔ Safe Error Handling
- INT4 and Dynamic INT8 skip automatically if GPU/RAM insufficient  
- ROUGE computation never crashes  
- Pipelines automatically adjust to Accelerate device maps  

---

## 🛠 Requirements

### Hardware
- **GPU:** NVIDIA RTX 3060 (12GB) or higher  
- **RAM:** At least 16GB recommended  
- **Disk:** ~15GB free for offload caching  

### Software
Install dependencies using **pip** or **uv**:

```bash
pip install torch transformers accelerate bitsandbytes yfinance evaluate psutil
