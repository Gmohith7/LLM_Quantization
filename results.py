import os
import time
import psutil
import torch
import yfinance as yf
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
import evaluate

# ------------------------
# CONFIG
# ------------------------
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
OFFLOAD_DIR = "./offload_dir"
os.makedirs(OFFLOAD_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
rouge = evaluate.load("rouge")


# ------------------------
# STOCK PROMPT BUILDER
# ------------------------

def fetch_stock_prompt(ticker):
    ticker = ticker.upper().strip()
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period="1d")
        if hist.empty:
            return None, f"Could not retrieve data for ticker {ticker}"

        open_p = hist["Open"].iloc[-1]
        close = hist["Close"].iloc[-1]
        high = hist["High"].iloc[-1]
        low = hist["Low"].iloc[-1]
        volume = int(hist["Volume"].iloc[-1])

        prompt = (
            f"Ticker: {ticker}\n"
            f"Open: ${open_p:.2f}\n"
            f"Close: ${close:.2f}\n"
            f"High: ${high:.2f}\n"
            f"Low: ${low:.2f}\n"
            f"Volume: {volume:,}\n\n"
            "Using this data, generate a detailed stock performance summary covering:\n"
            "- Market sentiment\n"
            "- Key drivers of price movement\n"
            "- Analyst commentary\n"
            "- Risks and opportunities\n"
            "- Professional investor outlook\n"
        )
        return prompt, None
    except Exception as e:
        return None, f"Error fetching data for {ticker}: {e}"


# ------------------------
# HELPERS
# ------------------------

def get_memory_usage():
    if DEVICE == "cuda":
        return torch.cuda.max_memory_allocated() / 1e6
    return psutil.virtual_memory().used / 1e6


def compute_rouge(ref, gen):
    try:
        score = rouge.compute(predictions=[gen], references=[ref])
        if isinstance(score, float):
            return float(score)
        if "rougeL" in score:
            rl = score["rougeL"]
            if hasattr(rl, "mid"):
                return rl.mid.fmeasure
            return float(rl)
        return 0.0
    except:
        return 0.0


def measure_generation(model, prompt):
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()

    start_mem = get_memory_usage()
    start = time.time()

    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

    out = pipe(prompt, max_new_tokens=150, temperature=0.7, do_sample=False)
    latency = time.time() - start

    text = out[0]["generated_text"]
    mem_used = get_memory_usage() - start_mem
    out_len = len(text.split())

    return text, latency, mem_used, out_len


# ------------------------
# LOADERS (Option A Offload)
# ------------------------

def load_fp16():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
    )


def load_int8():
    conf = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=conf,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
    )


def load_int4():
    print("⚠ INT4 MAY OOM ON RTX 3060 — Attempting...")
    try:
        conf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=True
        )
        return AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=conf,
            device_map="auto",
            offload_folder=OFFLOAD_DIR,
            offload_state_dict=True,
        )
    except Exception:
        return None


def load_hybrid():
    conf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_enable_fp32_cpu_offload=True,
    )
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=conf,
        device_map="auto",
        offload_folder=OFFLOAD_DIR,
        offload_state_dict=True,
    )


def load_dynamic():
    ram_gb = psutil.virtual_memory().total / (1024**3)

    if ram_gb < 32:
        print(f"⌛ Skipping Dynamic INT8 — requires 32GB RAM (you have {ram_gb:.1f}GB)")
        return None

    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
        )
        model.eval()
        return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    except:
        return None


# ------------------------
# MODES
# ------------------------

modes = {
    "FP16": load_fp16,
    "INT8": load_int8,
    "INT4": load_int4,
    "Hybrid": load_hybrid,
}


# ------------------------
# MAIN
# ------------------------

def main():
    ticker = input("Enter Stock Ticker (e.g., AAPL, TSLA): ").strip().upper()
    prompt, err = fetch_stock_prompt(ticker)

    if err:
        print("❌ Error:", err)
        return

    print("\n=== STOCK PROMPT ===\n")
    print(prompt)
    print("\n====================\n")

    results = {}
    reference_text = None

    for name, loader in modes.items():
        print(f"\n🔍 Loading {name}...\n")

        model = loader()
        if model is None:
            print(f"❌ {name} FAILED — Skipping")
            results[name] = None
            continue

        try:
            text, latency, mem, tokens = measure_generation(model, prompt)

            if reference_text is None:
                reference_text = text
                rougeL = 1.0
            else:
                rougeL = compute_rouge(reference_text, text)

            results[name] = {
                "Latency": round(latency, 2),
                "Memory": round(mem, 2),
                "Tokens": tokens,
                "ROUGE-L": round(rougeL, 3),
            }

            print(f"✅ {name} Completed")
            print("Preview:", text[:200], "...")

        except Exception as e:
            print(f"❌ Error running {name}: {e}")
            results[name] = None

        del model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        time.sleep(1)

    print("\n=== FINAL SUMMARY ===")
    for name, data in results.items():
        print(f"\n{name}:")
        print(data if data else "FAILED")


if __name__ == "__main__":
    main()