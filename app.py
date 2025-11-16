from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import time
import threading
import queue
import os
import sys

# Import your results.py functions
from results import (
    fetch_stock_prompt,
    load_fp16,
    load_int8,
    load_int4,
    load_hybrid,
    load_dynamic,
    measure_generation,
    compute_rouge,
    DEVICE,
    tokenizer
)
import torch

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

def analyze_stock_stream(ticker, result_queue):
    """Analyze stock with different quantization methods"""
    # Get stock data and prompt
    prompt, err = fetch_stock_prompt(ticker)
    
    if err:
        result_queue.put(json.dumps({"type": "error", "message": err}))
        return
    
    # Extract stock data from prompt
    lines = prompt.split('\n')
    stock_data = {
        "ticker": ticker,
        "open": float(lines[1].split('$')[1]),
        "close": float(lines[2].split('$')[1]),
        "high": float(lines[3].split('$')[1]),
        "low": float(lines[4].split('$')[1]),
        "volume": int(lines[5].split(': ')[1].replace(',', ''))
    }
    
    # Send stock data
    result_queue.put(json.dumps({"type": "stock_data", "data": stock_data}))
    
    # Define quantization modes
    modes = {
        "FP16": load_fp16,
        "INT8": load_int8,
        "INT4": load_int4,
        "Hybrid": load_hybrid,
        "Dynamic INT8": load_dynamic,
    }
    
    results = {}
    reference_text = None
    
    for i, (name, loader) in enumerate(modes.items()):
        # Send progress
        progress = int((i / len(modes)) * 100)
        result_queue.put(json.dumps({
            "type": "progress", 
            "progress": progress, 
            "model": name
        }))
        
        try:
            model = loader()
            if model is None:
                results[name] = None
                continue
            
            # Generate text and measure performance
            text, latency, mem, tokens = measure_generation(model, prompt)
            
            # Calculate ROUGE-L score
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
            
            # Cleanup
            del model
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error with {name}: {e}")
            results[name] = None
    
    # Send final results
    result_queue.put(json.dumps({"type": "results", "data": results}))

@app.route('/analyze', methods=['POST', 'OPTIONS'])
def analyze():
    if request.method == 'OPTIONS':
        # Handle preflight request
        return '', 200
    
    data = request.json
    ticker = data.get('ticker', '')
    
    if not ticker:
        return jsonify({"error": "Ticker is required"}), 400
    
    def generate():
        result_queue = queue.Queue()
        thread = threading.Thread(
            target=analyze_stock_stream, 
            args=(ticker, result_queue)
        )
        thread.start()
        
        while thread.is_alive() or not result_queue.empty():
            try:
                result = result_queue.get(timeout=0.1)
                yield f"data: {result}\n\n"
            except queue.Empty:
                continue
        
        thread.join()
    
    return Response(
        generate(), 
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no'
        }
    )

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "device": DEVICE})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)