#!/usr/bin/env python3
"""
Test small model swarm with CLI fallback
"""

import subprocess
import time
import concurrent.futures
from typing import List, Dict, Any


def test_small_model_cli(model: str, prompt: str) -> Dict[str, Any]:
    """Test small model via CLI"""
    try:
        start_time = time.time()
        cmd = ["ollama", "run", model, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        execution_time = time.time() - start_time

        if result.returncode == 0:
            response = result.stdout.strip()
            return {
                "model": model,
                "success": True,
                "response": response[:100] + "..." if len(response) > 100 else response,
                "execution_time": execution_time,
                "response_length": len(response),
            }
        else:
            return {
                "model": model,
                "success": False,
                "error": result.stderr,
                "execution_time": execution_time,
            }
    except Exception as e:
        return {
            "model": model,
            "success": False,
            "error": str(e),
            "execution_time": 30.0,
        }


def test_model_swarm():
    """Test swarm of small models"""
    print("=== SMALL MODEL SWARM TEST ===")

    models = [
        "llama3.2:1b",  # 1.3GB - smallest
        "deepseek-coder:1.3b",  # 776MB - coding focused
        "qwen3:8b",  # 5.2GB - comparison
    ]

    prompt = "Evaluate fitness score 0.75"

    results = []

    # Test each model
    for model in models:
        print(f"\nTesting {model}...")
        result = test_small_model_cli(model, prompt)
        results.append(result)

        if result["success"]:
            print(
                f"✅ {model}: {result['execution_time']:.1f}s, {result['response_length']} chars"
            )
            print(f"   Response: {result['response']}")
        else:
            print(f"❌ {model}: {result['error']}")

    # Parallel swarm test
    print(f"\n=== PARALLEL SWARM TEST ===")
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        start_time = time.time()
        futures = [
            executor.submit(test_small_model_cli, model, prompt) for model in models
        ]
        parallel_results = [
            future.result() for future in concurrent.futures.as_completed(futures)
        ]
        total_time = time.time() - start_time

    successful = [r for r in parallel_results if r["success"]]
    print(f"Parallel execution: {total_time:.1f}s total")
    print(f"Successful models: {len(successful)}/{len(models)}")

    if successful:
        fastest = min(successful, key=lambda x: x["execution_time"])
        print(f"Fastest: {fastest['model']} ({fastest['execution_time']:.1f}s)")
        return True
    else:
        print("❌ No models responded successfully")
        return False


if __name__ == "__main__":
    success = test_model_swarm()
    print(f"\nSwarm test: {'PASSED' if success else 'FAILED'}")
