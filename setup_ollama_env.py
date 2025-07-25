#!/usr/bin/env python3
"""
Setup Ollama environment variables for optimal performance
"""

import os
import subprocess
import time


def setup_ollama_env():
    """Set Ollama environment variables and restart service"""
    print("=== Setting Ollama Environment Variables ===")

    # Set environment variables
    env_vars = {
        "OLLAMA_KEEP_ALIVE": "30m",
        "OLLAMA_NUM_PARALLEL": "4",
        "OLLAMA_HOST": "0.0.0.0:11434",
    }

    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"✅ Set {key}={value}")

    # Stop existing Ollama processes
    print("\n=== Restarting Ollama Service ===")
    try:
        subprocess.run(
            [
                "powershell",
                "-Command",
                "Get-Process ollama -ErrorAction SilentlyContinue | Stop-Process -Force",
            ],
            capture_output=True,
            timeout=10,
        )
        print("✅ Stopped existing Ollama processes")
        time.sleep(2)
    except Exception as e:
        print(f"⚠️ Stop process warning: {e}")

    # Start Ollama with new environment
    try:
        subprocess.Popen(
            ["D:\\ollama\\bin\\ollama.exe", "serve"], env=dict(os.environ, **env_vars)
        )
        print("✅ Started Ollama with new environment")
        time.sleep(5)

        # Verify service
        result = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print("✅ Ollama service verified")
            return True
        else:
            print("❌ Service verification failed")
            return False

    except Exception as e:
        print(f"❌ Service start failed: {e}")
        return False


if __name__ == "__main__":
    success = setup_ollama_env()
    print(f"\nEnvironment setup: {'SUCCESS' if success else 'FAILED'}")
