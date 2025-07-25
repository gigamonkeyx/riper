"""
Force Complete text.pt Download
Delete incomplete text.pt and force fresh download
"""

import os
import sys
import glob
import shutil

sys.path.insert(0, "D:/pytorch")

# Set environment for D: drive compliance
from aggressive_cache_control import setup_aggressive_d_drive_cache

setup_aggressive_d_drive_cache()

# Set API key
os.environ[
    "OPENROUTER_API_KEY"
] = "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"

print("=== FORCE COMPLETE text.pt DOWNLOAD ===")

# Step 1: Find and delete incomplete text.pt
cache_dirs = [
    "D:\\huggingface_cache",
    "D:\\transformers_cache",
    "D:\\bark_cache",
    "D:\\torch_cache",
]

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        text_files = glob.glob(
            os.path.join(cache_dir, "**", "*text.pt"), recursive=True
        )

        for text_file in text_files:
            try:
                size = os.path.getsize(text_file)
                size_gb = size / (1024**3)

                print(f"Found text.pt: {text_file} ({size_gb:.2f} GB)")

                if size_gb < 3.0:  # Incomplete
                    print(f"DELETING incomplete text.pt: {text_file}")
                    os.remove(text_file)
                    print("Incomplete text.pt deleted successfully")
                else:
                    print("text.pt appears complete, keeping it")

            except Exception as e:
                print(f"Error processing {text_file}: {e}")

print("\nStep 2: Force fresh text.pt download...")

try:
    # Import with weights_only=False to handle the loading issue
    import torch

    # Monkey patch torch.load to use weights_only=False
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(*args, **kwargs)

    torch.load = patched_load
    print("Patched torch.load to use weights_only=False")

    # Now import Bark and force model download
    from bark import preload_models

    print("Bark imported successfully")
    print("Starting fresh text.pt download...")

    # This should download text.pt fresh
    preload_models()

    print("Model preloading completed")
    print("Fresh text.pt download should be complete")

except Exception as e:
    print(f"Fresh download failed: {e}")
    import traceback

    traceback.print_exc()

print("\n=== FORCE DOWNLOAD ATTEMPT FINISHED ===")
