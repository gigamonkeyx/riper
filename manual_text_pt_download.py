"""
Manual text.pt Download
Attempt to manually download the complete text.pt model
"""

import os
import sys
import requests
import shutil
from tqdm import tqdm

sys.path.insert(0, "D:/pytorch")

# Set environment for D: drive compliance
from aggressive_cache_control import setup_aggressive_d_drive_cache

setup_aggressive_d_drive_cache()

print("=== MANUAL text.pt DOWNLOAD ===")

# Bark text.pt model URL (from HuggingFace)
TEXT_MODEL_URL = "https://huggingface.co/suno/bark/resolve/main/text.pt"
CACHE_DIR = "D:\\bark_cache"
TEXT_PT_PATH = os.path.join(CACHE_DIR, "text.pt")

print(f"Target URL: {TEXT_MODEL_URL}")
print(f"Download path: {TEXT_PT_PATH}")

# Delete existing incomplete file
if os.path.exists(TEXT_PT_PATH):
    size = os.path.getsize(TEXT_PT_PATH) / (1024**3)
    print(f"Existing text.pt: {size:.2f} GB")

    if size < 3.0:
        print("Deleting incomplete text.pt...")
        os.remove(TEXT_PT_PATH)
        print("Incomplete file deleted")
    else:
        print("File appears complete, keeping it")
        sys.exit(0)

print("\nStarting manual download...")

try:
    # Download with progress bar
    response = requests.get(TEXT_MODEL_URL, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    print(f"Expected download size: {total_size / (1024**3):.2f} GB")

    with open(TEXT_PT_PATH, "wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True, desc="text.pt") as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    # Verify download
    final_size = os.path.getsize(TEXT_PT_PATH) / (1024**3)
    print(f"\nDownload completed!")
    print(f"Final size: {final_size:.2f} GB")

    if final_size >= 3.0:
        print("✅ text.pt download SUCCESSFUL")
        print("✅ File size is within expected range")
    else:
        print("⚠️ text.pt download may be incomplete")
        print("⚠️ File size is smaller than expected")

except Exception as e:
    print(f"❌ Manual download failed: {e}")
    import traceback

    traceback.print_exc()

print("\n=== MANUAL DOWNLOAD ATTEMPT FINISHED ===")
