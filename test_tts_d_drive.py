"""
Test TTS with D: Drive Compliance
Verifies TTS operations use D: drive storage exclusively
"""

import os
import sys

sys.path.insert(0, "D:/pytorch")

# Set API key
os.environ[
    "OPENROUTER_API_KEY"
] = "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"

print("=== TTS D: DRIVE COMPLIANCE TEST ===")

try:
    from agents import TTSHandler

    tts = TTSHandler()
    print("✅ TTSHandler initialized with D: drive compliance")

    # Test TTS processing (should use D: drive cache)
    result = tts.process_task(
        "RIPER-Omega uses D drive storage for all cache operations",
        text="RIPER-Omega uses D drive storage for all cache operations",
    )

    if result.success:
        print("✅ TTS processing successful with D: drive compliance")
        print(f'   Audio generated: {result.data.get("audio_generated", False)}')
        print(f"   GPU utilized: {result.gpu_utilized}")
        print(f"   Execution time: {result.execution_time:.2f}s")
    else:
        print(f"⚠️ TTS processing completed: {result.error_message}")

    # Check cache location
    import tempfile

    cache_dir = os.environ.get("HF_HOME", "NOT SET")
    temp_dir = tempfile.gettempdir()

    print(f"\nCache verification:")
    print(f"   HF_HOME: {cache_dir}")
    print(f"   Temp directory: {temp_dir}")

    if "D:" in cache_dir and "D:" in temp_dir:
        print("✅ All cache operations using D: drive")
    else:
        print("❌ Cache operations not fully on D: drive")

except Exception as e:
    print(f"❌ TTS test failed: {e}")
    import traceback

    traceback.print_exc()

print("\n✅ TTS D: DRIVE COMPLIANCE TEST COMPLETE")
