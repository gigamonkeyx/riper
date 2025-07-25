"""
Test Bark TTS Fix with PyTorch 2.6+ Compatibility
"""

import os
import sys

sys.path.insert(0, "D:/pytorch")

# Set environment
os.environ[
    "OPENROUTER_API_KEY"
] = "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"

print("=== BARK COMPATIBILITY FIX TEST ===")

try:
    from bark_compatibility import test_bark_compatibility

    results = test_bark_compatibility()

    print("Compatibility Test Results:")
    for test, status in results.items():
        if test != "errors":
            status_icon = "✅" if status else "❌"
            print(f"   {test}: {status_icon}")

    if results["errors"]:
        print(f'   Errors: {results["errors"]}')

    # Test TTS with fixed compatibility
    if results["bark_import"]:
        print("\n🔄 Testing TTS with fixed Bark compatibility...")

        from agents import TTSHandler

        tts = TTSHandler()

        print(f"   Bark available: {tts.bark_available}")

        # Test audio generation
        result = tts.process_task(
            "Bark compatibility test successful",
            text="Bark compatibility test successful",
        )

        if result.success:
            print("✅ TTS processing successful with fixed Bark")
            print(f'   Audio generated: {result.data.get("audio_generated", False)}')
            print(f"   Execution time: {result.execution_time:.2f}s")
        else:
            print(f"⚠️ TTS processing: {result.error_message}")
    else:
        print("\n⚠️ Bark import failed - testing text-only TTS...")

        from agents import TTSHandler

        tts = TTSHandler()

        result = tts.process_task("Text-only TTS test", text="Text-only TTS test")

        if result.success:
            print("✅ Text-only TTS processing successful")
            print(f"   Execution time: {result.execution_time:.2f}s")

    print("\n✅ BARK COMPATIBILITY FIX TEST COMPLETE")

except Exception as e:
    print(f"❌ Compatibility test failed: {e}")
    import traceback

    traceback.print_exc()
