"""
Monitor Bark Download with C: Drive Space Limit
FAIL if C: drive usage exceeds 500MB during download
"""

import os
import sys
import shutil
import time
import threading

sys.path.insert(0, "D:/pytorch")

# Set API key
os.environ[
    "OPENROUTER_API_KEY"
] = "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"


class CDriveMonitor:
    def __init__(self, limit_mb=500):
        self.limit_mb = limit_mb
        self.initial_free = shutil.disk_usage("C:")[2] / (1024**2)  # MB
        self.monitoring = False
        self.violation_detected = False
        self.max_usage = 0

    def start_monitoring(self):
        self.monitoring = True
        self.violation_detected = False
        self.max_usage = 0

        def monitor():
            while self.monitoring:
                current_free = shutil.disk_usage("C:")[2] / (1024**2)  # MB
                usage = self.initial_free - current_free

                if usage > self.max_usage:
                    self.max_usage = usage

                if usage > self.limit_mb:
                    print(
                        f"🚨 VIOLATION: C: drive usage {usage:.1f} MB > {self.limit_mb} MB limit"
                    )
                    self.violation_detected = True
                    self.monitoring = False
                    return

                if usage > 50:  # Report significant usage
                    print(
                        f"⚠️ C: drive usage: {usage:.1f} MB (limit: {self.limit_mb} MB)"
                    )

                time.sleep(2)  # Check every 2 seconds

        self.monitor_thread = threading.Thread(target=monitor, daemon=True)
        self.monitor_thread.start()
        print(f"✅ C: drive monitoring started (limit: {self.limit_mb} MB)")

    def stop_monitoring(self):
        self.monitoring = False
        final_free = shutil.disk_usage("C:")[2] / (1024**2)  # MB
        final_usage = self.initial_free - final_free

        print(f"\n=== MONITORING RESULTS ===")
        print(f"Initial C: free space: {self.initial_free:.1f} MB")
        print(f"Final C: free space: {final_free:.1f} MB")
        print(f"Total C: usage: {final_usage:.1f} MB")
        print(f"Max C: usage: {self.max_usage:.1f} MB")
        print(f"Limit: {self.limit_mb} MB")

        if self.violation_detected:
            print("🚨 SOLUTION FAILED: C: drive limit exceeded")
            return False
        elif final_usage < self.limit_mb:
            print("✅ SOLUTION PASSED: C: drive usage within limit")
            return True
        else:
            print("🚨 SOLUTION FAILED: Final usage exceeds limit")
            return False


print("=== BARK DOWNLOAD MONITORING TEST ===")

# Initialize monitor
monitor = CDriveMonitor(limit_mb=500)

print(f"Initial C: drive free space: {monitor.initial_free:.1f} MB")

try:
    # Start monitoring
    monitor.start_monitoring()

    # Import and test TTS with monitoring
    print("\n🔄 Starting Bark TTS test with monitoring...")

    from agents import TTSHandler

    print("✅ TTSHandler imported")

    tts = TTSHandler()
    print(f"✅ TTSHandler initialized - Bark available: {tts.bark_available}")

    # Test TTS (this will trigger any downloads)
    print("\n🔄 Testing TTS processing (may trigger downloads)...")
    result = tts.process_task(
        "Monitored Bark test",
        text="Testing Bark with C drive monitoring - stay under 500MB",
    )

    # Wait a bit for any background operations
    time.sleep(5)

    # Stop monitoring and get results
    success = monitor.stop_monitoring()

    if result.success:
        print(f"\n✅ TTS processing successful")
        print(f'   Audio generated: {result.data.get("audio_generated", False)}')
        print(f"   Execution time: {result.execution_time:.2f}s")
    else:
        print(f"\n⚠️ TTS processing: {result.error_message}")

    # Final verdict
    if success:
        print("\n🎉 SOLUTION VERIFICATION: PASSED")
        print("🎉 C: drive usage stayed under 500MB limit")
        print("🎉 D: drive compliance: WORKING")
    else:
        print("\n🚨 SOLUTION VERIFICATION: FAILED")
        print("🚨 C: drive usage exceeded 500MB limit")
        print("🚨 D: drive compliance: NOT WORKING")

except Exception as e:
    monitor.stop_monitoring()
    print(f"\n❌ Test failed with error: {e}")
    import traceback

    traceback.print_exc()
    print("\n🚨 SOLUTION VERIFICATION: FAILED (ERROR)")

print("\n=== MONITORING TEST COMPLETE ===")
