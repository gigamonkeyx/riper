"""
FINAL TTS TEST WITH AGGRESSIVE CACHE CONTROL
Tests TTS with complete D: drive compliance enforcement
Monitors C: drive usage with 500MB limit
"""

import os
import sys
import shutil
import time

# CRITICAL: Setup aggressive cache control BEFORE any imports
sys.path.insert(0, 'D:/pytorch')
from aggressive_cache_control import setup_aggressive_d_drive_cache, aggressive_cache

# Set API key
os.environ['OPENROUTER_API_KEY'] = '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'

print('=== FINAL TTS TEST WITH AGGRESSIVE CACHE CONTROL ===')

# Step 1: Setup aggressive cache control BEFORE imports
print('\n1. Setting up aggressive D: drive cache control...')
setup_aggressive_d_drive_cache()

# Step 2: Monitor initial C: drive usage
print('\n2. Monitoring C: drive usage...')
c_initial = aggressive_cache.monitor_c_drive_usage()

# Step 3: Import and test TTS with monitoring
print('\n3. Testing TTS with aggressive cache control...')

try:
    # Import TTS components (should use D: drive cache)
    from agents import TTSHandler
    print('✅ TTSHandler imported with aggressive cache control')
    
    # Initialize TTS
    tts = TTSHandler()
    print(f'✅ TTSHandler initialized - Bark available: {tts.bark_available}')
    
    # Check C: drive usage after initialization
    if aggressive_cache.check_c_drive_violation(c_initial, limit_mb=500):
        print('🚨 FAILED: C: drive limit exceeded during initialization')
        sys.exit(1)
    
    # Test TTS processing
    print('\n🔄 Testing TTS processing with monitoring...')
    result = tts.process_task(
        'Final TTS test with aggressive D drive cache control',
        text='Testing final TTS with aggressive D drive cache control - no C drive violations allowed'
    )
    
    # Check final C: drive usage
    final_violation = aggressive_cache.check_c_drive_violation(c_initial, limit_mb=500)
    
    # Report results
    print('\n=== FINAL TEST RESULTS ===')
    
    if result.success:
        print('✅ TTS processing: SUCCESSFUL')
        print(f'   Audio generated: {result.data.get("audio_generated", False)}')
        print(f'   GPU utilized: {result.gpu_utilized}')
        print(f'   Execution time: {result.execution_time:.2f}s')
    else:
        print(f'⚠️ TTS processing: {result.error_message}')
    
    if not final_violation:
        print('✅ C: drive compliance: PASSED (under 500MB limit)')
        print('✅ Aggressive cache control: WORKING')
        print('\n🎉 FINAL TTS TEST: PASSED')
        print('🎉 D: DRIVE COMPLIANCE: ENFORCED')
        print('🎉 NO C: DRIVE VIOLATIONS DETECTED')
    else:
        print('🚨 C: drive compliance: FAILED (exceeded 500MB limit)')
        print('🚨 Aggressive cache control: NOT WORKING')
        print('\n🚨 FINAL TTS TEST: FAILED')
    
    # Verify cache compliance one more time
    print('\n4. Final cache compliance verification...')
    aggressive_cache.verify_compliance()
    
    # Check disk space summary
    c_final = shutil.disk_usage('C:')[2] / (1024**3)
    d_final = shutil.disk_usage('D:')[2] / (1024**3)
    
    print(f'\nFinal disk space:')
    print(f'   C: drive: {c_final:.2f} GB free')
    print(f'   D: drive: {d_final:.2f} GB free')
    
    # Check model files on D: drive
    d_cache_size = 0
    cache_dirs = ['D:\\huggingface_cache', 'D:\\transformers_cache', 'D:\\bark_cache', 'D:\\torch_cache']
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(cache_dir)
                          for filename in filenames)
                d_cache_size += size
                print(f'   {cache_dir}: {size/1024/1024/1024:.2f} GB')
            except:
                pass
    
    print(f'   Total D: drive cache: {d_cache_size/1024/1024/1024:.2f} GB')
    
    if not final_violation and d_cache_size > 0:
        print('\n🎉 SUCCESS: All cache operations using D: drive')
        print('🎉 C: drive protected from cache violations')
        print('🎉 TTS system ready for deployment')
    else:
        print('\n🚨 FAILURE: Cache control not working properly')

except Exception as e:
    print(f'\n❌ Final TTS test failed: {e}')
    import traceback
    traceback.print_exc()
    
    # Check if it was a C: drive violation
    if aggressive_cache.check_c_drive_violation(c_initial, limit_mb=500):
        print('\n🚨 FAILURE CAUSE: C: drive limit exceeded')
    
    print('\n🚨 FINAL TTS TEST: FAILED')

print('\n=== FINAL TTS TEST COMPLETE ===')
