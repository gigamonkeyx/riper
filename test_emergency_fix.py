"""
Test Emergency D: Drive Fix for Bark TTS
Verifies no more C: drive cache violations
"""

import os
import sys
import shutil
sys.path.insert(0, 'D:/pytorch')

# Set API key
os.environ['OPENROUTER_API_KEY'] = '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'

print('=== EMERGENCY D: DRIVE FIX VERIFICATION ===')

# Check initial disk space
c_initial = shutil.disk_usage('C:')[2] / (1024**3)
d_initial = shutil.disk_usage('D:')[2] / (1024**3)

print(f'Initial disk space:')
print(f'   C: drive: {c_initial:.2f} GB free')
print(f'   D: drive: {d_initial:.2f} GB free')

try:
    from agents import TTSHandler
    
    print('\n✅ TTSHandler imported with emergency D: drive fix')
    
    tts = TTSHandler()
    print(f'✅ TTSHandler initialized - Bark available: {tts.bark_available}')
    
    # Test TTS with emergency fix (should use D: drive only)
    print('\n🔄 Testing TTS with emergency D: drive compliance...')
    result = tts.process_task(
        'Emergency D drive fix verification',
        text='Testing emergency D drive fix - no C drive usage allowed'
    )
    
    if result.success:
        print('✅ TTS processing successful with emergency D: drive fix')
        print(f'   Audio generated: {result.data.get("audio_generated", False)}')
        print(f'   GPU utilized: {result.gpu_utilized}')
        print(f'   Execution time: {result.execution_time:.2f}s')
    else:
        print(f'❌ TTS processing failed: {result.error_message}')
    
    # Check final disk space
    c_final = shutil.disk_usage('C:')[2] / (1024**3)
    d_final = shutil.disk_usage('D:')[2] / (1024**3)
    
    print(f'\nFinal disk space:')
    print(f'   C: drive: {c_final:.2f} GB free')
    print(f'   D: drive: {d_final:.2f} GB free')
    
    # Verify no C: drive usage
    c_change = c_initial - c_final
    d_change = d_initial - d_final
    
    print(f'\nDisk space changes:')
    print(f'   C: drive change: {c_change:.3f} GB')
    print(f'   D: drive change: {d_change:.3f} GB')
    
    if abs(c_change) < 0.001:  # Less than 1MB change
        print('✅ SUCCESS: No C: drive usage detected')
        print('✅ Emergency D: drive compliance: WORKING')
    else:
        print('❌ FAILURE: C: drive usage detected')
        print('❌ Cache leak still present')
    
    if d_change > 0:
        print(f'✅ D: drive used for cache: {d_change:.3f} GB')
    
    print('\n=== EMERGENCY FIX VERIFICATION COMPLETE ===')
    
    if abs(c_change) < 0.001:
        print('🎉 EMERGENCY D: DRIVE FIX: SUCCESSFUL')
        print('🎉 NO MORE C: DRIVE VIOLATIONS')
    else:
        print('🚨 EMERGENCY FIX: FAILED')
        print('🚨 C: DRIVE STILL BEING USED')
    
except Exception as e:
    print(f'❌ Emergency fix verification failed: {e}')
    import traceback
    traceback.print_exc()
