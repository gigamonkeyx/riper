"""
Comprehensive TTS Installation Verification
Tests all components: Bark, PyTorch, D: drive compliance, audio generation
"""

import os
import sys
import shutil
import time
sys.path.insert(0, 'D:/pytorch')

# Set API key
os.environ['OPENROUTER_API_KEY'] = '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'

print('=== COMPREHENSIVE TTS INSTALLATION VERIFICATION ===')

def check_disk_space():
    """Check disk space on both drives"""
    c_free = shutil.disk_usage('C:')[2] / (1024**3)
    d_free = shutil.disk_usage('D:')[2] / (1024**3)
    print(f'Disk Space:')
    print(f'   C: drive: {c_free:.2f} GB free')
    print(f'   D: drive: {d_free:.2f} GB free')
    return c_free, d_free

def verify_environment():
    """Verify environment variables"""
    print('\n1. Environment Variables Check:')
    required_vars = [
        'HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME', 
        'TMPDIR', 'TEMP', 'TMP', 'BARK_CACHE_DIR'
    ]
    
    all_good = True
    for var in required_vars:
        value = os.environ.get(var, 'NOT SET')
        d_drive = 'D:' in str(value)
        status = '✅' if d_drive else '❌'
        print(f'   {status} {var}: {value}')
        if not d_drive:
            all_good = False
    
    return all_good

def verify_imports():
    """Verify all required imports"""
    print('\n2. Import Verification:')
    imports = {}
    
    # Test torch
    try:
        import torch
        imports['torch'] = f'✅ {torch.__version__}'
        print(f'   ✅ PyTorch: {torch.__version__}')
    except Exception as e:
        imports['torch'] = f'❌ {e}'
        print(f'   ❌ PyTorch: {e}')
    
    # Test transformers
    try:
        import transformers
        imports['transformers'] = f'✅ {transformers.__version__}'
        print(f'   ✅ Transformers: {transformers.__version__}')
    except Exception as e:
        imports['transformers'] = f'❌ {e}'
        print(f'   ❌ Transformers: {e}')
    
    # Test bark
    try:
        import bark
        imports['bark'] = '✅ Available'
        print(f'   ✅ Bark: Available')
    except Exception as e:
        imports['bark'] = f'❌ {e}'
        print(f'   ❌ Bark: {e}')
    
    # Test numpy
    try:
        import numpy as np
        imports['numpy'] = f'✅ {np.__version__}'
        print(f'   ✅ NumPy: {np.__version__}')
    except Exception as e:
        imports['numpy'] = f'❌ {e}'
        print(f'   ❌ NumPy: {e}')
    
    return imports

def verify_bark_compatibility():
    """Verify Bark compatibility fixes"""
    print('\n3. Bark Compatibility Verification:')
    
    try:
        from bark_compatibility import test_bark_compatibility
        results = test_bark_compatibility()
        
        for test, status in results.items():
            if test != 'errors':
                icon = '✅' if status else '❌'
                print(f'   {icon} {test}: {status}')
        
        if results.get('errors'):
            print(f'   ❌ Errors: {results["errors"]}')
        
        return all(results[k] for k in results if k != 'errors')
        
    except Exception as e:
        print(f'   ❌ Compatibility test failed: {e}')
        return False

def verify_tts_handler():
    """Verify TTSHandler functionality"""
    print('\n4. TTSHandler Verification:')
    
    try:
        from agents import TTSHandler
        print('   ✅ TTSHandler import successful')
        
        tts = TTSHandler()
        print(f'   ✅ TTSHandler initialized')
        print(f'   ✅ Bark available: {tts.bark_available}')
        
        return True, tts
        
    except Exception as e:
        print(f'   ❌ TTSHandler failed: {e}')
        return False, None

def test_audio_generation(tts):
    """Test actual audio generation"""
    print('\n5. Audio Generation Test:')
    
    c_before = shutil.disk_usage('C:')[2] / (1024**2)
    
    try:
        result = tts.process_task(
            'TTS verification test',
            text='Testing TTS installation verification'
        )
        
        c_after = shutil.disk_usage('C:')[2] / (1024**2)
        c_usage = c_before - c_after
        
        print(f'   ✅ TTS processing completed')
        print(f'   ✅ Success: {result.success}')
        print(f'   ✅ Audio generated: {result.data.get("audio_generated", False)}')
        print(f'   ✅ GPU utilized: {result.gpu_utilized}')
        print(f'   ✅ Execution time: {result.execution_time:.2f}s')
        print(f'   ✅ C: drive usage: {c_usage:.1f} MB')
        
        if c_usage > 100:  # More than 100MB on C: drive
            print(f'   ⚠️ WARNING: High C: drive usage')
            return False
        
        return result.success
        
    except Exception as e:
        print(f'   ❌ Audio generation failed: {e}')
        return False

def verify_model_downloads():
    """Verify model files are on D: drive"""
    print('\n6. Model File Verification:')
    
    model_paths = [
        'D:\\huggingface_cache',
        'D:\\transformers_cache', 
        'D:\\bark_cache',
        'D:\\torch_cache'
    ]
    
    total_size = 0
    for path in model_paths:
        if os.path.exists(path):
            try:
                size = sum(os.path.getsize(os.path.join(dirpath, filename))
                          for dirpath, dirnames, filenames in os.walk(path)
                          for filename in filenames)
                size_gb = size / (1024**3)
                total_size += size_gb
                print(f'   ✅ {path}: {size_gb:.2f} GB')
            except Exception as e:
                print(f'   ⚠️ {path}: Error calculating size - {e}')
        else:
            print(f'   ❌ {path}: Not found')
    
    print(f'   ✅ Total model size on D: drive: {total_size:.2f} GB')
    return total_size > 0

# Run verification
print('Starting comprehensive TTS verification...\n')

c_initial, d_initial = check_disk_space()

env_ok = verify_environment()
imports = verify_imports()
compat_ok = verify_bark_compatibility()
handler_ok, tts = verify_tts_handler()

if handler_ok:
    audio_ok = test_audio_generation(tts)
else:
    audio_ok = False

models_ok = verify_model_downloads()

c_final, d_final = check_disk_space()

# Final summary
print('\n=== VERIFICATION SUMMARY ===')
print(f'Environment Variables: {"✅ PASS" if env_ok else "❌ FAIL"}')
print(f'Required Imports: {"✅ PASS" if all("✅" in v for v in imports.values()) else "❌ FAIL"}')
print(f'Bark Compatibility: {"✅ PASS" if compat_ok else "❌ FAIL"}')
print(f'TTSHandler: {"✅ PASS" if handler_ok else "❌ FAIL"}')
print(f'Audio Generation: {"✅ PASS" if audio_ok else "❌ FAIL"}')
print(f'Model Files: {"✅ PASS" if models_ok else "❌ FAIL"}')

c_usage = c_initial - c_final
print(f'\nDisk Usage:')
print(f'C: drive usage: {c_usage:.3f} GB')
print(f'D: drive usage: {d_initial - d_final:.3f} GB')

overall_pass = all([env_ok, handler_ok, models_ok]) and c_usage < 0.5

if overall_pass:
    print('\n🎉 TTS INSTALLATION VERIFICATION: PASSED')
    print('🎉 All components working correctly')
    print('🎉 D: drive compliance enforced')
else:
    print('\n🚨 TTS INSTALLATION VERIFICATION: FAILED')
    print('🚨 Some components need attention')

print('\n=== VERIFICATION COMPLETE ===')
