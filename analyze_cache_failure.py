"""
Analyze why D: drive redirection is failing
Identify root cause of cache leak to C: drive
"""

import os
import sys
sys.path.insert(0, 'D:/pytorch')

print('=== D: DRIVE REDIRECTION FAILURE ANALYSIS ===')

# Check current environment
print('\n1. Current Environment Variables:')
cache_vars = ['HF_HOME', 'TRANSFORMERS_CACHE', 'HF_DATASETS_CACHE', 'TORCH_HOME', 'TMPDIR', 'TEMP', 'TMP']
for var in cache_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f'   {var}: {value}')

# Check what HuggingFace is actually using
print('\n2. HuggingFace Cache Detection:')
try:
    import transformers
    print(f'   transformers.file_utils.default_cache_path: {transformers.file_utils.default_cache_path}')
except:
    try:
        from transformers.utils import default_cache_path
        print(f'   transformers default_cache_path: {default_cache_path}')
    except Exception as e:
        print(f'   transformers cache detection failed: {e}')

try:
    from huggingface_hub import HF_HUB_CACHE
    print(f'   HF_HUB_CACHE: {HF_HUB_CACHE}')
except Exception as e:
    print(f'   HF_HUB_CACHE detection failed: {e}')

# Check torch hub
print('\n3. PyTorch Hub Detection:')
try:
    import torch
    print(f'   torch.hub.get_dir(): {torch.hub.get_dir()}')
except Exception as e:
    print(f'   torch.hub detection failed: {e}')

# Check what Bark is using
print('\n4. Bark Cache Detection:')
try:
    import bark
    print(f'   bark module location: {bark.__file__}')
    
    # Check if bark has cache settings
    if hasattr(bark, 'CACHE_DIR'):
        print(f'   bark.CACHE_DIR: {bark.CACHE_DIR}')
    
    # Check bark generation module
    try:
        from bark.generation import CACHE_DIR
        print(f'   bark.generation.CACHE_DIR: {CACHE_DIR}')
    except:
        print('   bark.generation.CACHE_DIR: Not found')
        
except Exception as e:
    print(f'   Bark cache detection failed: {e}')

# Check actual cache directories that exist
print('\n5. Existing Cache Directories:')
potential_caches = [
    'C:\\Users\\Ifightcats\\.cache',
    'C:\\Users\\Ifightcats\\AppData\\Local\\huggingface',
    'C:\\Users\\Ifightcats\\AppData\\Local\\torch',
    'D:\\huggingface_cache',
    'D:\\transformers_cache',
    'D:\\bark_cache'
]

for cache_dir in potential_caches:
    if os.path.exists(cache_dir):
        try:
            size = sum(os.path.getsize(os.path.join(dirpath, filename))
                      for dirpath, dirnames, filenames in os.walk(cache_dir)
                      for filename in filenames)
            print(f'   ✅ {cache_dir}: {size/1024/1024/1024:.2f} GB')
        except:
            print(f'   ✅ {cache_dir}: EXISTS (size unknown)')
    else:
        print(f'   ❌ {cache_dir}: NOT FOUND')

# Check what's actually in the environment when modules load
print('\n6. Runtime Environment Check:')
print('   Setting D: drive variables NOW...')

# Force set all variables
os.environ['HF_HOME'] = 'D:\\huggingface_cache'
os.environ['TRANSFORMERS_CACHE'] = 'D:\\transformers_cache'
os.environ['HF_DATASETS_CACHE'] = 'D:\\datasets_cache'
os.environ['TORCH_HOME'] = 'D:\\torch_cache'
os.environ['HUGGINGFACE_HUB_CACHE'] = 'D:\\huggingface_cache'

print('   Variables set. Checking if modules respect them...')

try:
    # Re-import to see if they pick up new values
    import importlib
    if 'transformers' in sys.modules:
        importlib.reload(sys.modules['transformers'])
    
    from huggingface_hub import HF_HUB_CACHE
    print(f'   HF_HUB_CACHE after setting: {HF_HUB_CACHE}')
except Exception as e:
    print(f'   Post-set HF_HUB_CACHE check failed: {e}')

print('\n=== ANALYSIS COMPLETE ===')
