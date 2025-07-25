"""
Complete text.pt Download
Force completion of the incomplete text.pt model file
"""

import os
import sys
sys.path.insert(0, 'D:/pytorch')

# Set environment for D: drive compliance
from aggressive_cache_control import setup_aggressive_d_drive_cache
setup_aggressive_d_drive_cache()

# Set API key
os.environ['OPENROUTER_API_KEY'] = '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'

print('=== COMPLETING text.pt DOWNLOAD ===')
print('Current text.pt size: 2.16 GB (incomplete)')
print('Expected text.pt size: ~3.3 GB')
print('Missing: ~1.1 GB')

try:
    # Force re-download of text.pt by importing Bark
    print('\nForcing text.pt download completion...')
    
    import torch
    from bark import preload_models
    
    print('Bark imported successfully')
    
    # This should complete the text.pt download
    print('Preloading models (will complete text.pt download)...')
    preload_models()
    
    print('Model preloading completed')
    print('text.pt download should now be complete')
    
except Exception as e:
    print(f'Download completion failed: {e}')
    import traceback
    traceback.print_exc()

print('\n=== DOWNLOAD COMPLETION ATTEMPT FINISHED ===')
