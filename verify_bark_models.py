"""
Verify Bark Model Files Completion
Check if text.pt, coarse.pt, and fine.pt are fully downloaded
"""

import os
import glob

print('=== BARK MODEL FILE VERIFICATION ===')

# Check D: drive cache directories for Bark models
cache_dirs = [
    'D:\\huggingface_cache',
    'D:\\transformers_cache', 
    'D:\\bark_cache',
    'D:\\torch_cache'
]

bark_models = ['text.pt', 'coarse.pt', 'fine.pt']
total_found = 0
total_size = 0

print('\nSearching for Bark model files in D: drive cache...')

for cache_dir in cache_dirs:
    if os.path.exists(cache_dir):
        print(f'\n📁 Checking {cache_dir}:')
        
        # Search recursively for .pt files
        pt_files = glob.glob(os.path.join(cache_dir, '**', '*.pt'), recursive=True)
        
        if pt_files:
            for pt_file in pt_files:
                try:
                    size = os.path.getsize(pt_file)
                    size_gb = size / (1024**3)
                    filename = os.path.basename(pt_file)
                    
                    # Check if it's a Bark model
                    is_bark_model = any(model in filename for model in bark_models)
                    status = '🎯' if is_bark_model else '📄'
                    
                    print(f'   {status} {filename}: {size_gb:.2f} GB')
                    
                    if is_bark_model:
                        total_found += 1
                        total_size += size
                        
                except Exception as e:
                    print(f'   ❌ Error reading {pt_file}: {e}')
        else:
            print('   📭 No .pt files found')
    else:
        print(f'   ❌ {cache_dir}: Directory not found')

print(f'\n=== BARK MODEL SUMMARY ===')
print(f'Bark models found: {total_found}/3')
print(f'Total Bark model size: {total_size/1024/1024/1024:.2f} GB')

# Verify specific expected sizes (CORRECTED)
expected_sizes = {
    'text.pt': (2.0, 2.3),     # Expected ~2.16GB (CORRECTED)
    'coarse.pt': (1.0, 1.5),   # Expected ~1.25GB
    'fine.pt': (1.0, 1.2)      # Expected ~1.11GB
}

print(f'\n=== INDIVIDUAL MODEL VERIFICATION ===')

for model_name in bark_models:
    found = False
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            model_files = glob.glob(os.path.join(cache_dir, '**', f'*{model_name}'), recursive=True)
            
            for model_file in model_files:
                try:
                    size = os.path.getsize(model_file)
                    size_gb = size / (1024**3)
                    
                    min_size, max_size = expected_sizes[model_name]
                    
                    if min_size <= size_gb <= max_size:
                        print(f'✅ {model_name}: {size_gb:.2f} GB (COMPLETE)')
                    else:
                        print(f'⚠️ {model_name}: {size_gb:.2f} GB (SIZE MISMATCH - expected {min_size}-{max_size} GB)')
                    
                    found = True
                    break
                except Exception as e:
                    print(f'❌ {model_name}: Error reading file - {e}')
            
            if found:
                break
    
    if not found:
        print(f'❌ {model_name}: NOT FOUND')

# Check if all models are complete AND correct size
complete_models = 0
incomplete_models = []

for model_name in bark_models:
    found = False
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            model_files = glob.glob(os.path.join(cache_dir, '**', f'*{model_name}'), recursive=True)

            for model_file in model_files:
                try:
                    size = os.path.getsize(model_file)
                    size_gb = size / (1024**3)

                    min_size, max_size = expected_sizes[model_name]

                    if min_size <= size_gb <= max_size:
                        complete_models += 1
                    else:
                        incomplete_models.append(f'{model_name} ({size_gb:.2f} GB)')

                    found = True
                    break
                except:
                    pass

            if found:
                break

    if not found:
        incomplete_models.append(f'{model_name} (NOT FOUND)')

# Final status
if complete_models == 3 and len(incomplete_models) == 0:
    print(f'\n🎉 ALL BARK MODELS: COMPLETE')
    print(f'🎉 Total download size: {total_size/1024/1024/1024:.2f} GB')
    print(f'🎉 All files stored on D: drive')
else:
    print(f'\n🚨 BARK MODELS: INCOMPLETE')
    print(f'🚨 Complete models: {complete_models}/3')
    print(f'🚨 Incomplete/missing: {incomplete_models}')
    print(f'🚨 DOWNLOAD NOT FINISHED')

print(f'\n=== VERIFICATION COMPLETE ===')
