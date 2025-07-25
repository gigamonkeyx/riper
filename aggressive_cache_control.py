"""
AGGRESSIVE CACHE CONTROL SYSTEM
Forces ALL cache operations to D: drive BEFORE any imports
Prevents C: drive cache violations completely
"""

import os
import sys
import tempfile
import shutil

class AggressiveCacheController:
    """Completely override all cache mechanisms to use D: drive"""
    
    def __init__(self):
        self.d_drive_enforced = False
        self.original_functions = {}
        
    def enforce_d_drive_before_imports(self):
        """MUST be called before ANY ML library imports"""
        if self.d_drive_enforced:
            return
        
        print("🚨 ENFORCING D: DRIVE CACHE BEFORE IMPORTS")
        
        # 1. Set ALL environment variables
        self._set_all_environment_variables()
        
        # 2. Create all directories
        self._create_cache_directories()
        
        # 3. Monkey patch import system
        self._monkey_patch_import_system()
        
        # 4. Override tempfile
        self._override_tempfile()
        
        self.d_drive_enforced = True
        print("✅ D: DRIVE ENFORCEMENT COMPLETE")
    
    def _set_all_environment_variables(self):
        """Set every possible cache environment variable"""
        cache_vars = {
            # HuggingFace
            'HF_HOME': 'D:\\huggingface_cache',
            'TRANSFORMERS_CACHE': 'D:\\transformers_cache',
            'HF_DATASETS_CACHE': 'D:\\datasets_cache',
            'HUGGINGFACE_HUB_CACHE': 'D:\\huggingface_cache',
            'HF_HUB_CACHE': 'D:\\huggingface_cache',
            'PYTORCH_TRANSFORMERS_CACHE': 'D:\\transformers_cache',
            'PYTORCH_PRETRAINED_BERT_CACHE': 'D:\\transformers_cache',
            
            # PyTorch
            'TORCH_HOME': 'D:\\torch_cache',
            'TORCH_HUB': 'D:\\torch_cache',
            
            # General
            'XDG_CACHE_HOME': 'D:\\cache',
            'TMPDIR': 'D:\\temp',
            'TEMP': 'D:\\temp',
            'TMP': 'D:\\temp',
            
            # Bark/Suno
            'SUNO_OFFLOAD_CPU': 'True',
            'SUNO_USE_SMALL_MODELS': 'True',
            'BARK_CACHE_DIR': 'D:\\bark_cache',
            
            # Additional
            'CONDA_PKGS_DIRS': 'D:\\conda_cache',
            'PIP_CACHE_DIR': 'D:\\pip_cache',
            'NLTK_DATA': 'D:\\nltk_data'
        }
        
        for var, path in cache_vars.items():
            os.environ[var] = path
            print(f"SET {var} = {path}")
    
    def _create_cache_directories(self):
        """Create all cache directories on D: drive"""
        cache_dirs = [
            'D:\\huggingface_cache',
            'D:\\transformers_cache', 
            'D:\\datasets_cache',
            'D:\\torch_cache',
            'D:\\cache',
            'D:\\temp',
            'D:\\bark_cache',
            'D:\\conda_cache',
            'D:\\pip_cache',
            'D:\\nltk_data'
        ]
        
        for cache_dir in cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"CREATED {cache_dir}")
    
    def _monkey_patch_import_system(self):
        """Monkey patch the import system to intercept cache functions"""
        import builtins
        original_import = builtins.__import__

        def patched_import(name, *args, **kwargs):
            module = original_import(name, *args, **kwargs)

            # Patch transformers immediately on import
            if name == 'transformers' or name.startswith('transformers.'):
                self._patch_transformers(module)

            # Patch torch immediately on import
            elif name == 'torch' or name.startswith('torch.'):
                self._patch_torch(module)

            # Patch bark immediately on import
            elif name == 'bark' or name.startswith('bark.'):
                self._patch_bark(module)

            return module

        builtins.__import__ = patched_import
        print("PATCHED import system")
    
    def _patch_transformers(self, module):
        """Patch transformers cache functions"""
        try:
            if hasattr(module, 'file_utils'):
                module.file_utils.default_cache_path = "D:\\huggingface_cache"
                print("PATCHED transformers.file_utils.default_cache_path")
        except:
            pass
    
    def _patch_torch(self, module):
        """Patch torch cache functions"""
        try:
            if hasattr(module, 'hub'):
                module.hub.set_dir('D:\\torch_cache')
                print("PATCHED torch.hub directory")
        except:
            pass
    
    def _patch_bark(self, module):
        """Patch bark cache functions"""
        try:
            if hasattr(module, 'generation'):
                module.generation.CACHE_DIR = "D:\\bark_cache"
                print("PATCHED bark.generation.CACHE_DIR")
        except:
            pass
    
    def _override_tempfile(self):
        """Override tempfile to use D: drive"""
        tempfile.tempdir = "D:\\temp"
        
        # Monkey patch tempfile functions
        original_gettempdir = tempfile.gettempdir
        
        def force_d_tempdir():
            return "D:\\temp"
        
        tempfile.gettempdir = force_d_tempdir
        print("PATCHED tempfile.gettempdir")
    
    def verify_compliance(self):
        """Verify all cache operations are using D: drive"""
        print("\n=== CACHE COMPLIANCE VERIFICATION ===")
        
        # Check environment variables
        critical_vars = ['HF_HOME', 'TRANSFORMERS_CACHE', 'TORCH_HOME']
        for var in critical_vars:
            value = os.environ.get(var, 'NOT SET')
            compliance = '✅' if 'D:' in str(value) else '❌'
            print(f"{compliance} {var}: {value}")
        
        # Check actual module cache paths
        try:
            import transformers
            cache_path = getattr(transformers.file_utils, 'default_cache_path', 'NOT FOUND')
            compliance = '✅' if 'D:' in str(cache_path) else '❌'
            print(f"{compliance} transformers cache: {cache_path}")
        except:
            print("❌ transformers not imported")
        
        try:
            import torch
            hub_dir = torch.hub.get_dir()
            compliance = '✅' if 'D:' in hub_dir else '❌'
            print(f"{compliance} torch hub: {hub_dir}")
        except:
            print("❌ torch not imported")
        
        try:
            import tempfile
            temp_dir = tempfile.gettempdir()
            compliance = '✅' if 'D:' in temp_dir else '❌'
            print(f"{compliance} tempfile: {temp_dir}")
        except:
            print("❌ tempfile check failed")
        
        print("=== VERIFICATION COMPLETE ===")
    
    def monitor_c_drive_usage(self):
        """Monitor C: drive usage during operations"""
        c_free_before = shutil.disk_usage('C:')[2] / (1024**2)  # MB
        print(f"C: drive free space: {c_free_before:.1f} MB")
        return c_free_before
    
    def check_c_drive_violation(self, initial_free, limit_mb=500):
        """Check if C: drive usage exceeded limit"""
        c_free_after = shutil.disk_usage('C:')[2] / (1024**2)  # MB
        usage = initial_free - c_free_after
        
        if usage > limit_mb:
            print(f"🚨 C: DRIVE VIOLATION: {usage:.1f} MB used > {limit_mb} MB limit")
            return True
        else:
            print(f"✅ C: drive usage: {usage:.1f} MB (within {limit_mb} MB limit)")
            return False


# Global instance
aggressive_cache = AggressiveCacheController()


def setup_aggressive_d_drive_cache():
    """Setup aggressive D: drive cache control - call BEFORE any imports"""
    aggressive_cache.enforce_d_drive_before_imports()


def verify_d_drive_compliance():
    """Verify D: drive compliance is working"""
    aggressive_cache.verify_compliance()


if __name__ == "__main__":
    # Test the aggressive cache control
    setup_aggressive_d_drive_cache()
    verify_d_drive_compliance()
