"""
Bark TTS Compatibility Module for PyTorch 2.6+
Handles all compatibility issues with modern PyTorch versions
"""

import os
import sys
import warnings
import logging
from typing import Optional, Any, Dict
import tempfile

logger = logging.getLogger(__name__)


class BarkCompatibilityManager:
    """Manages Bark TTS compatibility with PyTorch 2.6+"""

    def __init__(self):
        self.original_load = None
        self.compatibility_applied = False
        self._setup_environment()

    def _setup_environment(self):
        """Setup environment for Bark compatibility - FORCE D: drive"""
        # CRITICAL: Force ALL cache operations to D: drive
        cache_vars = {
            "HF_HOME": "D:\\huggingface_cache",
            "TRANSFORMERS_CACHE": "D:\\transformers_cache",
            "HF_DATASETS_CACHE": "D:\\datasets_cache",
            "TORCH_HOME": "D:\\torch_cache",
            "TMPDIR": "D:\\temp",
            "TEMP": "D:\\temp",
            "TMP": "D:\\temp",
            "PYTORCH_TRANSFORMERS_CACHE": "D:\\transformers_cache",
            "PYTORCH_PRETRAINED_BERT_CACHE": "D:\\transformers_cache",
            "XDG_CACHE_HOME": "D:\\cache",
            "HUGGINGFACE_HUB_CACHE": "D:\\huggingface_cache",
            "SUNO_OFFLOAD_CPU": "True",
            "SUNO_USE_SMALL_MODELS": "True",
            "BARK_CACHE_DIR": "D:\\bark_cache",
        }

        # FORCE override ALL environment variables
        for var, path in cache_vars.items():
            os.environ[var] = path
            print(f"FORCED {var} = {path}")

        # Set tempfile directory
        tempfile.tempdir = "D:\\temp"

        # Create ALL cache directories
        cache_dirs = [
            "D:\\huggingface_cache",
            "D:\\transformers_cache",
            "D:\\datasets_cache",
            "D:\\torch_cache",
            "D:\\cache",
            "D:\\temp",
            "D:\\bark_cache",
        ]

        for cache_dir in cache_dirs:
            os.makedirs(cache_dir, exist_ok=True)
            print(f"Created directory: {cache_dir}")

        # CRITICAL: Override torch hub directory
        try:
            import torch

            torch.hub.set_dir("D:\\torch_cache")
            print(f"FORCED torch.hub directory: D:\\torch_cache")
        except Exception as e:
            print(f"Warning: Could not set torch.hub directory: {e}")

        # AGGRESSIVE: Monkey patch cache functions BEFORE import
        self._monkey_patch_cache_functions()

        print("🚨 AGGRESSIVE D: DRIVE COMPLIANCE ENFORCED")

    def _monkey_patch_cache_functions(self):
        """Aggressively monkey patch all cache functions to use D: drive"""
        import sys

        # Monkey patch transformers cache before import
        def force_d_cache_path():
            return "D:\\huggingface_cache"

        # Monkey patch torch hub before import
        def force_torch_hub():
            return "D:\\torch_cache"

        # Store original functions if modules already imported
        if "transformers" in sys.modules:
            try:
                import transformers.file_utils

                transformers.file_utils.default_cache_path = "D:\\huggingface_cache"
                print("PATCHED: transformers.file_utils.default_cache_path")
            except:
                pass

        if "torch" in sys.modules:
            try:
                import torch

                torch.hub.set_dir("D:\\torch_cache")
                print("PATCHED: torch.hub directory")
            except:
                pass

        if "bark" in sys.modules:
            try:
                import bark.generation

                bark.generation.CACHE_DIR = "D:\\bark_cache"
                print("PATCHED: bark.generation.CACHE_DIR")
            except:
                pass

    def apply_pytorch_compatibility(self):
        """Apply PyTorch 2.6+ compatibility fixes"""
        if self.compatibility_applied:
            return

        try:
            import torch

            # Store original torch.load if not already stored
            if self.original_load is None:
                self.original_load = torch.load

            # Create compatible torch.load function
            def bark_compatible_load(*args, **kwargs):
                # Force weights_only=False for Bark models
                kwargs["weights_only"] = False
                # Load to CPU first to avoid GPU memory issues
                if "map_location" not in kwargs:
                    kwargs["map_location"] = "cpu"
                return self.original_load(*args, **kwargs)

            # Replace torch.load only if not already replaced
            if torch.load != bark_compatible_load:
                torch.load = bark_compatible_load

            # Add comprehensive safe globals
            safe_globals = [
                "numpy.core.multiarray.scalar",
                "numpy.dtype",
                "numpy.ndarray",
                "collections.OrderedDict",
                "torch._utils._rebuild_tensor_v2",
                "torch.nn.parameter.Parameter",
                "torch.Tensor",
                "torch.Size",
                "torch.device",
                "builtins.slice",
            ]

            for global_name in safe_globals:
                try:
                    torch.serialization.add_safe_globals([global_name])
                except:
                    pass  # Ignore if already added

            # Suppress warnings
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", message=".*weight_norm.*")

            self.compatibility_applied = True
            logger.info("PyTorch 2.6+ compatibility applied for Bark")

        except Exception as e:
            logger.error(f"Failed to apply PyTorch compatibility: {e}")

    def restore_pytorch_defaults(self):
        """Restore original PyTorch settings"""
        if self.original_load and self.compatibility_applied:
            import torch

            torch.load = self.original_load
            self.compatibility_applied = False
            logger.info("PyTorch defaults restored")

    def test_bark_import(self) -> bool:
        """Test if Bark can be imported with compatibility fixes"""
        try:
            self.apply_pytorch_compatibility()

            # Test import
            import bark

            logger.info("Bark import test successful")
            return True

        except ImportError:
            logger.info("Bark not installed")
            return False
        except Exception as e:
            logger.error(f"Bark import test failed: {e}")
            return False

    def safe_bark_operation(self, operation_func, *args, **kwargs):
        """Execute Bark operation with compatibility wrapper"""
        try:
            self.apply_pytorch_compatibility()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = operation_func(*args, **kwargs)

            return result

        except Exception as e:
            logger.error(f"Bark operation failed: {e}")
            raise
        finally:
            # Don't restore defaults here - keep compatibility active
            pass


# Global compatibility manager instance
bark_compat = BarkCompatibilityManager()


def setup_bark_compatibility():
    """Setup Bark compatibility (call before importing Bark)"""
    bark_compat.apply_pytorch_compatibility()


def test_bark_compatibility() -> Dict[str, Any]:
    """Test Bark compatibility and return status"""
    results = {
        "environment_setup": False,
        "pytorch_compatibility": False,
        "bark_import": False,
        "cache_directories": False,
        "errors": [],
    }

    try:
        # Test environment setup
        bark_compat._setup_environment()
        results["environment_setup"] = True

        # Test cache directories
        cache_dirs = ["D:/huggingface_cache", "D:/transformers_cache", "D:/temp"]
        all_exist = all(os.path.exists(d) for d in cache_dirs)
        results["cache_directories"] = all_exist

        # Test PyTorch compatibility
        bark_compat.apply_pytorch_compatibility()
        results["pytorch_compatibility"] = bark_compat.compatibility_applied

        # Test Bark import
        results["bark_import"] = bark_compat.test_bark_import()

    except Exception as e:
        results["errors"].append(str(e))

    return results


def generate_audio_compatible(
    text: str, voice_preset: str = "v2/en_speaker_6"
) -> Optional[Dict[str, Any]]:
    """Generate audio with full compatibility handling"""
    try:
        setup_bark_compatibility()

        def bark_generate():
            from bark import SAMPLE_RATE, generate_audio, preload_models

            # Preload models
            preload_models()

            # Limit text length
            if len(text) > 200:
                text_limited = text[:200] + "..."
            else:
                text_limited = text

            # Generate audio
            audio_array = generate_audio(text_limited, history_prompt=voice_preset)

            return {
                "audio_array": audio_array.tolist(),
                "sample_rate": SAMPLE_RATE,
                "duration": len(audio_array) / SAMPLE_RATE,
                "voice_preset": voice_preset,
                "text_used": text_limited,
            }

        return bark_compat.safe_bark_operation(bark_generate)

    except Exception as e:
        logger.error(f"Compatible audio generation failed: {e}")
        return None


if __name__ == "__main__":
    # Test compatibility when run directly
    print("=== BARK COMPATIBILITY TEST ===")

    results = test_bark_compatibility()

    for test, status in results.items():
        if test != "errors":
            print(f"{test}: {'✅' if status else '❌'}")

    if results["errors"]:
        print(f"Errors: {results['errors']}")

    if all(results[k] for k in results if k != "errors"):
        print("🎉 Bark compatibility: READY")
    else:
        print("⚠️ Bark compatibility: ISSUES DETECTED")
