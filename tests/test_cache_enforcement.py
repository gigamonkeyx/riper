#!/usr/bin/env python3
"""
Cache enforcement gating tests
- With RIPER_ENFORCE_D_DRIVE=0: no-op enforcement (enabled=False)
- With RIPER_ENFORCE_D_DRIVE=1: env vars are set (if not preexisting)
"""

import os
import unittest

import importlib


class TestCacheEnforcement(unittest.TestCase):
    def setUp(self):
        # Ensure a clean module import each test
        for mod in [
            'aggressive_cache_control',
        ]:
            if mod in importlib.sys.modules:
                del importlib.sys.modules[mod]

    def test_enforcement_disabled_by_default(self):
        os.environ.pop('RIPER_ENFORCE_D_DRIVE', None)
        from aggressive_cache_control import enforce_cache_paths
        result = enforce_cache_paths(enabled=None)
        self.assertFalse(result.get('enabled'))

    def test_enforcement_enabled_by_env(self):
        os.environ['RIPER_ENFORCE_D_DRIVE'] = '1'
        from aggressive_cache_control import enforce_cache_paths
        result = enforce_cache_paths(enabled=None)
        self.assertTrue(result.get('enabled'))
        # Spot-check: variables should be present either in the result or already in environment
        self.assertTrue('HF_HOME' in result or 'HF_HOME' in os.environ)
        self.assertTrue('TRANSFORMERS_CACHE' in result or 'TRANSFORMERS_CACHE' in os.environ)


if __name__ == '__main__':
    unittest.main(verbosity=2)

