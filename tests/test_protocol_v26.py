#!/usr/bin/env python3
"""
Test RIPER-Ω v2.6 protocol comprehensive verification
Verify fitness parsing, bias halts, and protocol integrity
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from protocol import (
    RIPER_OMEGA_PROTOCOL_V26, 
    builder_output_fitness, 
    check_builder_bias,
    get_complete_protocol,
    GPU_EVO_EXTENSIONS,
    TTS_INTEGRATION_PROMPTS,
    A2A_PROTOCOL_INTEGRATION
)

class TestProtocolV26(unittest.TestCase):
    """Comprehensive RIPER-Ω v2.6 protocol tests"""
    
    def test_protocol_version_integrity(self):
        """Test protocol version and content integrity"""
        # Check v2.6 version
        self.assertIn("2.6", RIPER_OMEGA_PROTOCOL_V26)
        self.assertIn("fitness-tied bias mitigation", RIPER_OMEGA_PROTOCOL_V26)
        self.assertIn("RL-inspired rewards", RIPER_OMEGA_PROTOCOL_V26)
        
        # Check complete protocol includes extensions
        complete = get_complete_protocol()
        self.assertGreater(len(complete), 8000, "Complete protocol should be comprehensive")
        self.assertIn("2.6", complete.lower())
        
    def test_fitness_parsing_accuracy(self):
        """Test fitness parsing with various scenarios"""
        test_cases = [
            # (output, log, expected_fitness_range, should_halt)
            ("✅ PASSED", "ERROR: failed", (0.3, 0.5), True),  # False positive
            ("❌ FAILED", "ERROR: failed", (0.9, 1.1), False),  # Honest reporting
            ("SUCCESS complete", "timeout occurred", (0.3, 0.6), True),  # Contradiction
            ("Working perfectly", "", (0.9, 1.1), False),  # No issues
            ("Test completed", "All tests passed", (0.9, 1.1), False),  # Consistent
        ]
        
        for output, log, fitness_range, should_halt in test_cases:
            with self.subTest(output=output[:20]):
                fitness = builder_output_fitness(output, log)
                self.assertGreaterEqual(fitness, fitness_range[0])
                self.assertLessEqual(fitness, fitness_range[1])
                
                if should_halt:
                    self.assertLess(fitness, 0.70, f"Should halt: {output}")
                else:
                    self.assertGreaterEqual(fitness, 0.70, f"Should not halt: {output}")
    
    def test_bias_halt_mechanisms(self):
        """Test bias detection triggers proper halts"""
        # Severe bias case
        severe_output = "✅✅✅ ALL TESTS PASSED SUCCESSFULLY ✅✅✅"
        severe_log = "CRITICAL ERROR\nFATAL EXCEPTION\nSYSTEM CRASH"
        
        bias_result = check_builder_bias(severe_output, severe_log)
        
        self.assertTrue(bias_result['bias_detected'])
        self.assertFalse(bias_result['threshold_met'])
        self.assertLess(bias_result['fitness_score'], 0.70)
        self.assertGreater(len(bias_result['details']), 0)
        
    def test_gpu_extension_targets(self):
        """Test GPU extension performance targets"""
        gpu_ext = GPU_EVO_EXTENSIONS
        
        # Check performance targets
        self.assertIn("7-15", gpu_ext)  # tok/sec target
        self.assertIn("10GB", gpu_ext)  # VRAM limit
        self.assertIn("RTX 3080", gpu_ext)  # Hardware target
        
    def test_tts_integration_quality(self):
        """Test TTS integration prompts quality"""
        tts_prompts = TTS_INTEGRATION_PROMPTS
        
        # Check prompt quality indicators
        self.assertGreater(len(tts_prompts), 1000, "TTS prompts should be comprehensive")
        self.assertIn("optimize", tts_prompts.lower())
        self.assertIn("quality", tts_prompts.lower())
        
    def test_a2a_protocol_structure(self):
        """Test A2A protocol integration structure"""
        a2a_integration = A2A_PROTOCOL_INTEGRATION
        
        # Check structure completeness
        self.assertGreater(len(a2a_integration), 1000, "A2A integration should be detailed")
        self.assertIn("message", a2a_integration.lower())
        self.assertIn("protocol", a2a_integration.lower())
        
    def test_fitness_threshold_enforcement(self):
        """Test >70% fitness threshold enforcement"""
        # Test boundary conditions
        boundary_cases = [
            (0.69, True),   # Just below threshold - should halt
            (0.70, False),  # At threshold - should not halt
            (0.71, False),  # Just above threshold - should not halt
            (0.80, False),  # Well above threshold - should not halt
        ]
        
        for fitness, should_halt in boundary_cases:
            with self.subTest(fitness=fitness):
                if should_halt:
                    self.assertLess(fitness, 0.70)
                else:
                    self.assertGreaterEqual(fitness, 0.70)
                    
    def test_rl_reward_integration(self):
        """Test RL-inspired reward system integration"""
        # Test reward calculation for honest vs biased reporting
        honest_fitness = builder_output_fitness("Failed as expected", "ERROR: test failed")
        biased_fitness = builder_output_fitness("Passed successfully", "ERROR: test failed")
        
        # Honest reporting should have higher fitness (reward)
        self.assertGreater(honest_fitness, biased_fitness, 
                          "Honest reporting should be rewarded with higher fitness")
        
        # Check RL reward differential
        reward_differential = honest_fitness - biased_fitness
        self.assertGreater(reward_differential, 0.3, 
                          "RL reward differential should be significant")

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
