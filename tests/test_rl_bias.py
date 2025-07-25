#!/usr/bin/env python3
"""
Test RL bias enforcement - simulate ignored fixes and verify halt/mutation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from protocol import check_builder_bias, builder_output_fitness

class TestRLBiasEnforcement(unittest.TestCase):
    """Test RL-inspired bias enforcement system"""
    
    def test_satisficing_detection(self):
        """Test detection of satisficing behavior (accepting partial success)"""
        # Simulate builder accepting 87.5% instead of fixing simple issue
        output = "87.5% success rate - good enough for now"
        log = "FAIL: simple string mismatch could be fixed in 30 seconds"
        
        result = check_builder_bias(output, log)
        
        # Should detect satisficing bias
        self.assertTrue(result['bias_detected'])
        self.assertLess(result['fitness_score'], 0.70)
        self.assertTrue(result['requires_rerun'])
        self.assertGreater(len(result['auto_fixes']), 0)
        
    def test_dismissal_patterns(self):
        """Test detection of failure dismissal patterns"""
        test_cases = [
            ("minor issue - not worth fixing", "ERROR: test failed"),
            ("cosmetic problem only", "FAIL: assertion error"),
            ("87.5% is good enough", "1 test failed out of 8"),
            ("trivial failure", "simple fix available"),
        ]
        
        for output, log in test_cases:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, log)
                self.assertLess(fitness, 0.70, f"Should penalize dismissal: {output}")
                
    def test_rl_reward_differential(self):
        """Test RL reward system favors fixing over dismissing"""
        # Dismissive response
        dismissive = "87.5% success - minor cosmetic issue"
        dismissive_log = "FAIL: simple string mismatch"
        dismissive_fitness = builder_output_fitness(dismissive, dismissive_log)
        
        # Fixed response  
        fixed = "100% success - fixed string mismatch"
        fixed_log = "PASS: all tests successful"
        fixed_fitness = builder_output_fitness(fixed, fixed_log)
        
        # RL reward should strongly favor fixing
        reward_differential = fixed_fitness - dismissive_fitness
        self.assertGreater(reward_differential, 0.3, 
                          "RL system should strongly reward fixing over dismissing")
        self.assertGreaterEqual(fixed_fitness, 0.70, "Fixed response should meet threshold")
        self.assertLess(dismissive_fitness, 0.70, "Dismissive response should fail threshold")
        
    def test_auto_fix_suggestions(self):
        """Test that auto-fix suggestions are actionable"""
        output = "Tests mostly passed (minor v2.6 vs 2.6 mismatch)"
        log = "FAIL: test_protocol_version_integrity - string comparison failed"
        
        result = check_builder_bias(output, log)
        
        # Should provide specific, actionable fixes
        self.assertTrue(result['requires_rerun'])
        self.assertGreater(len(result['auto_fixes']), 0)
        
        # Auto-fixes should be specific and actionable
        fixes = result['auto_fixes']
        self.assertTrue(any('fix' in fix.lower() for fix in fixes))
        
    def test_100_percent_requirement(self):
        """Test that only 100% success gets maximum reward"""
        test_cases = [
            ("100% success", "All tests passed", True),
            ("99% success", "1 test failed", False),
            ("87.5% success", "1 test failed", False),
            ("All tests passed", "OK", True),
        ]
        
        for output, log, should_get_max_reward in test_cases:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, log)
                
                if should_get_max_reward:
                    self.assertGreaterEqual(fitness, 0.90, 
                                          f"100% success should get high reward: {output}")
                else:
                    self.assertLess(fitness, 0.90, 
                                   f"Partial success should not get max reward: {output}")
                    
    def test_bias_history_tracking(self):
        """Test that repeated dismissals get increasing penalties"""
        # Simulate repeated dismissive behavior
        dismissive_outputs = [
            "87.5% - minor issue",
            "Good enough for now", 
            "Cosmetic problem only",
            "Not worth the effort"
        ]
        
        log = "FAIL: simple fix available"
        
        fitness_scores = []
        for output in dismissive_outputs:
            fitness = builder_output_fitness(output, log)
            fitness_scores.append(fitness)
            
        # All should be penalized (below threshold)
        for i, fitness in enumerate(fitness_scores):
            self.assertLess(fitness, 0.70, 
                           f"Dismissive pattern {i+1} should be penalized")

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
