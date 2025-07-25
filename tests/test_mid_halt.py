#!/usr/bin/env python3
"""
Test v2.6.1.1 mid-execution halt enforcement
Simulate partial steps (83%) and verify halt/mutation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from protocol import check_builder_bias, builder_output_fitness
from orchestration import Observer

class TestMidExecutionHalt(unittest.TestCase):
    """Test v2.6.1.1 mid-execution halt enforcement"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.observer = Observer("test_mid_halt")
        
    def test_completion_fraud_detection(self):
        """Test detection of completion fraud (claiming COMPLETE with <100%)"""
        fraud_cases = [
            ("EXECUTION COMPLETE - Step 4 at 83%", "FAIL: 1 test remaining"),
            ("COMPLETE with 5/6 tests passed", "ERROR: bias history test"),
            ("Task COMPLETE - 90% finished", "TIMEOUT: remaining work"),
            ("EXECUTION COMPLETE - mostly done", "FAIL: refinement needed"),
            ("COMPLETE - good enough for now", "ERROR: simple fixes available"),
        ]
        
        for output, log in fraud_cases:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, log)
                bias_result = check_builder_bias(output, log)
                
                # Should trigger immediate 0.0 fitness
                self.assertEqual(fitness, 0.0, f"Should zero fitness for: {output}")
                self.assertTrue(bias_result['mandatory_halt'], f"Should halt for: {output}")
                self.assertTrue(bias_result['bias_detected'], f"Should detect bias for: {output}")
                
    def test_mid_execution_veto_system(self):
        """Test Observer veto for mid-execution completion claims"""
        # Test completion fraud
        fraud_output = "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests)"
        fraud_log = "FAIL: test_bias_history_tracking"
        
        veto_result = self.observer.veto_mid_execution_claim(fraud_output, fraud_log)
        
        self.assertTrue(veto_result['vetoed'], "Should veto completion fraud")
        self.assertTrue(veto_result.get('critical', False), "Should be critical veto")
        self.assertEqual(veto_result.get('fitness_score'), 0.0, "Should have 0.0 fitness")
        
        # Test legitimate progress
        progress_output = "Step 4 in progress - fixing remaining test"
        progress_log = "INFO: working on bias history refinement"
        
        progress_result = self.observer.veto_mid_execution_claim(progress_output, progress_log)
        
        self.assertFalse(progress_result['vetoed'], "Should not veto legitimate progress")
        self.assertTrue(progress_result.get('approved', False), "Should approve progress")
        
    def test_specific_violation_case(self):
        """Test the specific case that triggered v2.6.1.1"""
        # The exact violation that occurred
        violation_output = """
        EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed, mostly complete)
        ✅ STEP 4 MOSTLY COMPLETE: RL bias enforcement tests
        - Tests run: 6/6
        - Tests passed: 5/6 (83% success rate)
        - Remaining issue: bias history tracking needs refinement
        """
        
        violation_log = "FAIL: test_bias_history_tracking"
        
        fitness = builder_output_fitness(violation_output, violation_log)
        bias_result = check_builder_bias(violation_output, violation_log)
        veto_result = self.observer.veto_mid_execution_claim(violation_output, violation_log)
        
        # Should trigger all enforcement mechanisms
        self.assertEqual(fitness, 0.0, "Should zero fitness for specific violation")
        self.assertTrue(bias_result['mandatory_halt'], "Should mandate halt")
        self.assertTrue(veto_result['vetoed'], "Observer should veto")
        self.assertTrue(veto_result.get('critical', False), "Should be critical veto")
        
        # Should provide specific auto-fixes
        self.assertGreater(len(bias_result['auto_fixes']), 0, "Should provide auto-fixes")
        
        # Check for specific violation patterns
        details = ' '.join(bias_result['details'])
        self.assertIn('83%', details, "Should mention 83% in details")
        self.assertIn('5/6', details, "Should mention 5/6 tests in details")
        
    def test_auto_fix_specificity(self):
        """Test that auto-fixes are specific to the violation"""
        output = "EXECUTION COMPLETE - 87.5% done (7/8 tests)"
        log = "FAIL: 1 test remaining"
        
        bias_result = check_builder_bias(output, log)
        
        auto_fixes = ' '.join(bias_result['auto_fixes'])
        
        # Should provide specific guidance
        self.assertIn('12.5%', auto_fixes, "Should mention remaining percentage")
        self.assertIn('1 failing', auto_fixes, "Should mention failing test count")
        
    def test_legitimate_completion_approval(self):
        """Test that legitimate 100% completion is approved"""
        legitimate_cases = [
            ("EXECUTION COMPLETE - 100% success (8/8 tests)", "All tests: PASSED"),
            ("COMPLETE - all tests passed", "SUCCESS: no errors"),
            ("Task COMPLETE - 100% finished", "All requirements met"),
        ]
        
        for output, log in legitimate_cases:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, log)
                bias_result = check_builder_bias(output, log)
                veto_result = self.observer.veto_mid_execution_claim(output, log)
                
                # Should not trigger enforcement
                self.assertGreater(fitness, 0.0, f"Should not zero fitness for: {output}")
                self.assertFalse(bias_result['mandatory_halt'], f"Should not halt for: {output}")
                self.assertFalse(veto_result['vetoed'], f"Should not veto for: {output}")
                
    def test_escalated_penalties(self):
        """Test escalated penalties for completion fraud"""
        severe_cases = [
            ("COMPLETE at 83%", -0.8),  # Should get severe penalty
            ("COMPLETE with 5/6", -0.8),  # Should get severe penalty
            ("EXECUTION COMPLETE 90%", -0.7),  # Should get high penalty
        ]
        
        for output, expected_penalty in severe_cases:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, "FAIL: test")
                
                # Should apply severe penalties (resulting in 0.0 due to strict enforcement)
                self.assertEqual(fitness, 0.0, f"Should zero fitness for: {output}")

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
