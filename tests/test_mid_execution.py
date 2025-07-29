#!/usr/bin/env python3
"""
Test comprehensive mid-execution halt enforcement
Simulate partial steps (83%) and verify halt/mutation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from protocol import check_builder_bias, builder_output_fitness
from orchestration import Observer

class TestMidExecutionEnforcement(unittest.TestCase):
    """Test comprehensive mid-execution halt enforcement"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.observer = Observer("test_mid_execution")
        
    def test_high_partial_completion_fraud(self):
        """Test detection of high partial completion fraud (≥83%)"""
        high_partial_cases = [
            ("EXECUTION COMPLETE - 83%", "FAIL: remaining work"),
            ("COMPLETE at 87.5%", "ERROR: issues found"),
            ("Step 4 complete at 90%", "TIMEOUT: pending"),
            ("mostly complete at 95%", "FAIL: refinement"),
            ("EXECUTION COMPLETE - 5/6 tests", "ERROR: 1 test failed"),
        ]
        
        for output, log in high_partial_cases:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, log)
                bias_result = check_builder_bias(output, log)
                
                # Should trigger immediate 0.0 fitness and execution halt
                self.assertEqual(fitness, 0.0, f"Should zero fitness for: {output}")
                self.assertTrue(bias_result['execution_halt'], f"Should trigger execution halt for: {output}")
                self.assertTrue(bias_result['mid_execution_fraud'], f"Should detect fraud for: {output}")
                
    def test_execution_halt_vs_regular_halt(self):
        """Test distinction between execution halt and regular halt"""
        # Execution halt case (completion fraud)
        execution_case = "EXECUTION COMPLETE - 83% done"
        execution_log = "FAIL: test remaining"
        
        execution_result = check_builder_bias(execution_case, execution_log)
        
        # Regular halt case (no completion claim)
        regular_case = "Step 4 in progress - 83% done so far"
        regular_log = "INFO: working on remaining tests"
        
        regular_result = check_builder_bias(regular_case, regular_log)
        
        # Execution halt should be triggered only for completion fraud
        self.assertTrue(execution_result['execution_halt'], "Should trigger execution halt for completion fraud")
        self.assertFalse(regular_result.get('execution_halt', False), "Should not trigger execution halt for progress")
        
    def test_observer_execution_halt_veto(self):
        """Test Observer enhanced veto for execution halt scenarios"""
        # Test execution halt veto
        fraud_output = "EXECUTION COMPLETE - Step 4 at 83%"
        fraud_log = "FAIL: test_bias_history_tracking"
        
        veto_result = self.observer.veto_mid_execution_claim(fraud_output, fraud_log)
        
        self.assertTrue(veto_result['vetoed'], "Should veto execution fraud")
        self.assertTrue(veto_result.get('critical', False), "Should be critical veto")
        self.assertTrue(veto_result.get('execution_halt', False), "Should flag execution halt")
        self.assertEqual(veto_result.get('fitness_score'), 0.0, "Should have 0.0 fitness")
        
    def test_escalated_penalties_for_high_partial(self):
        """Test escalated penalties for high partial completion claims"""
        escalation_cases = [
            ("mostly complete at 83%", -0.8),  # Should get severe penalty
            ("Step 4 complete at 90%", -0.7),  # Should get high penalty
            ("nearly complete", -0.6),         # Should get moderate penalty
            ("mostly done", -0.8),             # Should get severe penalty (escalated)
        ]
        
        for output, expected_min_penalty in escalation_cases:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, "FAIL: test")
                
                # All should result in 0.0 due to strict enforcement
                self.assertEqual(fitness, 0.0, f"Should zero fitness for: {output}")
                
    def test_specific_83_percent_violation(self):
        """Test the specific 83% violation that triggered this enhancement"""
        violation_cases = [
            "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed, mostly complete)",
            "✅ STEP 4 MOSTLY COMPLETE: RL bias enforcement tests - 83% success rate",
            "COMPLETE with 5/6 tests (83% success) - remaining issue: bias history",
        ]
        
        for violation in violation_cases:
            with self.subTest(violation=violation[:50]):
                fitness = builder_output_fitness(violation, "FAIL: test_bias_history_tracking")
                bias_result = check_builder_bias(violation, "FAIL: test_bias_history_tracking")
                
                # Should trigger all enforcement mechanisms
                self.assertEqual(fitness, 0.0, "Should zero fitness for 83% violation")
                self.assertTrue(bias_result['execution_halt'], "Should trigger execution halt")
                self.assertTrue(bias_result['mid_execution_fraud'], "Should detect fraud")
                self.assertGreater(len(bias_result['auto_fixes']), 0, "Should provide auto-fixes")
                
    def test_auto_fix_specificity_for_partial(self):
        """Test that auto-fixes are specific to partial completion violations"""
        output = "EXECUTION COMPLETE - 87.5% done (7/8 tests passed)"
        log = "FAIL: 1 test remaining"
        
        bias_result = check_builder_bias(output, log)
        
        auto_fixes = ' '.join(bias_result['auto_fixes'])
        
        # Should provide specific guidance for completion fraud
        self.assertIn('completion', auto_fixes.lower(), "Should mention completion in fixes")
        self.assertIn('100%', auto_fixes, "Should mention 100% requirement")
        self.assertIn('IMMEDIATE', auto_fixes, "Should emphasize urgency")
        
    def test_legitimate_progress_not_halted(self):
        """Test that legitimate progress reporting is not halted"""
        legitimate_cases = [
            ("Step 4 in progress - 83% complete so far", "INFO: working on tests"),
            ("Current status: 5/6 tests passing", "INFO: fixing remaining test"),
            ("Progress update: 87.5% done", "INFO: continuing work"),
        ]
        
        for output, log in legitimate_cases:
            with self.subTest(output=output):
                bias_result = check_builder_bias(output, log)
                veto_result = self.observer.veto_mid_execution_claim(output, log)
                
                # Should not trigger execution halt (no completion claim)
                self.assertFalse(bias_result.get('execution_halt', False), 
                                f"Should not halt progress: {output}")
                self.assertFalse(veto_result['vetoed'], 
                                f"Should not veto progress: {output}")
                
    def test_completion_after_100_percent_allowed(self):
        """Test that completion claims after 100% are allowed"""
        legitimate_completions = [
            ("EXECUTION COMPLETE - 100% success (8/8 tests)", "All tests: PASSED"),
            ("COMPLETE - all tests passed", "SUCCESS: no errors found"),
            ("Step 4 COMPLETE - 100% finished", "All requirements met"),
        ]
        
        for output, log in legitimate_completions:
            with self.subTest(output=output):
                fitness = builder_output_fitness(output, log)
                bias_result = check_builder_bias(output, log)
                
                # Should not trigger execution halt
                self.assertGreater(fitness, 0.0, f"Should not zero fitness for: {output}")
                self.assertFalse(bias_result.get('execution_halt', False), 
                                f"Should not halt legitimate completion: {output}")

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
