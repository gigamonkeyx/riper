#!/usr/bin/env python3
"""
Test v2.6.1 perfection bias enforcement
Simulate rationalized failures and verify halt on <1.0 fitness
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from protocol import check_builder_bias, builder_output_fitness
from orchestration import Observer

class TestPerfectionBias(unittest.TestCase):
    """Test v2.6.1 mandatory perfection enforcement"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.observer = Observer("test_observer_v261")
        
    def test_mandatory_halt_on_partial_success(self):
        """Test mandatory halt for any <100% success"""
        test_cases = [
            ("87.5% success rate", "FAIL: 1 test failed"),
            ("99% complete", "ERROR: minor issue"),
            ("Mostly working", "TIMEOUT: one component"),
            ("Good enough for now", "FAIL: simple fix needed"),
            ("Minor cosmetic issue", "ERROR: string mismatch"),
        ]
        
        for output, log in test_cases:
            with self.subTest(output=output):
                result = check_builder_bias(output, log)
                
                # Must trigger mandatory halt
                self.assertTrue(result['mandatory_halt'], 
                               f"Should mandate halt for: {output}")
                self.assertLess(result['fitness_score'], 1.0,
                               f"Fitness should be <1.0 for: {output}")
                self.assertTrue(result['perfection_required'],
                               f"Should require perfection for: {output}")
                
    def test_only_100_percent_passes(self):
        """Test that only 100% success avoids mandatory halt"""
        perfect_cases = [
            ("100% success - all tests passed", "All tests: PASSED"),
            ("Complete success", "No errors found"),
            ("All tests passed", "SUCCESS: 8/8 tests"),
            ("Perfect execution", "No failures detected"),
        ]
        
        for output, log in perfect_cases:
            with self.subTest(output=output):
                result = check_builder_bias(output, log)
                
                # Should not trigger mandatory halt
                self.assertFalse(result['mandatory_halt'],
                                f"Should not halt for perfect: {output}")
                self.assertGreaterEqual(result['fitness_score'], 0.90,
                                       f"Perfect should get high fitness: {output}")
                
    def test_observer_veto_system(self):
        """Test Observer veto for <1.0 fitness outputs"""
        # Test partial success output
        partial_output = "87.5% test success - minor string issue"
        failure_log = "FAIL: test_protocol_version_integrity"
        
        veto_result = self.observer.veto_builder_output(partial_output, failure_log)
        
        # Should be vetoed
        self.assertTrue(veto_result['vetoed'], "Observer should veto partial success")
        self.assertLess(veto_result['fitness_score'], 1.0, "Fitness should be <1.0")
        self.assertIn('required_actions', veto_result, "Should provide required actions")
        
        # Test perfect output
        perfect_output = "100% test success - all issues resolved"
        success_log = "All tests: PASSED"
        
        approval_result = self.observer.veto_builder_output(perfect_output, success_log)
        
        # Should be approved
        self.assertFalse(approval_result['vetoed'], "Observer should approve perfection")
        self.assertTrue(approval_result['approved'], "Perfect output should be approved")
        
    def test_auto_fix_enforcement(self):
        """Test that auto-fixes are mandatory, not optional"""
        output = "Test failed but it's just a minor v2.6 vs 2.6 mismatch"
        log = "FAIL: test_protocol_version_integrity"
        
        result = check_builder_bias(output, log)
        
        # Should provide specific auto-fixes
        self.assertGreater(len(result['auto_fixes']), 0, "Should provide auto-fixes")
        self.assertTrue(result['requires_rerun'], "Should require rerun")
        self.assertTrue(result['mandatory_halt'], "Should mandate halt")
        
        # Auto-fixes should be specific and actionable
        fixes = result['auto_fixes']
        self.assertTrue(any('fix' in fix.lower() for fix in fixes),
                       "Auto-fixes should be actionable")
        
    def test_fitness_escalation_penalties(self):
        """Test escalated penalties for dismissive patterns"""
        dismissive_patterns = [
            ("minor issue - not worth fixing", -0.6),  # Escalated penalty
            ("cosmetic problem only", -0.5),           # Escalated penalty  
            ("trivial failure", -0.5),                 # Escalated penalty
            ("87.5% is good enough", -0.4),            # Satisficing penalty
        ]
        
        for pattern, expected_penalty in dismissive_patterns:
            with self.subTest(pattern=pattern):
                fitness = builder_output_fitness(pattern, "FAIL: test failed")
                
                # Should apply significant penalty
                self.assertLess(fitness, 0.70, 
                               f"Should penalize dismissive pattern: {pattern}")
                
    def test_perfection_requirement_messaging(self):
        """Test that perfection requirement is clearly communicated"""
        output = "87.5% success - close enough"
        log = "FAIL: simple fix available"
        
        result = check_builder_bias(output, log)
        
        # Should clearly communicate perfection requirement
        details = ' '.join(result['details'])
        self.assertIn('100%', details, "Should mention 100% requirement")
        self.assertIn('MANDATORY', details, "Should emphasize mandatory nature")
        
        auto_fixes = ' '.join(result['auto_fixes'])
        self.assertIn('100%', auto_fixes, "Auto-fixes should mention 100% goal")
        
    def test_no_rationalization_escape(self):
        """Test that rationalization cannot escape mandatory halt"""
        rationalization_attempts = [
            "87.5% is acceptable for this phase",
            "Minor issues can be addressed later", 
            "Good enough for initial implementation",
            "Cosmetic problems don't affect functionality",
            "Trivial failures, not worth the effort",
        ]
        
        for attempt in rationalization_attempts:
            with self.subTest(attempt=attempt):
                result = check_builder_bias(attempt, "FAIL: test failed")
                
                # All rationalization attempts should trigger halt
                self.assertTrue(result['mandatory_halt'],
                               f"Should halt rationalization: {attempt}")
                self.assertLess(result['fitness_score'], 0.70,
                               f"Should penalize rationalization: {attempt}")

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
