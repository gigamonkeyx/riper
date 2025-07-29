#!/usr/bin/env python3
"""
Test escalation trigger system
Simulate multiple low scores and verify halt/report
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from protocol import low_fitness_trigger, check_builder_bias, PROTOCOL_METADATA
from orchestration import Observer

class TestEscalationTrigger(unittest.TestCase):
    """Test escalation trigger system for multiple low fitness scores"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Clear fitness history before each test
        PROTOCOL_METADATA["fitness_history"] = []
        self.observer = Observer("test_escalation")
        
    def test_escalation_threshold(self):
        """Test escalation trigger at threshold (3 low scores)"""
        low_scores = [0.60, 0.45, 0.30]  # 3 scores <0.70
        
        for i, score in enumerate(low_scores):
            result = low_fitness_trigger(score, f"test output {i+1}", "FAIL: test")
            
            if i < 2:  # First 2 scores should not trigger
                self.assertFalse(result['trigger_activated'], 
                                f"Should not trigger on score {i+1}")
            else:  # 3rd score should trigger
                self.assertTrue(result['trigger_activated'], 
                               f"Should trigger on score {i+1}")
                self.assertTrue(result['halt_required'], "Should require halt")
                self.assertTrue(result['consultation_required'], "Should require consultation")
                
    def test_issues_report_generation(self):
        """Test issues report generation on escalation"""
        # Trigger escalation with 3 low scores
        low_scores = [0.60, 0.45, 0.30]
        
        for score in low_scores:
            result = low_fitness_trigger(score, "test output", "FAIL: test")
        
        # Last result should have issues report
        self.assertIsNotNone(result['issues_report'], "Should generate issues report")
        
        report = result['issues_report']
        self.assertEqual(report['low_score_count'], 3, "Should count 3 low scores")
        self.assertEqual(report['threshold'], 3, "Should use threshold of 3")
        self.assertTrue(report['consultation_required'], "Should require consultation")
        self.assertTrue(report['halt_builder'], "Should halt builder")
        self.assertGreater(len(report['bias_patterns']), 0, "Should identify bias patterns")
        self.assertGreater(len(report['recommended_actions']), 0, "Should provide actions")
        
    def test_bias_pattern_analysis(self):
        """Test bias pattern analysis in issues report"""
        # Test different fitness levels for pattern analysis
        test_cases = [
            (0.00, "Critical bias: Zero fitness"),
            (0.25, "Severe bias: Multiple false positive"),
            (0.55, "Moderate bias: Dismissive language")
        ]
        
        for fitness, expected_pattern in test_cases:
            with self.subTest(fitness=fitness):
                # Clear history and trigger with this fitness level
                PROTOCOL_METADATA["fitness_history"] = []
                
                # Add 3 scores to trigger escalation
                for _ in range(3):
                    result = low_fitness_trigger(fitness, "test", "FAIL")
                
                # Check pattern analysis
                patterns = result['issues_report']['bias_patterns']
                pattern_text = ' '.join(patterns)
                self.assertIn(expected_pattern, pattern_text, 
                             f"Should detect pattern for fitness {fitness}")
                
    def test_integration_with_bias_checking(self):
        """Test integration of escalation with bias checking"""
        # Clear history
        PROTOCOL_METADATA["fitness_history"] = []
        
        # Simulate sequence of bias checks leading to escalation
        bias_outputs = [
            ("87.5% success - minor issue", "FAIL: test"),
            ("mostly complete", "ERROR: remaining"),
            ("good enough for now", "TIMEOUT: pending")
        ]
        
        escalation_triggered = False
        for output, log in bias_outputs:
            result = check_builder_bias(output, log)
            
            if result.get('escalation_triggered', False):
                escalation_triggered = True
                self.assertTrue(result['consultation_required'], "Should require consultation")
                self.assertIsNotNone(result['issues_report'], "Should have issues report")
                break
        
        self.assertTrue(escalation_triggered, "Should trigger escalation through bias checking")
        
    def test_observer_consultation_integration(self):
        """Test observer consultation integration with escalation"""
        # Generate escalation report
        PROTOCOL_METADATA["fitness_history"] = []
        for _ in range(3):
            result = low_fitness_trigger(0.30, "test", "FAIL")
        
        issues_report = result['issues_report']
        
        # Test observer consultation
        consultation_result = self.observer.receive_issues_report(issues_report, "test_builder")
        
        self.assertTrue(consultation_result['consultation_provided'], "Should provide consultation")
        self.assertTrue(consultation_result['builder_halt_confirmed'], "Should confirm halt")
        self.assertFalse(consultation_result['resume_authorized'], "Should not authorize resume")
        
        response = consultation_result['consultation_response']
        self.assertIn(response['issue_severity'], ['MODERATE', 'HIGH'], "Should assess severity")
        self.assertGreater(len(response['observer_analysis']), 0, "Should provide analysis")
        self.assertGreater(len(response['recommended_fixes']), 0, "Should provide fixes")
        self.assertGreater(len(response['resume_conditions']), 0, "Should set conditions")
        
    def test_fitness_history_management(self):
        """Test fitness history management (max 10 entries)"""
        # Add more than 10 scores
        for i in range(15):
            low_fitness_trigger(0.50, f"test {i}", "FAIL")
        
        # Should keep only last 10
        self.assertLessEqual(len(PROTOCOL_METADATA["fitness_history"]), 10, 
                            "Should limit history to 10 entries")
        
    def test_no_false_escalation(self):
        """Test that high fitness scores don't trigger escalation"""
        # Add high fitness scores
        high_scores = [0.85, 0.90, 0.95, 1.00]
        
        for score in high_scores:
            result = low_fitness_trigger(score, "good output", "SUCCESS")
            self.assertFalse(result['trigger_activated'], 
                            f"Should not trigger on high score {score}")
            
    def test_mixed_scores_escalation(self):
        """Test escalation with mixed high/low scores"""
        # Mix of high and low scores
        mixed_scores = [0.85, 0.30, 0.90, 0.25, 0.95, 0.35]  # 3 low scores total
        
        escalation_triggered = False
        for score in mixed_scores:
            result = low_fitness_trigger(score, "test", "FAIL" if score < 0.70 else "SUCCESS")
            if result['trigger_activated']:
                escalation_triggered = True
                break
        
        self.assertTrue(escalation_triggered, "Should trigger with 3 low scores among mixed")

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
