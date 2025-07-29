#!/usr/bin/env python3
"""
Test communication handoff system
Simulate Qwen3 instruction to Ollama task start, verify no timeouts, fitness >70%
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import unittest
from unittest.mock import Mock, patch
from orchestration import Observer
from agents import FitnessScorer

class TestCommHandoff(unittest.TestCase):
    """Test communication handoff between OpenRouter Qwen3 and Ollama"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.observer = Observer("test_comm_handoff")
        self.scorer = FitnessScorer()
        
    def test_handoff_structure(self):
        """Test handoff message structure without API call"""
        # Mock successful OpenRouter response
        mock_response = Mock()
        mock_response.success = True
        mock_response.content = """
        1. Initialize bias detection patterns in protocol.py
        2. Add fitness scoring integration with halt condition <0.70
        3. Test pattern matching with sample outputs
        4. Validate fitness improvement >70%
        5. Generate completion report with metrics
        """
        
        with patch('orchestration.get_openrouter_client') as mock_client:
            mock_client.return_value.chat_completion.return_value = mock_response
            
            task_description = "Implement bias detection enhancement"
            result = self.observer.openrouter_to_ollama_handoff(task_description)
            
            # Validate handoff structure
            self.assertTrue(result['handoff_successful'], "Should succeed with mock")
            self.assertIn('instruction_checklist', result, "Should have checklist")
            self.assertIn('a2a_message', result, "Should have A2A message")
            self.assertEqual(result['fitness_requirement'], 0.70, "Should set fitness requirement")
            
            # Validate A2A message structure
            a2a_msg = result['a2a_message']
            self.assertEqual(a2a_msg['action'], 'goal_exchange', "Should be goal_exchange")
            self.assertEqual(a2a_msg['source'], 'openrouter_qwen3', "Should identify source")
            self.assertIn('ollama_', a2a_msg['target'], "Should target Ollama")
            self.assertEqual(a2a_msg['fitness_requirement'], 0.70, "Should set fitness requirement")
            
    def test_a2a_goal_reception(self):
        """Test A2A goal reception and execution preparation"""
        # Create A2A message from handoff
        a2a_message = {
            "action": "goal_exchange",
            "source": "openrouter_qwen3",
            "target": "ollama_qwen2.5-coder:32b",
            "instruction_type": "implementation_checklist",
            "checklist": """
            1. Add pattern matching for dismissive language
            2. Integrate fitness scoring with <0.70 halt
            3. Test with sample outputs and validate results
            4. Achieve >70% fitness improvement
            """,
            "fitness_requirement": 0.70,
            "halt_on_low_fitness": True
        }
        
        # Test goal reception
        result = self.scorer.receive_a2a_goal(a2a_message)
        
        self.assertTrue(result['goal_received'], "Should receive A2A goal")
        self.assertTrue(result['ready_to_execute'], "Should be ready to execute")
        self.assertEqual(result['fitness_requirement'], 0.70, "Should preserve fitness requirement")
        
        # Validate execution plan
        plan = result['execution_plan']
        self.assertEqual(plan['execution_mode'], 'EXECUTE', "Should set EXECUTE mode")
        self.assertTrue(plan['ready_to_start'], "Should be ready to start")
        self.assertTrue(plan['halt_on_low_fitness'], "Should halt on low fitness")
        
    def test_complete_handoff_chain(self):
        """Test complete handoff chain from Qwen3 to Ollama execution"""
        # Mock OpenRouter response
        mock_response = Mock()
        mock_response.success = True
        mock_response.content = """
        IMPLEMENTATION CHECKLIST:
        1. Update protocol.py with new bias patterns
        2. Add fitness integration with halt <0.70
        3. Test pattern detection on sample outputs
        4. Validate fitness scores >70%
        5. Generate completion metrics
        """
        
        with patch('orchestration.get_openrouter_client') as mock_client:
            mock_client.return_value.chat_completion.return_value = mock_response
            
            # Step 1: OpenRouter generates instruction
            task = "Enhance bias detection system"
            handoff_result = self.observer.openrouter_to_ollama_handoff(task)
            
            self.assertTrue(handoff_result['handoff_successful'], "Handoff should succeed")
            
            # Step 2: Ollama receives A2A goal
            a2a_message = handoff_result['a2a_message']
            goal_result = self.scorer.receive_a2a_goal(a2a_message)
            
            self.assertTrue(goal_result['goal_received'], "Goal should be received")
            self.assertTrue(goal_result['ready_to_execute'], "Should be ready to execute")
            
            # Step 3: Validate execution readiness
            execution_plan = goal_result['execution_plan']
            self.assertGreater(len(execution_plan['instruction_checklist']), 100, 
                              "Should have substantial checklist")
            self.assertEqual(execution_plan['fitness_requirement'], 0.70, 
                            "Should maintain fitness requirement")
            
    def test_handoff_error_handling(self):
        """Test handoff error handling and fallback"""
        # Mock OpenRouter failure
        mock_response = Mock()
        mock_response.success = False
        mock_response.error_message = "API rate limit exceeded"
        
        with patch('orchestration.get_openrouter_client') as mock_client:
            mock_client.return_value.chat_completion.return_value = mock_response
            
            task = "Test task"
            result = self.observer.openrouter_to_ollama_handoff(task)
            
            self.assertFalse(result['handoff_successful'], "Should fail with API error")
            self.assertTrue(result['fallback_required'], "Should require fallback")
            self.assertIn('error', result, "Should include error message")
            
    def test_fitness_trigger_integration(self):
        """Test fitness trigger integration with handoff system"""
        # Simulate low fitness scores that would trigger report
        low_scores = [0.60, 0.45, 0.30, 0.00]
        
        trigger_result = self.scorer.fitness_trigger_report(len(low_scores), low_scores)
        
        self.assertTrue(trigger_result['report_generated'], "Should generate report")
        self.assertTrue(trigger_result['observer_notification_required'], "Should notify observer")
        self.assertTrue(trigger_result['execution_halt_recommended'], "Should recommend halt")
        
        # Validate report content
        report = trigger_result['issues_report']
        self.assertEqual(report['severity'], 'HIGH', "Should be HIGH severity for 4 low scores")
        self.assertGreater(len(report['bias_indicators']), 0, "Should identify bias indicators")
        self.assertGreater(len(report['recommended_actions']), 0, "Should provide actions")
        
    def test_invalid_a2a_message(self):
        """Test handling of invalid A2A messages"""
        # Test invalid action
        invalid_message = {
            "action": "invalid_action",
            "source": "test",
            "checklist": "test checklist"
        }
        
        result = self.scorer.receive_a2a_goal(invalid_message)
        
        self.assertFalse(result['goal_received'], "Should reject invalid action")
        self.assertIn('error', result, "Should include error message")
        
    def test_fitness_requirement_propagation(self):
        """Test fitness requirement propagation through handoff chain"""
        # Mock response with fitness validation
        mock_response = Mock()
        mock_response.success = True
        mock_response.content = "Test checklist with fitness >70% requirement"
        
        with patch('orchestration.get_openrouter_client') as mock_client:
            mock_client.return_value.chat_completion.return_value = mock_response
            
            # Test handoff
            result = self.observer.openrouter_to_ollama_handoff("test task")
            
            # Verify fitness requirement in handoff
            self.assertEqual(result['fitness_requirement'], 0.70, "Handoff should set 0.70")
            
            # Test A2A reception
            a2a_msg = result['a2a_message']
            goal_result = self.scorer.receive_a2a_goal(a2a_msg)
            
            # Verify fitness requirement propagated
            self.assertEqual(goal_result['fitness_requirement'], 0.70, "Should propagate 0.70")
            
            execution_plan = goal_result['execution_plan']
            self.assertEqual(execution_plan['fitness_requirement'], 0.70, "Plan should have 0.70")

if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)
