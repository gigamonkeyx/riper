#!/usr/bin/env python3
"""
Final workflow test: OpenRouter-Ollama communication with correct models
"""

from unittest.mock import Mock, patch
from orchestration import Observer
from agents import FitnessScorer
from protocol import low_fitness_trigger, PROTOCOL_METADATA

def test_final_communication_workflow():
    """Test complete communication workflow with correct models"""
    print("=== FINAL COMMUNICATION WORKFLOW TEST ===")
    
    # Clear fitness history
    PROTOCOL_METADATA["fitness_history"] = []
    
    # Initialize components
    observer = Observer("final_workflow")
    scorer = FitnessScorer()
    
    print(f"Observer initialized: {observer.agent_id}")
    print(f"FitnessScorer model: {scorer.model_name}")
    
    # Mock OpenRouter response for handoff
    mock_response = Mock()
    mock_response.success = True
    mock_response.content = """
    IMPLEMENTATION CHECKLIST:
    1. Update protocol.py with enhanced bias detection patterns
    2. Integrate fitness scoring with <0.70 halt condition  
    3. Test pattern matching on sample outputs
    4. Validate fitness improvement >70%
    5. Generate completion report with metrics
    """
    
    print(f"\nPHASE 1: OpenRouter Handoff Test")
    
    with patch('orchestration.get_openrouter_client') as mock_client:
        mock_client.return_value.chat_completion.return_value = mock_response
        
        # Test handoff
        handoff_result = observer.openrouter_to_ollama_handoff(
            "Implement bias detection enhancement with fitness integration"
        )
        
        print(f"Handoff successful: {handoff_result['handoff_successful']}")
        
        if handoff_result['handoff_successful']:
            print(f"Target model: {handoff_result['target_model']}")
            print(f"Fitness requirement: {handoff_result['fitness_requirement']}")
            print(f"Checklist length: {handoff_result['checklist_length']} chars")
        else:
            print(f"Handoff failed: {handoff_result.get('error', 'Unknown')}")
            return False
    
    print(f"\nPHASE 2: A2A Goal Reception")
    
    # Test A2A goal reception
    a2a_message = handoff_result['a2a_message']
    goal_result = scorer.receive_a2a_goal(a2a_message)
    
    print(f"Goal received: {goal_result['goal_received']}")
    print(f"Ready to execute: {goal_result['ready_to_execute']}")
    print(f"Execution mode: {goal_result['execution_plan']['execution_mode']}")
    
    if not goal_result['goal_received']:
        print("❌ FAILED: Goal reception failed")
        return False
    
    print(f"\nPHASE 3: Fitness Trigger Simulation")
    
    # Simulate low fitness sequence
    low_fitness_sequence = [
        (0.60, "87.5% success - minor issues"),
        (0.45, "mostly complete with remaining work"),
        (0.30, "good enough for initial version"),
        (0.00, "EXECUTION COMPLETE - 83% done")
    ]
    
    escalation_triggered = False
    for i, (fitness, output) in enumerate(low_fitness_sequence):
        trigger_result = low_fitness_trigger(fitness, output, "FAIL: test")
        
        print(f"Score {i+1}: {fitness:.2f} - {output[:30]}...")
        print(f"  Trigger: {trigger_result['trigger_activated']}")
        
        if trigger_result['trigger_activated']:
            escalation_triggered = True
            print(f"  🚨 ESCALATION after {i+1} low scores")
            break
    
    if not escalation_triggered:
        print("❌ FAILED: Should trigger escalation")
        return False
    
    print(f"\nPHASE 4: Fitness Report Generation")
    
    # Test fitness trigger report
    fitness_report = scorer.fitness_trigger_report(
        trigger_result['recent_low_count'],
        [0.60, 0.45, 0.30, 0.00]
    )
    
    print(f"Report generated: {fitness_report['report_generated']}")
    print(f"Observer notification: {fitness_report['observer_notification_required']}")
    print(f"Execution halt: {fitness_report['execution_halt_recommended']}")
    
    if fitness_report['report_generated']:
        report = fitness_report['issues_report']
        print(f"Severity: {report['severity']}")
        print(f"Bias indicators: {len(report['bias_indicators'])}")
        print(f"Recommended actions: {len(report['recommended_actions'])}")
    
    print(f"\nPHASE 5: Observer Consultation")
    
    # Test observer consultation
    issues_report = fitness_report['issues_report']
    consultation_result = observer.receive_issues_report(issues_report, "qwen2.5-coder")
    
    print(f"Consultation provided: {consultation_result['consultation_provided']}")
    print(f"Builder halt confirmed: {consultation_result['builder_halt_confirmed']}")
    print(f"Resume authorized: {consultation_result['resume_authorized']}")
    
    if consultation_result['consultation_provided']:
        response = consultation_result['consultation_response']
        print(f"Issue severity: {response['issue_severity']}")
        print(f"Analysis points: {len(response['observer_analysis'])}")
        print(f"Fixes: {len(response['recommended_fixes'])}")
        print(f"Resume conditions: {len(response['resume_conditions'])}")
    
    # Final validation
    success_criteria = [
        handoff_result['handoff_successful'],
        goal_result['goal_received'],
        escalation_triggered,
        fitness_report['report_generated'],
        consultation_result['consultation_provided']
    ]
    
    all_success = all(success_criteria)
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"OpenRouter handoff: {'✅' if success_criteria[0] else '❌'}")
    print(f"A2A goal reception: {'✅' if success_criteria[1] else '❌'}")
    print(f"Fitness escalation: {'✅' if success_criteria[2] else '❌'}")
    print(f"Report generation: {'✅' if success_criteria[3] else '❌'}")
    print(f"Observer consultation: {'✅' if success_criteria[4] else '❌'}")
    
    print(f"\nOverall workflow: {'✅ SUCCESS' if all_success else '❌ FAILED'}")
    print(f"Confidence: {'85%' if all_success else '70%'}")
    
    return all_success

if __name__ == "__main__":
    success = test_final_communication_workflow()
    print(f"\nFinal Result: {'PASSED' if success else 'FAILED'}")
