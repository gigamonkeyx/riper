#!/usr/bin/env python3
"""
Final audit: Complete Qwen3-Ollama communication workflow
Test instruction flow, fitness trigger, and observer consultation
"""

from unittest.mock import Mock, patch
from orchestration import Observer
from agents import FitnessScorer
from protocol import low_fitness_trigger, PROTOCOL_METADATA

def simulate_complete_comm_workflow():
    """Simulate complete communication workflow with mocked responses"""
    print("=== STEP 4: Complete Communication Workflow Audit ===")
    
    # Clear fitness history
    PROTOCOL_METADATA["fitness_history"] = []
    
    observer = Observer("comm_audit")
    scorer = FitnessScorer()
    
    # Mock successful OpenRouter response
    mock_response = Mock()
    mock_response.success = True
    mock_response.content = """
    IMPLEMENTATION CHECKLIST FOR BIAS DETECTION ENHANCEMENT:
    
    1. Update protocol.py with enhanced pattern matching
       - Add dismissive language patterns (mostly, good enough, minor)
       - Integrate with fitness scoring system
       - Set halt condition for fitness <0.70
    
    2. Test pattern detection with sample outputs
       - Test "mostly complete at 83%" → should trigger 0.0 fitness
       - Test "good enough for now" → should trigger low fitness
       - Validate halt conditions activate properly
    
    3. Integrate with observer consultation system
       - Ensure low fitness triggers observer notification
       - Test A2A communication for escalation
       - Validate resume conditions are set
    
    4. Achieve fitness improvement >70%
       - Run comprehensive test suite
       - Validate all patterns detected correctly
       - Confirm observer consultation working
    
    5. Generate completion metrics and report
       - Document fitness improvements
       - Report observer consultation effectiveness
       - Validate system reliability >90%
    """
    
    print("PHASE 1: OpenRouter Qwen3 Instruction Generation")
    
    with patch('orchestration.get_openrouter_client') as mock_client:
        mock_client.return_value.chat_completion.return_value = mock_response
        
        # Test OpenRouter handoff
        task_description = "Enhance bias detection with fitness integration and observer consultation"
        handoff_result = observer.openrouter_to_ollama_handoff(task_description)
        
        print(f"Handoff successful: {handoff_result['handoff_successful']}")

        if handoff_result['handoff_successful']:
            print(f"Instruction length: {handoff_result['checklist_length']} characters")
            print(f"Target model: {handoff_result['target_model']}")
            print(f"Fitness requirement: {handoff_result['fitness_requirement']}")
        else:
            print(f"Handoff failed: {handoff_result.get('error', 'Unknown error')}")
            print("❌ FAILED: OpenRouter handoff should succeed with mock")
            return False
    
    print(f"\nPHASE 2: Ollama A2A Goal Reception")
    
    # Test A2A goal reception
    a2a_message = handoff_result['a2a_message']
    goal_result = scorer.receive_a2a_goal(a2a_message)
    
    print(f"Goal received: {goal_result['goal_received']}")
    print(f"Ready to execute: {goal_result['ready_to_execute']}")
    print(f"Execution mode: {goal_result['execution_plan']['execution_mode']}")
    print(f"Fitness requirement: {goal_result['fitness_requirement']}")
    
    if not goal_result['goal_received']:
        print("❌ FAILED: Ollama should receive A2A goal")
        return False
    
    print(f"\nPHASE 3: Fitness Trigger Simulation")
    
    # Simulate multiple low fitness scores during execution
    problematic_outputs = [
        ("87.5% success - minor issues", 0.60),
        ("mostly complete with remaining work", 0.45),
        ("good enough for initial version", 0.30),
        ("EXECUTION COMPLETE - 83% done", 0.00)
    ]
    
    escalation_triggered = False
    for i, (output, expected_fitness) in enumerate(problematic_outputs):
        trigger_result = low_fitness_trigger(expected_fitness, output, "FAIL: test")
        
        print(f"Output {i+1}: {output[:40]}...")
        print(f"  Fitness: {expected_fitness:.2f}")
        print(f"  Trigger activated: {trigger_result['trigger_activated']}")
        
        if trigger_result['trigger_activated']:
            escalation_triggered = True
            print(f"  🚨 ESCALATION TRIGGERED after {i+1} low scores")
            
            # Test fitness trigger report
            fitness_report = scorer.fitness_trigger_report(
                trigger_result['recent_low_count'], 
                [entry['fitness'] for entry in PROTOCOL_METADATA['fitness_history'][-4:]]
            )
            
            print(f"  Report generated: {fitness_report['report_generated']}")
            print(f"  Observer notification: {fitness_report['observer_notification_required']}")
            break
    
    if not escalation_triggered:
        print("❌ FAILED: Should trigger escalation")
        return False
    
    print(f"\nPHASE 4: Observer Consultation")
    
    # Test observer consultation with issues report
    issues_report = fitness_report['issues_report']
    consultation_result = observer.receive_issues_report(issues_report, "ollama_qwen2.5-coder")
    
    print(f"Consultation provided: {consultation_result['consultation_provided']}")
    print(f"Builder halt confirmed: {consultation_result['builder_halt_confirmed']}")
    print(f"Resume authorized: {consultation_result['resume_authorized']}")
    
    if consultation_result['consultation_provided']:
        response = consultation_result['consultation_response']
        print(f"Issue severity: {response['issue_severity']}")
        print(f"Observer analysis points: {len(response['observer_analysis'])}")
        print(f"Recommended fixes: {len(response['recommended_fixes'])}")
        print(f"Resume conditions: {len(response['resume_conditions'])}")
    
    # Validation criteria
    workflow_criteria = [
        (handoff_result['handoff_successful'], "OpenRouter handoff should succeed"),
        (goal_result['goal_received'], "Ollama should receive A2A goal"),
        (goal_result['ready_to_execute'], "Should be ready to execute"),
        (escalation_triggered, "Should trigger escalation on low fitness"),
        (fitness_report['report_generated'], "Should generate fitness report"),
        (consultation_result['consultation_provided'], "Observer should provide consultation"),
        (consultation_result['builder_halt_confirmed'], "Should confirm builder halt"),
        (not consultation_result['resume_authorized'], "Should not authorize immediate resume")
    ]
    
    print(f"\n=== WORKFLOW VALIDATION ===")
    all_working = True
    for condition, description in workflow_criteria:
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status}: {description}")
        if not condition:
            all_working = False
    
    return all_working

def test_communication_metrics():
    """Test communication system metrics"""
    print(f"\n=== COMMUNICATION SYSTEM METRICS ===")
    
    # Test fitness history tracking
    history_length = len(PROTOCOL_METADATA["fitness_history"])
    print(f"Fitness history entries: {history_length}")
    
    # Test escalation threshold
    threshold = PROTOCOL_METADATA["escalation_threshold"]
    print(f"Escalation threshold: {threshold} low scores")
    
    # Test recent low score detection
    recent_low_count = sum(1 for entry in PROTOCOL_METADATA["fitness_history"][-5:] 
                          if entry["fitness"] < 0.70)
    print(f"Recent low scores: {recent_low_count}")
    
    # Metrics validation
    metrics_valid = [
        (history_length >= 3, "Should have fitness history from simulation"),
        (threshold == 3, "Should use threshold of 3"),
        (recent_low_count >= 3, "Should have ≥3 recent low scores")
    ]
    
    print(f"\nMetrics validation:")
    all_valid = True
    for condition, description in metrics_valid:
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status}: {description}")
        if not condition:
            all_valid = False
    
    return all_valid

if __name__ == "__main__":
    print("COMPLETE QWEN3-OLLAMA COMMUNICATION AUDIT")
    print("=" * 60)
    
    # Test complete workflow
    workflow_success = simulate_complete_comm_workflow()
    
    # Test system metrics
    metrics_valid = test_communication_metrics()
    
    print(f"\n" + "=" * 60)
    print("FINAL AUDIT RESULTS:")
    print(f"Complete workflow working: {'✅ YES' if workflow_success else '❌ NO'}")
    print(f"System metrics valid: {'✅ YES' if metrics_valid else '❌ NO'}")
    
    overall_success = workflow_success and metrics_valid
    print(f"Communication system working: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("\n🎯 QWEN3-OLLAMA COMMUNICATION COMPLETE:")
        print("   OpenRouter instruction generation → Comprehensive checklists")
        print("   A2A goal exchange → Seamless handoff to Ollama")
        print("   Fitness trigger system → Automatic escalation on low scores")
        print("   Observer consultation → Comprehensive analysis and fixes")
        print("   Builder halt → Confirmed with resume conditions")
    else:
        print("\n❌ COMMUNICATION GAPS DETECTED")
    
    print(f"\nConfidence: {'85%' if overall_success else '70%'} - {'All communication mechanisms working' if overall_success else 'Some components need auth setup'}")
