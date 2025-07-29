#!/usr/bin/env python3
"""
Test agents.py enhancements for A2A goal reception and fitness trigger
"""

from agents import FitnessScorer

def test_a2a_goal_reception():
    """Test A2A goal reception from OpenRouter Qwen3"""
    print("=== STEP 2A: A2A Goal Reception Test ===")
    
    scorer = FitnessScorer()
    
    # Simulate A2A message from OpenRouter handoff
    a2a_message = {
        "action": "goal_exchange",
        "source": "openrouter_qwen3",
        "target": "ollama_qwen2.5-coder:32b",
        "instruction_type": "implementation_checklist",
        "checklist": """
        1. Add pattern matching for dismissive language in protocol.py
        2. Integrate fitness scoring with <0.70 halt condition
        3. Test with sample outputs: "mostly complete", "good enough"
        4. Validate results achieve >70% fitness improvement
        5. Generate completion report with metrics
        """,
        "fitness_requirement": 0.70,
        "halt_on_low_fitness": True,
        "handoff_timestamp": 1234567890
    }
    
    print(f"A2A message: {a2a_message['action']} from {a2a_message['source']}")
    print(f"Checklist length: {len(a2a_message['checklist'])} characters")
    
    # Test goal reception
    result = scorer.receive_a2a_goal(a2a_message)
    
    print(f"Goal received: {result['goal_received']}")
    print(f"Ready to execute: {result['ready_to_execute']}")
    print(f"Fitness requirement: {result['fitness_requirement']}")
    
    if result['goal_received']:
        plan = result['execution_plan']
        print(f"Execution plan:")
        print(f"  - Source: {plan['source']}")
        print(f"  - Mode: {plan['execution_mode']}")
        print(f"  - Fitness requirement: {plan['fitness_requirement']}")
        print(f"  - Halt on low fitness: {plan['halt_on_low_fitness']}")
        print(f"  - Ready to start: {plan['ready_to_start']}")
        
        return True
    
    print("❌ FAILED: Should receive A2A goal")
    return False

def test_fitness_trigger_report():
    """Test fitness trigger report generation"""
    print("\n=== STEP 2B: Fitness Trigger Report Test ===")
    
    scorer = FitnessScorer()
    
    # Simulate multiple low fitness scores
    recent_scores = [0.60, 0.45, 0.30, 0.00]  # 4 low scores
    low_fitness_count = len(recent_scores)
    
    print(f"Low fitness scores: {recent_scores}")
    print(f"Count: {low_fitness_count}")
    
    # Test fitness trigger report
    result = scorer.fitness_trigger_report(low_fitness_count, recent_scores)
    
    print(f"Report generated: {result['report_generated']}")
    print(f"Observer notification required: {result['observer_notification_required']}")
    print(f"Execution halt recommended: {result['execution_halt_recommended']}")
    
    if result['report_generated']:
        report = result['issues_report']
        print(f"Issues report:")
        print(f"  - Trigger source: {report['trigger_source']}")
        print(f"  - Severity: {report['severity']}")
        print(f"  - Low fitness count: {report['low_fitness_count']}")
        print(f"  - Bias indicators: {len(report['bias_indicators'])}")
        print(f"  - Recommended actions: {len(report['recommended_actions'])}")
        
        print("Bias indicators:")
        for indicator in report['bias_indicators']:
            print(f"    - {indicator}")
        
        print("Recommended actions (first 3):")
        for action in report['recommended_actions'][:3]:
            print(f"    - {action}")
        
        return True
    
    print("❌ FAILED: Should generate fitness trigger report")
    return False

if __name__ == "__main__":
    print("AGENTS.PY ENHANCEMENT TEST")
    print("=" * 50)
    
    # Test A2A goal reception
    goal_success = test_a2a_goal_reception()
    
    # Test fitness trigger report
    trigger_success = test_fitness_trigger_report()
    
    print(f"\n" + "=" * 50)
    print("ENHANCEMENT RESULTS:")
    print(f"A2A goal reception: {'✅ PASSED' if goal_success else '❌ FAILED'}")
    print(f"Fitness trigger report: {'✅ PASSED' if trigger_success else '❌ FAILED'}")
    
    overall_success = goal_success and trigger_success
    print(f"Overall enhancement: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    print(f"\nStep 2 Result: {'PASSED' if overall_success else 'FAILED'}")
