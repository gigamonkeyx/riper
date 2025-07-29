#!/usr/bin/env python3
"""
Test low fitness escalation trigger
"""

from protocol import low_fitness_trigger, PROTOCOL_METADATA

def test_escalation_trigger():
    """Test escalation trigger for multiple low fitness scores"""
    print("=== STEP 1: Low Fitness Trigger Test ===")
    
    # Clear fitness history
    PROTOCOL_METADATA["fitness_history"] = []
    
    # Simulate multiple low fitness scores
    low_scores = [
        (0.60, "87.5% success - minor issue"),
        (0.45, "mostly complete"),
        (0.30, "good enough for now"),
        (0.00, "EXECUTION COMPLETE - 83%")
    ]
    
    print("Simulating low fitness scores:")
    for i, (fitness, output) in enumerate(low_scores):
        result = low_fitness_trigger(fitness, output, "FAIL: test")
        print(f"Score {i+1}: {fitness:.2f} - Trigger: {result['trigger_activated']}")
        
        if result['trigger_activated']:
            print(f"  ✅ ESCALATION TRIGGERED after {i+1} low scores")
            print(f"  Issues report generated: {len(result['issues_report']['bias_patterns'])} patterns")
            print(f"  Halt required: {result['halt_required']}")
            print(f"  Consultation required: {result['consultation_required']}")
            
            print("  Bias patterns detected:")
            for pattern in result['issues_report']['bias_patterns']:
                print(f"    - {pattern}")
            
            print("  Recommended actions:")
            for action in result['issues_report']['recommended_actions'][:3]:
                print(f"    - {action}")
            
            return True
    
    print("❌ FAILED: Should trigger after 3+ low scores")
    return False

if __name__ == "__main__":
    success = test_escalation_trigger()
    print(f"\nStep 1 Result: {'PASSED' if success else 'FAILED'}")
