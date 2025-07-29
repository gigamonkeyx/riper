#!/usr/bin/env python3
"""
Test enhanced bias checking with escalation
"""

from protocol import check_builder_bias, PROTOCOL_METADATA

def test_bias_escalation():
    """Test bias checking with escalation trigger"""
    print("=== STEP 2: Bias Escalation Test ===")
    
    # Clear fitness history
    PROTOCOL_METADATA["fitness_history"] = []
    
    # Simulate sequence leading to escalation
    test_sequence = [
        ("87.5% success - minor issue", "FAIL: test"),
        ("mostly complete", "ERROR: remaining"),
        ("good enough for now", "TIMEOUT: pending"),
        ("EXECUTION COMPLETE - 83%", "FAIL: test")
    ]
    
    print("Testing bias checking with escalation:")
    for i, (output, log) in enumerate(test_sequence):
        result = check_builder_bias(output, log)
        
        print(f"Check {i+1}: {output[:30]}...")
        print(f"  Fitness: {result['fitness_score']:.3f}")
        print(f"  Escalation triggered: {result['escalation_triggered']}")
        
        if result['escalation_triggered']:
            print(f"  ✅ ESCALATION ACTIVATED")
            print(f"  Consultation required: {result['consultation_required']}")
            print(f"  Issues report: {result['issues_report'] is not None}")
            print(f"  Mandatory halt: {result['mandatory_halt']}")
            
            if result['issues_report']:
                report = result['issues_report']
                print(f"  Report details:")
                print(f"    - Low score count: {report['low_score_count']}")
                print(f"    - Bias patterns: {len(report['bias_patterns'])}")
                print(f"    - Recommended actions: {len(report['recommended_actions'])}")
            
            return True
    
    print("❌ FAILED: Should trigger escalation")
    return False

if __name__ == "__main__":
    success = test_bias_escalation()
    print(f"\nStep 2 Result: {'PASSED' if success else 'FAILED'}")
