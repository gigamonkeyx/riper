#!/usr/bin/env python3
"""
CRITICAL: Check protocol violation - claiming COMPLETE at 83%
"""

from protocol import check_builder_bias, builder_output_fitness

def check_violation():
    """Check the current violation against v2.6.1 protocol"""
    
    # The violation: Claiming COMPLETE with 83% success
    output = "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed, mostly complete)"
    log = "FAIL: test_bias_history_tracking - bias history refinement needed"
    
    print("PROTOCOL VIOLATION CHECK")
    print("=" * 50)
    print(f"Output: {output}")
    print(f"Log: {log}")
    print()
    
    # Check fitness score
    fitness = builder_output_fitness(output, log)
    print(f"Fitness Score: {fitness:.3f}")
    
    # Check bias analysis
    bias_result = check_builder_bias(output, log)
    print(f"Bias Detected: {bias_result['bias_detected']}")
    print(f"Mandatory Halt: {bias_result['mandatory_halt']}")
    print(f"Perfection Required: {bias_result['perfection_required']}")
    print(f"Auto-fixes: {len(bias_result['auto_fixes'])}")
    
    print("\nDETAILS:")
    for detail in bias_result['details']:
        print(f"  - {detail}")
    
    print("\nREQUIRED ACTIONS:")
    for fix in bias_result['auto_fixes']:
        print(f"  - {fix}")
    
    # Protocol enforcement
    print("\nPROTOCOL ENFORCEMENT:")
    if bias_result['mandatory_halt']:
        print("🚨 MANDATORY HALT TRIGGERED")
        print("❌ EXECUTION MUST STOP")
        print("🔧 FIXES REQUIRED BEFORE PROCEEDING")
    else:
        print("✅ APPROVED TO PROCEED")
    
    return bias_result['mandatory_halt']

if __name__ == "__main__":
    halt_required = check_violation()
    print(f"\nFINAL VERDICT: {'HALT REQUIRED' if halt_required else 'APPROVED'}")
