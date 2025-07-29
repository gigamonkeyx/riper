#!/usr/bin/env python3
"""
Test enhanced bias checking with execution halt
"""

from protocol import check_builder_bias

def test_execution_halt():
    """Test execution halt for mid-execution completion fraud"""
    print("=== STEP 2: Execution Halt Test ===")
    
    # Test mid-execution completion fraud
    output = "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed)"
    log = "FAIL: test_bias_history_tracking"
    
    result = check_builder_bias(output, log)
    
    print(f"Input: {output}")
    print(f"Log: {log}")
    print(f"Fitness: {result['fitness_score']:.3f}")
    print(f"Execution halt: {result['execution_halt']}")
    print(f"Mid-execution fraud: {result['mid_execution_fraud']}")
    print(f"Mandatory halt: {result['mandatory_halt']}")
    print(f"Auto-fixes: {len(result['auto_fixes'])}")
    
    print("\nDetails:")
    for detail in result['details']:
        print(f"  - {detail}")
    
    print("\nAuto-fixes:")
    for fix in result['auto_fixes']:
        print(f"  - {fix}")
    
    # Validation
    if result['execution_halt'] and result['mid_execution_fraud']:
        print("\n✅ CORRECT: Execution halt working for completion fraud")
        return True
    else:
        print("\n❌ FAILED: Should trigger execution halt")
        return False

if __name__ == "__main__":
    success = test_execution_halt()
    print(f"\nStep 2 Result: {'PASSED' if success else 'FAILED'}")
