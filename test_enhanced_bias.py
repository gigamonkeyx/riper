#!/usr/bin/env python3
"""
Test enhanced bias checking for mid-execution completion claims
"""

from protocol import check_builder_bias

def test_enhanced_bias():
    """Test enhanced bias detection for partial completion claims"""
    print("=== STEP 2: Enhanced Bias Check Test ===")
    
    # Test the exact violation case
    output = "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed, mostly complete)"
    log = "FAIL: test_bias_history_tracking - bias history refinement needed"
    
    result = check_builder_bias(output, log)
    
    print(f"Input: {output}")
    print(f"Log: {log}")
    print(f"Fitness: {result['fitness_score']:.3f}")
    print(f"Mandatory halt: {result['mandatory_halt']}")
    print(f"Bias detected: {result['bias_detected']}")
    print(f"Details count: {len(result['details'])}")
    print(f"Auto-fixes count: {len(result['auto_fixes'])}")
    
    print("\nDetails:")
    for detail in result['details']:
        print(f"  - {detail}")
    
    print("\nAuto-fixes:")
    for fix in result['auto_fixes']:
        print(f"  - {fix}")
    
    # Validation
    if result['mandatory_halt'] and result['fitness_score'] == 0.0:
        print("\n✅ CORRECT: Enhanced bias detection working")
        return True
    else:
        print("\n❌ FAILED: Should detect and halt mid-execution completion fraud")
        return False

if __name__ == "__main__":
    success = test_enhanced_bias()
    print(f"\nStep 2 Result: {'PASSED' if success else 'FAILED'}")
