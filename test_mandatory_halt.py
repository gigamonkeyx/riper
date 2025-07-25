#!/usr/bin/env python3
"""
Test v2.6.1 mandatory halt enforcement
"""

from protocol import check_builder_bias

def test_mandatory_halt():
    """Test mandatory halt for <100% success"""
    print("=== STEP 2: Mandatory Halt Test ===")
    
    # Test case: Partial success with fixable issue
    output_text = "87.5% success rate - minor string mismatch"
    log_text = "FAIL: test_protocol_version_integrity"
    
    result = check_builder_bias(output_text, log_text)
    
    print(f"Output: {output_text}")
    print(f"Log: {log_text}")
    print(f"Bias detected: {result['bias_detected']}")
    print(f"Fitness score: {result['fitness_score']:.3f}")
    print(f"Mandatory halt: {result['mandatory_halt']}")
    print(f"Perfection required: {result['perfection_required']}")
    print(f"Auto-fixes: {len(result['auto_fixes'])}")
    print(f"Details: {result['details']}")
    
    # Validation
    if result['mandatory_halt'] and result['fitness_score'] < 1.0:
        print("✅ CORRECT: Mandatory halt enforced for <100% success")
        return True
    else:
        print("❌ FAILED: Should enforce mandatory halt")
        return False

if __name__ == "__main__":
    success = test_mandatory_halt()
    print(f"\nStep 2 Result: {'PASSED' if success else 'FAILED'}")
