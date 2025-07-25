#!/usr/bin/env python3
"""
Step 2: Test enhanced bias checking with auto-fixes
"""

from protocol import check_builder_bias

def test_enhanced_bias_check():
    """Test enhanced bias checking with auto-fix suggestions"""
    print("=== STEP 2: Enhanced Bias Check Test ===")
    
    # Test case: Dismissing failure as minor
    output_text = "87.5% success rate (minor issue - v2.6 vs 2.6)"
    log_text = "FAIL: test_protocol_version_integrity"
    
    result = check_builder_bias(output_text, log_text)
    
    print(f"Output: {output_text}")
    print(f"Log: {log_text}")
    print(f"Bias detected: {result['bias_detected']}")
    print(f"Fitness score: {result['fitness_score']:.3f}")
    print(f"Threshold met: {result['threshold_met']}")
    print(f"Requires rerun: {result['requires_rerun']}")
    print(f"Details: {result['details']}")
    print(f"Auto-fixes: {result['auto_fixes']}")
    
    # Validation
    if result['bias_detected'] and result['fitness_score'] < 0.70:
        print("✅ CORRECT: Bias detected for dismissing failures")
        return True
    else:
        print("❌ FAILED: Should detect bias in dismissal")
        return False

if __name__ == "__main__":
    success = test_enhanced_bias_check()
    print(f"\nStep 2 Result: {'PASSED' if success else 'FAILED'}")
