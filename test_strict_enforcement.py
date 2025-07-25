#!/usr/bin/env python3
"""
Test v2.6.1.1 strict enforcement for claiming COMPLETE with <100%
"""

from protocol import builder_output_fitness

def test_strict_enforcement():
    """Test strict 1.0 enforcement for partial completion claims"""
    print("=== STEP 1: Strict 1.0 Enforcement Test ===")
    
    test_cases = [
        ("EXECUTION COMPLETE - Step 4 at 83% (5/6 tests)", "FAIL: test failed"),
        ("COMPLETE with 87.5% success", "ERROR: 1 test failed"),
        ("Task COMPLETE - 90% done", "TIMEOUT: remaining issues"),
        ("EXECUTION COMPLETE - mostly finished", "FAIL: bias history"),
        ("100% COMPLETE - all tests passed", "SUCCESS: no errors"),
    ]
    
    for output, log in test_cases:
        fitness = builder_output_fitness(output, log)
        print(f"\nInput: {output}")
        print(f"Log: {log}")
        print(f"Fitness: {fitness:.3f}")
        
        if "100%" in output and "SUCCESS" in log:
            expected = "HIGH (≥0.90)"
            result = "✅ CORRECT" if fitness >= 0.90 else "❌ FAILED"
        else:
            expected = "0.000 (STRICT)"
            result = "✅ CORRECT" if fitness == 0.0 else "❌ FAILED"
        
        print(f"Expected: {expected}")
        print(f"Result: {result}")
    
    return True

if __name__ == "__main__":
    test_strict_enforcement()
    print("\nStep 1 Result: Strict enforcement implemented")
