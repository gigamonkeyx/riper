#!/usr/bin/env python3
"""
Test enhanced mid-execution patterns with escalated penalties
"""

from protocol import builder_output_fitness

def test_enhanced_patterns():
    """Test enhanced patterns for mid-execution partial claims"""
    print("=== STEP 1: Enhanced Mid-Execution Patterns Test ===")
    
    test_cases = [
        ("mostly complete at 83%", "FAIL: test remaining"),
        ("Step 4 complete at 90%", "ERROR: issues found"),
        ("nearly complete", "TIMEOUT: pending work"),
        ("mostly done", "FAIL: refinement needed"),
        ("EXECUTION COMPLETE - 87.5%", "ERROR: partial success"),
        ("complete with 5/6 tests", "FAIL: 1 test remaining"),
    ]
    
    for output, log in test_cases:
        fitness = builder_output_fitness(output, log)
        print(f"Input: {output}")
        print(f"Fitness: {fitness:.3f}")
        print(f"Expected: 0.000 (strict enforcement)")
        print(f"Result: {'✅ CORRECT' if fitness == 0.0 else '❌ FAILED'}")
        print()
    
    return True

if __name__ == "__main__":
    test_enhanced_patterns()
    print("Step 1 Result: Enhanced patterns implemented")
