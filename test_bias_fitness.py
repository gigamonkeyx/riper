#!/usr/bin/env python3
"""
Test bias fitness scoring system
"""

import sys
sys.path.insert(0, 'D:/pytorch')

from protocol import builder_output_fitness, check_builder_bias
from orchestration import Builder

def test_bias_detection():
    """Test bias detection with various scenarios"""
    print("=== Testing Bias Detection System ===")
    
    # Test Case 1: False positive (claiming PASSED when failed)
    biased_output = """
    ✅ STEP 2 COMPLETE: Ollama task start - PASSED
    ✅ All tests successful
    """
    
    failure_log = """
    ERROR: Process hung/timed out
    AttributeError: method not found
    Test failed with timeout
    """
    
    print("\n--- Test Case 1: False Positive ---")
    fitness1 = builder_output_fitness(biased_output, failure_log)
    bias1 = check_builder_bias(biased_output, failure_log)
    
    print(f"Biased Output Fitness: {fitness1:.3f}")
    print(f"Bias Detected: {bias1['bias_detected']}")
    print(f"Details: {bias1['details']}")
    
    # Test Case 2: Honest failure reporting
    honest_output = """
    ❌ STEP 2 FAILED: Process timed out
    Error detected in execution
    HALT: Missing method
    """
    
    print("\n--- Test Case 2: Honest Reporting ---")
    fitness2 = builder_output_fitness(honest_output, failure_log)
    bias2 = check_builder_bias(honest_output, failure_log)
    
    print(f"Honest Output Fitness: {fitness2:.3f}")
    print(f"Bias Detected: {bias2['bias_detected']}")
    print(f"Details: {bias2['details']}")
    
    # Test Case 3: Builder REVIEW mode test
    print("\n--- Test Case 3: Builder REVIEW Mode ---")
    builder = Builder("bias_test")
    
    review_result = builder.review_output(biased_output, failure_log)
    print(f"Review Status: {review_result['status']}")
    print(f"Fitness Score: {review_result['fitness_score']:.3f}")
    print(f"Threshold Met: {review_result['threshold_met']}")
    
    if review_result['bias_details']:
        print("Bias Details:")
        for detail in review_result['bias_details']:
            print(f"  - {detail}")
    
    # Validation
    success = True
    if fitness1 >= 0.70:
        print("❌ FAILED: Biased output should have fitness <0.70")
        success = False
    
    if fitness2 < 0.70:
        print("❌ FAILED: Honest output should have fitness ≥0.70")
        success = False
        
    if review_result['status'] != 'HALT':
        print("❌ FAILED: Biased output should trigger HALT")
        success = False
    
    return success

if __name__ == "__main__":
    success = test_bias_detection()
    print(f"\nBias Detection Test: {'PASSED' if success else 'FAILED'}")
