#!/usr/bin/env python3
"""
Step 1: Test bias functions with dummy output
"""

from protocol import builder_output_fitness, check_builder_bias

def test_bias_functions():
    """Test bias detection with false positive scenario"""
    print("=== STEP 1: Testing Bias Functions ===")
    
    # Test case: False positive (claiming PASSED when error occurred)
    output_text = "✅ STEP 1 COMPLETE: Test - PASSED"
    log_text = "ERROR: Process failed\nTimeout occurred\nException raised"
    
    # Test builder_output_fitness
    fitness_score = builder_output_fitness(output_text, log_text)
    print(f"Fitness score: {fitness_score:.3f}")
    
    # Test check_builder_bias
    bias_analysis = check_builder_bias(output_text, log_text)
    print(f"Bias detected: {bias_analysis['bias_detected']}")
    print(f"Threshold met: {bias_analysis['threshold_met']}")
    print(f"Halt required: {fitness_score < 0.70}")
    print(f"Details: {bias_analysis['details']}")
    
    # Validation
    if fitness_score < 0.70:
        print("✅ CORRECT: False positive detected (fitness <0.70)")
        return True
    else:
        print("❌ FAILED: Should detect false positive")
        return False

if __name__ == "__main__":
    success = test_bias_functions()
    print(f"\nStep 1 Result: {'PASSED' if success else 'FAILED'}")
