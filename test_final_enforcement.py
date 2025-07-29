#!/usr/bin/env python3
"""
Final audit: Test enhanced enforcement on exact violation pattern
"""

from protocol import check_builder_bias, builder_output_fitness
from orchestration import Observer

def test_exact_violation_enhanced():
    """Test enhanced enforcement on the exact violation that occurred"""
    print("=== STEP 5: Final Enhanced Enforcement Audit ===")
    
    # The exact violation pattern
    violation_output = "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed, mostly complete)"
    violation_log = "FAIL: test_bias_history_tracking - bias history refinement needed"
    
    print("EXACT VIOLATION PATTERN:")
    print(f"Output: {violation_output}")
    print(f"Log: {violation_log}")
    
    # Test enhanced enforcement
    fitness_score = builder_output_fitness(violation_output, violation_log)
    bias_analysis = check_builder_bias(violation_output, violation_log)
    
    print(f"\nENHANCED ENFORCEMENT RESULTS:")
    print(f"Fitness score: {fitness_score:.3f}")
    print(f"Execution halt: {bias_analysis.get('execution_halt', False)}")
    print(f"Mid-execution fraud: {bias_analysis.get('mid_execution_fraud', False)}")
    print(f"Mandatory halt: {bias_analysis['mandatory_halt']}")
    print(f"Auto-fixes: {len(bias_analysis['auto_fixes'])}")
    
    print(f"\nDETAILS:")
    for detail in bias_analysis['details']:
        print(f"  - {detail}")
    
    print(f"\nAUTO-FIXES:")
    for fix in bias_analysis['auto_fixes']:
        print(f"  - {fix}")
    
    # Test Observer enhanced veto
    observer = Observer("final_enforcement")
    veto_result = observer.veto_mid_execution_claim(violation_output, violation_log)
    
    print(f"\nOBSERVER ENHANCED VETO:")
    print(f"Vetoed: {veto_result['vetoed']}")
    print(f"Critical: {veto_result.get('critical', False)}")
    print(f"Execution halt: {veto_result.get('execution_halt', False)}")
    print(f"Required actions: {len(veto_result.get('required_actions', []))}")
    
    # Validation criteria
    enforcement_criteria = [
        (fitness_score == 0.0, "Fitness should be 0.0"),
        (bias_analysis.get('execution_halt', False), "Should trigger execution halt"),
        (bias_analysis.get('mid_execution_fraud', False), "Should detect mid-execution fraud"),
        (bias_analysis['mandatory_halt'], "Should trigger mandatory halt"),
        (len(bias_analysis['auto_fixes']) >= 4, "Should provide multiple auto-fixes"),
        (veto_result['vetoed'], "Observer should veto"),
        (veto_result.get('execution_halt', False), "Observer should flag execution halt"),
    ]
    
    print(f"\nENFORCEMENT VALIDATION:")
    all_working = True
    for condition, description in enforcement_criteria:
        status = "✅ PASS" if condition else "❌ FAIL"
        print(f"  {status}: {description}")
        if not condition:
            all_working = False
    
    return all_working

def test_enhanced_patterns():
    """Test enhanced pattern detection"""
    print(f"\n=== ENHANCED PATTERN DETECTION ===")
    
    enhanced_cases = [
        ("mostly complete at 83%", "FAIL: remaining"),
        ("Step 4 complete at 90%", "ERROR: issues"),
        ("nearly complete", "TIMEOUT: pending"),
        ("mostly done", "FAIL: refinement"),
        ("EXECUTION COMPLETE - 87.5%", "ERROR: partial"),
    ]
    
    all_detected = True
    for output, log in enhanced_cases:
        fitness = builder_output_fitness(output, log)
        bias_result = check_builder_bias(output, log)
        
        print(f"Pattern: {output}")
        print(f"  Fitness: {fitness:.3f}")
        print(f"  Execution halt: {bias_result.get('execution_halt', False)}")
        
        if fitness != 0.0:
            print(f"  ❌ FAILED: Should zero fitness")
            all_detected = False
        else:
            print(f"  ✅ CORRECT: Zero fitness applied")
    
    return all_detected

if __name__ == "__main__":
    print("ENHANCED MID-EXECUTION HALT ENFORCEMENT AUDIT")
    print("=" * 60)
    
    # Test exact violation with enhanced enforcement
    violation_blocked = test_exact_violation_enhanced()
    
    # Test enhanced pattern detection
    patterns_detected = test_enhanced_patterns()
    
    print(f"\n" + "=" * 60)
    print("FINAL AUDIT RESULTS:")
    print(f"Exact violation blocked: {'✅ YES' if violation_blocked else '❌ NO'}")
    print(f"Enhanced patterns detected: {'✅ YES' if patterns_detected else '❌ NO'}")
    
    overall_success = violation_blocked and patterns_detected
    print(f"Enhanced enforcement working: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")
    
    if overall_success:
        print("\n🎯 ENHANCED ENFORCEMENT COMPLETE:")
        print("   Mid-execution completion fraud → Immediate execution halt")
        print("   Enhanced pattern detection → Zero fitness for all partial claims")
        print("   Observer veto system → Enhanced with execution halt flags")
        print("   Auto-fix specificity → Immediate completion claim removal")
    else:
        print("\n❌ ENFORCEMENT GAPS DETECTED")
    
    print(f"\nConfidence: {'95%' if overall_success else '70%'} - {'All enhanced mechanisms working' if overall_success else 'Some enhancements need refinement'}")
