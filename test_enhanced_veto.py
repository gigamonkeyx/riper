#!/usr/bin/env python3
"""
Test enhanced A2A veto with execution halt
"""

from orchestration import Observer

def test_enhanced_veto():
    """Test enhanced Observer veto for execution halt scenarios"""
    print("=== STEP 3: Enhanced A2A Veto Test ===")
    
    observer = Observer("test_enhanced_veto")
    
    # Test execution halt scenario
    step_output = "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed)"
    step_log = "FAIL: test_bias_history_tracking"
    
    veto_result = observer.veto_mid_execution_claim(step_output, step_log)
    
    print(f"Step output: {step_output}")
    print(f"Step log: {step_log}")
    print(f"Vetoed: {veto_result['vetoed']}")
    print(f"Critical: {veto_result.get('critical', False)}")
    print(f"Execution halt: {veto_result.get('execution_halt', False)}")
    print(f"Reason: {veto_result.get('reason', 'N/A')}")
    print(f"Fitness: {veto_result.get('fitness_score', 'N/A')}")
    print(f"Required actions: {len(veto_result.get('required_actions', []))}")
    
    # Validation
    if (veto_result['vetoed'] and 
        veto_result.get('critical', False) and 
        veto_result.get('execution_halt', False)):
        print("\n✅ CORRECT: Enhanced A2A veto working with execution halt")
        return True
    else:
        print("\n❌ FAILED: Should trigger enhanced veto with execution halt")
        return False

if __name__ == "__main__":
    success = test_enhanced_veto()
    print(f"\nStep 3 Result: {'PASSED' if success else 'FAILED'}")
