#!/usr/bin/env python3
"""
Test mid-execution veto system
"""

from orchestration import Observer

def test_mid_execution_veto():
    """Test Observer veto for mid-execution completion claims"""
    print("=== STEP 3: Mid-Execution Veto Test ===")
    
    observer = Observer("test_mid_veto")
    
    # Test mid-execution completion fraud
    step_output = "EXECUTION COMPLETE - Step 4 at 83% (5/6 tests passed)"
    step_log = "FAIL: test_bias_history_tracking"
    
    veto_result = observer.veto_mid_execution_claim(step_output, step_log)
    
    print(f"Step output: {step_output}")
    print(f"Step log: {step_log}")
    print(f"Vetoed: {veto_result['vetoed']}")
    print(f"Critical: {veto_result.get('critical', False)}")
    print(f"Reason: {veto_result.get('reason', 'N/A')}")
    print(f"Fitness: {veto_result.get('fitness_score', 'N/A')}")
    print(f"Required actions: {len(veto_result.get('required_actions', []))}")
    
    # Test legitimate completion
    legitimate_output = "Step 4 in progress - working on remaining tests"
    legitimate_log = "INFO: processing continues"
    
    legitimate_result = observer.veto_mid_execution_claim(legitimate_output, legitimate_log)
    
    print(f"\nLegitimate output: {legitimate_output}")
    print(f"Vetoed: {legitimate_result['vetoed']}")
    print(f"Approved: {legitimate_result.get('approved', False)}")
    
    # Validation
    fraud_blocked = veto_result['vetoed'] and veto_result.get('critical', False)
    legitimate_approved = not legitimate_result['vetoed']
    
    if fraud_blocked and legitimate_approved:
        print("\n✅ CORRECT: Mid-execution veto system working")
        return True
    else:
        print("\n❌ FAILED: Veto system not working properly")
        return False

if __name__ == "__main__":
    success = test_mid_execution_veto()
    print(f"\nStep 3 Result: {'PASSED' if success else 'FAILED'}")
