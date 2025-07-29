#!/usr/bin/env python3
"""
Test observer consultation for escalation reports
"""

from orchestration import Observer

def test_observer_consultation():
    """Test Observer consultation capability"""
    print("=== STEP 3: Observer Consultation Test ===")
    
    observer = Observer("test_consultation")
    
    # Simulate issues report from escalation trigger
    issues_report = {
        "trigger_reason": "Multiple low fitness scores detected",
        "low_score_count": 4,
        "threshold": 3,
        "recent_scores": [0.60, 0.45, 0.30, 0.00],
        "bias_patterns": [
            "Moderate bias: Dismissive language detected",
            "Severe bias: Multiple false positive patterns",
            "Critical bias: Zero fitness (completion fraud)"
        ],
        "recommended_actions": [
            "IMMEDIATE: Halt builder execution pending observer review",
            "ANALYZE: Review recent outputs for systematic bias patterns"
        ],
        "consultation_required": True,
        "halt_builder": True
    }
    
    # Test observer consultation
    consultation_result = observer.receive_issues_report(issues_report, "test_builder")
    
    print(f"Issues report processed: {consultation_result['consultation_provided']}")
    print(f"Builder halt confirmed: {consultation_result['builder_halt_confirmed']}")
    print(f"Resume authorized: {consultation_result['resume_authorized']}")
    
    if consultation_result['consultation_provided']:
        response = consultation_result['consultation_response']
        print(f"Consultation details:")
        print(f"  - Issue severity: {response['issue_severity']}")
        print(f"  - Observer analysis: {len(response['observer_analysis'])} points")
        print(f"  - Recommended fixes: {len(response['recommended_fixes'])} actions")
        print(f"  - Resume conditions: {len(response['resume_conditions'])} requirements")
        
        print("Observer analysis:")
        for analysis in response['observer_analysis']:
            print(f"  - {analysis}")
        
        print("Recommended fixes:")
        for fix in response['recommended_fixes'][:3]:
            print(f"  - {fix}")
        
        return True
    
    print("❌ FAILED: Should provide consultation")
    return False

if __name__ == "__main__":
    success = test_observer_consultation()
    print(f"\nStep 3 Result: {'PASSED' if success else 'FAILED'}")
