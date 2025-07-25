#!/usr/bin/env python3
"""
Step 2: Test Ollama task start with coordination message
"""

import sys
sys.path.insert(0, 'D:/pytorch')

from orchestration import Builder, A2AMessage

def test_ollama_task_start():
    """Test process_coordination_message with sample checklist"""
    print("=== STEP 2: Ollama Task Start Test ===")
    
    try:
        # Initialize Builder
        builder = Builder("test_builder")
        print(f"✅ Builder initialized: {builder.agent_id}")
        
        # Create sample coordination message from Qwen3
        coordination_msg = A2AMessage(
            sender_id="qwen3_observer",
            receiver_id="test_builder", 
            message_type="coordination",
            payload={
                "action": "start_evolution",
                "checklist": [
                    "Initialize Ollama specialists",
                    "Preload models for GPU optimization", 
                    "Start evolution cycle with fitness >0.70",
                    "Monitor VRAM usage <8GB",
                    "Report progress via A2A"
                ],
                "fitness_threshold": 0.70,
                "gpu_target": "rtx_3080"
            }
        )
        
        print(f"✅ Coordination message created: {coordination_msg.message_type}")
        print(f"   Action: {coordination_msg.payload['action']}")
        print(f"   Checklist items: {len(coordination_msg.payload['checklist'])}")
        
        # Process coordination message
        print("\nProcessing coordination message...")
        result = builder.process_coordination_message(coordination_msg)
        
        print(f"✅ Message processed successfully")
        print(f"   Result: {result.get('status', 'unknown')}")
        print(f"   Actions completed: {result.get('actions_completed', 0)}")
        
        # Check if Ollama specialists were called
        if 'ollama_calls' in result:
            print(f"✅ Ollama specialist calls: {result['ollama_calls']}")
        
        # Verify fitness logging
        if 'fitness_logged' in result:
            print(f"✅ Fitness logged: {result['fitness_logged']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_task_start()
    print(f"\nStep 2 Result: {'PASSED' if success else 'FAILED'}")
