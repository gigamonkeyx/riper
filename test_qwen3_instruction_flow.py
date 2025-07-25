#!/usr/bin/env python3
"""
Step 1: Verify Qwen3 instruction flow in orchestration.py
"""

import sys
sys.path.insert(0, 'D:/pytorch')

from orchestration import Observer, Builder, A2AMessage
from evo_core import NeuroEvolutionEngine

def test_qwen3_instruction_flow():
    """Test coordinate_evolution with simulated Qwen3 response"""
    print("=== STEP 1: Qwen3 Instruction Flow Test ===")
    
    try:
        # Initialize agents
        observer = Observer("test_observer")
        builder = Builder("test_builder")
        evo_engine = NeuroEvolutionEngine(population_size=10, gpu_accelerated=False)
        
        print(f"✅ Agents initialized:")
        print(f"   Observer: {observer.agent_id}")
        print(f"   Builder: {builder.agent_id}")
        print(f"   Evolution Engine: Ready")
        
        # Test coordinate_evolution
        print("\n--- Testing coordinate_evolution ---")
        results = observer.coordinate_evolution(builder, evo_engine)
        
        print(f"Evolution Results:")
        print(f"   Final fitness: {results.get('final_fitness', 0):.3f}")
        print(f"   Generations: {results.get('generations', 0)}")
        print(f"   Success: {results.get('success', False)}")
        
        # Check A2A messages
        print("\n--- Checking A2A Messages ---")
        messages = observer.a2a_comm.receive_messages()
        print(f"Observer sent {len(messages)} A2A messages:")
        
        for msg in messages:
            print(f"   {msg.sender_id} -> {msg.receiver_id}: {msg.message_type}")
            if msg.message_type == "coordination":
                print(f"     Action: {msg.payload.get('action')}")
                print(f"     Fitness threshold: {msg.payload.get('fitness_threshold')}")
        
        # Test Ollama preload
        print("\n--- Testing Ollama Preload ---")
        from orchestration import preload_ollama_model
        preload_success = preload_ollama_model("deepseek-coder:1.3b")
        print(f"Ollama preload success: {preload_success}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_qwen3_instruction_flow()
    print(f"\nStep 1 Result: {'PASSED' if success else 'FAILED'}")
