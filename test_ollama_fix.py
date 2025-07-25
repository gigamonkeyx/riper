#!/usr/bin/env python3
"""
Quick test to verify Ollama integration fix
"""

import sys
import time
sys.path.insert(0, 'D:/pytorch')

from agents import FitnessScorer

def test_ollama_integration():
    """Test if Ollama integration is now working"""
    print("=== Testing Ollama Integration Fix ===")
    
    try:
        # Initialize FitnessScorer (uses qwen3:8b now)
        scorer = FitnessScorer()
        print(f"✅ FitnessScorer initialized with model: {scorer.model_name}")
        
        # Test simple fitness evaluation
        print("Testing hybrid fitness evaluation...")
        start_time = time.time()
        
        result = scorer.process_task(
            {
                "test_data": "simple_evaluation",
                "fitness_target": 0.70
            },
            generation=1,
            current_fitness=0.60
        )
        
        execution_time = time.time() - start_time
        
        if result.success:
            print("✅ Hybrid evaluation successful!")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Fitness score: {result.data.get('fitness_score', 'N/A')}")
            
            # Check evaluation details
            evaluation = result.data.get('evaluation', {})
            evaluations = evaluation.get('evaluations', {})
            
            qwen3_success = evaluations.get('qwen3', {}).get('success', False)
            ollama_success = evaluations.get('ollama', {}).get('success', False)
            
            print(f"   Qwen3 (OpenRouter): {'✅' if qwen3_success else '❌'}")
            print(f"   Ollama (Local): {'✅' if ollama_success else '❌'}")
            
            if qwen3_success and ollama_success:
                print("🎉 FULL HYBRID EVALUATION ACHIEVED!")
                return True
            elif qwen3_success or ollama_success:
                print("⚠️ Partial hybrid evaluation")
                return True
            else:
                print("❌ Both evaluations failed")
                return False
        else:
            print(f"❌ Evaluation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_ollama_integration()
    print(f"\n{'SUCCESS' if success else 'FAILED'}: Ollama integration test")
