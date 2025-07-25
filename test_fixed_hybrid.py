#!/usr/bin/env python3
"""
QA validation for fixed hybrid system
"""

import sys
import time
sys.path.insert(0, 'D:/pytorch')

from agents import FitnessScorer, check_gpu_memory

def test_fixed_hybrid():
    """Test fixed hybrid system with all improvements"""
    print("=== QA VALIDATION: Fixed Hybrid System ===")
    
    # Check GPU before test
    gpu_info = check_gpu_memory()
    print(f"GPU Status: {gpu_info.get('used_mb', 0)}MB / {gpu_info.get('total_mb', 0)}MB")
    
    try:
        scorer = FitnessScorer()
        print(f"✅ FitnessScorer initialized: {scorer.model_name}")
        
        # Test hybrid evaluation
        start_time = time.time()
        result = scorer.process_task(
            {
                "test_data": "qa_validation",
                "target_fitness": 0.75
            },
            generation=1,
            current_fitness=0.65
        )
        execution_time = time.time() - start_time
        
        if result.success:
            fitness_score = result.data.get('fitness_score', 0)
            evaluation = result.data.get('evaluation', {})
            evaluations = evaluation.get('evaluations', {})
            
            qwen3_success = evaluations.get('qwen3', {}).get('success', False)
            ollama_success = evaluations.get('ollama', {}).get('success', False)
            
            print(f"✅ Hybrid evaluation successful")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Fitness score: {fitness_score:.3f}")
            print(f"   Qwen3 (OpenRouter): {'✅' if qwen3_success else '❌'}")
            print(f"   Ollama (Local): {'✅' if ollama_success else '❌'}")
            
            # Validation criteria
            response_time_ok = execution_time < 30.0
            fitness_ok = fitness_score >= 0.70
            hybrid_ok = qwen3_success and ollama_success
            
            print(f"\n=== VALIDATION RESULTS ===")
            print(f"Response time <30s: {'✅' if response_time_ok else '❌'} ({execution_time:.1f}s)")
            print(f"Fitness >70%: {'✅' if fitness_ok else '❌'} ({fitness_score:.1f}%)")
            print(f"Full hybrid: {'✅' if hybrid_ok else '⚠️'} (partial)")
            
            if response_time_ok and fitness_ok:
                print("🎉 QA VALIDATION PASSED")
                return True
            else:
                print("❌ QA VALIDATION FAILED")
                return False
        else:
            print(f"❌ Evaluation failed: {result.error_message}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_fixed_hybrid()
    print(f"\nValidation: {'PASSED' if success else 'FAILED'}")
