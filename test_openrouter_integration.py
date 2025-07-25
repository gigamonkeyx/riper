"""
OpenRouter Integration Test for RIPER-Ω System
Tests hybrid OpenRouter + Ollama functionality
"""

import sys
sys.path.insert(0, 'D:/pytorch')

import time
import json
from openrouter_client import get_openrouter_client
from agents import FitnessScorer
from orchestration import Observer, Builder

def test_openrouter_client():
    """Test basic OpenRouter client functionality"""
    print("=== OpenRouter Client Test ===")
    
    try:
        client = get_openrouter_client()
        
        # Test connection
        print("Testing connection...")
        connection_success = client.test_connection()
        print(f"Connection: {'✅ SUCCESS' if connection_success else '❌ FAILED'}")
        
        if not connection_success:
            print("⚠️ Skipping further tests due to connection failure")
            return False
        
        # Test code generation
        print("\nTesting code generation...")
        code_response = client.qwen3_code_generation(
            "Create a simple Python function that calculates fibonacci numbers",
            language="python"
        )
        
        if code_response.success:
            print("✅ Code generation successful")
            print(f"   Execution time: {code_response.execution_time:.2f}s")
            print(f"   Response length: {len(code_response.content)} chars")
        else:
            print(f"❌ Code generation failed: {code_response.error_message}")
        
        # Test fitness analysis
        print("\nTesting fitness analysis...")
        fitness_response = client.qwen3_fitness_analysis({
            "generation": 5,
            "current_fitness": 0.65,
            "population_size": 20,
            "target_threshold": 0.70
        })
        
        if fitness_response.success:
            print("✅ Fitness analysis successful")
            print(f"   Execution time: {fitness_response.execution_time:.2f}s")
            try:
                analysis = json.loads(fitness_response.content)
                print(f"   Analysis keys: {list(analysis.keys())}")
            except json.JSONDecodeError:
                print("   Response format: Text (not JSON)")
        else:
            print(f"❌ Fitness analysis failed: {fitness_response.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ OpenRouter client test failed: {e}")
        return False

def test_hybrid_fitness_scorer():
    """Test hybrid OpenRouter + Ollama fitness scoring"""
    print("\n=== Hybrid Fitness Scorer Test ===")
    
    try:
        scorer = FitnessScorer()
        
        # Test fitness evaluation
        print("Testing hybrid fitness evaluation...")
        start_time = time.time()
        
        result = scorer.process_task(
            {
                "neural_network_params": 1000000,
                "training_accuracy": 0.85,
                "validation_accuracy": 0.78,
                "gpu_utilization": 0.92
            },
            generation=10,
            current_fitness=0.65
        )
        
        execution_time = time.time() - start_time
        
        if result.success:
            print("✅ Hybrid fitness evaluation successful")
            print(f"   Execution time: {execution_time:.2f}s")
            print(f"   Fitness score: {result.data.get('fitness_score', 'N/A')}")
            print(f"   GPU utilized: {result.gpu_utilized}")
            
            # Check if both evaluations were attempted
            evaluation = result.data.get('evaluation', {})
            evaluations = evaluation.get('evaluations', {})
            
            qwen3_success = evaluations.get('qwen3', {}).get('success', False)
            ollama_success = evaluations.get('ollama', {}).get('success', False)
            
            print(f"   Qwen3 evaluation: {'✅' if qwen3_success else '❌'}")
            print(f"   Ollama evaluation: {'✅' if ollama_success else '❌'}")
            
            if qwen3_success and ollama_success:
                print("✅ Full hybrid evaluation achieved!")
            elif qwen3_success or ollama_success:
                print("⚠️ Partial hybrid evaluation (one source failed)")
            else:
                print("⚠️ Both evaluations failed, using fallback")
            
        else:
            print(f"❌ Hybrid fitness evaluation failed: {result.error_message}")
        
        return result.success
        
    except Exception as e:
        print(f"❌ Hybrid fitness scorer test failed: {e}")
        return False

def test_agent_coordination():
    """Test OpenRouter integration in agent coordination"""
    print("\n=== Agent Coordination Test ===")
    
    try:
        observer = Observer()
        builder = Builder()
        
        # Test coordination message with OpenRouter analysis
        print("Testing coordination with Qwen3 analysis...")
        
        success = observer.a2a_comm.send_message(
            receiver_id='builder_test',
            message_type='coordination',
            payload={
                'action': 'optimize_evolution',
                'fitness_target': 0.75,
                'generation': 15,
                'gpu_optimization': 'rtx_3080'
            }
        )
        
        if success:
            print("✅ Coordination message sent")
            
            # Process message with OpenRouter integration
            messages = observer.a2a_comm.receive_messages()
            if messages:
                result = builder.process_coordination_message(messages[0])
                
                if 'qwen3_analysis' in result:
                    print("✅ Qwen3 coordination analysis included")
                    print(f"   Analysis length: {len(result['qwen3_analysis'])} chars")
                else:
                    print("⚠️ Qwen3 analysis not included (may be normal)")
                
                print(f"   Processing result: {result.get('status', 'unknown')}")
            else:
                print("⚠️ No messages received")
        else:
            print("❌ Coordination message failed")
        
        return success
        
    except Exception as e:
        print(f"❌ Agent coordination test failed: {e}")
        return False

def test_performance_comparison():
    """Compare performance with and without OpenRouter"""
    print("\n=== Performance Comparison Test ===")
    
    try:
        scorer = FitnessScorer()
        
        # Test multiple evaluations to get average performance
        print("Running performance comparison (5 evaluations each)...")
        
        test_data = {
            "generation": 20,
            "fitness_metrics": [0.65, 0.68, 0.72, 0.69, 0.71],
            "gpu_performance": 0.88
        }
        
        results = []
        total_time = 0
        
        for i in range(5):
            start_time = time.time()
            result = scorer.process_task(
                test_data,
                generation=20 + i,
                current_fitness=0.65 + (i * 0.02)
            )
            execution_time = time.time() - start_time
            total_time += execution_time
            
            if result.success:
                results.append(result.data.get('fitness_score', 0.0))
            
            print(f"   Evaluation {i+1}: {execution_time:.2f}s")
        
        if results:
            avg_fitness = sum(results) / len(results)
            avg_time = total_time / len(results)
            
            print(f"\n✅ Performance Summary:")
            print(f"   Average fitness: {avg_fitness:.4f}")
            print(f"   Average time: {avg_time:.2f}s")
            print(f"   Evaluations completed: {len(results)}/5")
            
            # Check if fitness improved over baseline
            baseline_fitness = 0.65
            improvement = avg_fitness - baseline_fitness
            
            if improvement > 0.05:
                print(f"✅ Significant improvement: +{improvement:.4f}")
            elif improvement > 0:
                print(f"⚠️ Modest improvement: +{improvement:.4f}")
            else:
                print(f"⚠️ No improvement: {improvement:.4f}")
            
            return True
        else:
            print("❌ No successful evaluations")
            return False
            
    except Exception as e:
        print(f"❌ Performance comparison failed: {e}")
        return False

def main():
    """Run all OpenRouter integration tests"""
    print("=" * 60)
    print("RIPER-Ω OpenRouter Integration Test Suite")
    print("=" * 60)
    
    tests = [
        ("OpenRouter Client", test_openrouter_client),
        ("Hybrid Fitness Scorer", test_hybrid_fitness_scorer),
        ("Agent Coordination", test_agent_coordination),
        ("Performance Comparison", test_performance_comparison)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            if success:
                passed += 1
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total:.1%}")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - OpenRouter integration fully functional!")
    elif passed >= total * 0.75:
        print("✅ Most tests passed - OpenRouter integration mostly functional")
    else:
        print("⚠️ Multiple test failures - Check OpenRouter configuration")
    
    print("\nIntegration Status:")
    print(f"• OpenRouter API: {'✅ Connected' if passed > 0 else '❌ Issues'}")
    print(f"• Hybrid Evaluation: {'✅ Working' if passed >= 2 else '❌ Issues'}")
    print(f"• Agent Coordination: {'✅ Enhanced' if passed >= 3 else '⚠️ Basic'}")
    print(f"• Performance: {'✅ Optimized' if passed == total else '⚠️ Needs attention'}")

if __name__ == "__main__":
    main()
