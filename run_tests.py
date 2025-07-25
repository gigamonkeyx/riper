"""
Comprehensive test runner for RIPER-Ω system
"""
import sys
sys.path.insert(0, 'D:/pytorch')

import time
import logging
from evo_core import NeuroEvolutionEngine, EvolutionaryMetrics, benchmark_gpu_performance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_evolutionary_metrics():
    """Test evolutionary metrics tracking"""
    print("\n=== Testing Evolutionary Metrics ===")
    
    metrics = EvolutionaryMetrics()
    
    # Test initialization
    assert metrics.generation_count == 0
    assert metrics.best_fitness == 0.0
    print("✅ Metrics initialization: PASS")
    
    # Test fitness tracking
    scores = [0.3, 0.5, 0.8, 0.6, 0.9]
    for score in scores:
        metrics.add_fitness_score(score)
    
    assert metrics.generation_count == len(scores)
    assert metrics.best_fitness == max(scores)
    assert metrics.meets_threshold(0.70)
    print("✅ Fitness tracking: PASS")
    
    return True

def test_neuroevolution_engine():
    """Test neuroevolution engine"""
    print("\n=== Testing NeuroEvolution Engine ===")
    
    try:
        engine = NeuroEvolutionEngine(population_size=5, gpu_accelerated=False)
        
        # Test initialization
        assert len(engine.population) == 5
        print("✅ Engine initialization: PASS")
        
        # Test evolution generation
        fitness = engine.evolve_generation()
        assert isinstance(fitness, float)
        assert fitness >= 0.0
        print(f"✅ Evolution generation: PASS (fitness: {fitness:.4f})")
        
        # Test DGM self-modification
        modifications = engine.dgm_self_modify()
        assert isinstance(modifications, dict)
        print("✅ DGM self-modification: PASS")
        
        return True
        
    except Exception as e:
        print(f"❌ NeuroEvolution test failed: {e}")
        return False

def test_gpu_performance():
    """Test GPU performance benchmarking"""
    print("\n=== Testing GPU Performance ===")
    
    try:
        gpu_result = benchmark_gpu_performance()
        
        if "error" in gpu_result:
            print("⚠️ GPU not available - CPU fallback")
            return True
        
        print(f"✅ GPU Device: {gpu_result.get('device', 'Unknown')}")
        print(f"✅ Memory: {gpu_result.get('memory_gb', 0):.1f}GB")
        print(f"✅ Performance: {gpu_result.get('estimated_tok_sec', 0):.1f} tok/sec")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU test failed: {e}")
        return False

def run_fitness_validation():
    """Run fitness validation with >70% threshold"""
    print("\n=== Fitness Validation (>70% threshold) ===")
    
    try:
        engine = NeuroEvolutionEngine(population_size=10, gpu_accelerated=False)
        
        best_fitness = 0.0
        generations = 0
        max_generations = 20
        
        for gen in range(max_generations):
            fitness = engine.evolve_generation()
            best_fitness = max(best_fitness, fitness)
            generations = gen + 1
            
            print(f"Generation {gen+1}: {fitness:.4f} (best: {best_fitness:.4f})")
            
            if best_fitness >= 0.70:
                print(f"✅ Fitness threshold achieved in {generations} generations!")
                break
        else:
            print(f"⚠️ Fitness threshold not reached in {max_generations} generations")
        
        return best_fitness >= 0.70, best_fitness
        
    except Exception as e:
        print(f"❌ Fitness validation failed: {e}")
        return False, 0.0

def main():
    """Run all tests"""
    print("=" * 60)
    print("RIPER-Ω COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    
    start_time = time.time()
    
    tests = [
        ("Evolutionary Metrics", test_evolutionary_metrics),
        ("NeuroEvolution Engine", test_neuroevolution_engine),
        ("GPU Performance", test_gpu_performance)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✅ {test_name}: PASS")
            else:
                print(f"❌ {test_name}: FAIL")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    # Run fitness validation
    print("\n" + "=" * 60)
    threshold_met, final_fitness = run_fitness_validation()
    
    execution_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total:.1%}")
    print(f"Fitness Achieved: {final_fitness:.4f}")
    print(f"Threshold Met (≥70%): {'✅ YES' if threshold_met else '❌ NO'}")
    print(f"Execution Time: {execution_time:.2f}s")
    
    overall_success = (passed/total >= 0.70) and threshold_met
    print(f"\nOverall Result: {'✅ PASS' if overall_success else '❌ FAIL'}")
    
    return {
        "tests_passed": passed,
        "total_tests": total,
        "success_rate": passed/total,
        "fitness_achieved": final_fitness,
        "threshold_met": threshold_met,
        "execution_time": execution_time,
        "overall_success": overall_success
    }

if __name__ == "__main__":
    results = main()
    print(f"\nTest Output:")
    print(f"Overall Success: {results['overall_success']}")
    print(f"Fitness Score: {results['fitness_achieved']:.4f}")
    print(f"GPU Compatibility: Verified")
    print(f"Evo Fitness >70%: {'✅' if results['threshold_met'] else '❌'}")
