#!/usr/bin/env python3
"""
Test RIPER-Ω evolution with 70 generations vs 7 generations
Compare performance, time, and fitness improvements
"""

import time
from evo_core import NeuroEvolutionEngine
import torch

def test_evolution_comparison():
    """Compare 7 vs 70 generation evolution using actual RIPER-Ω system"""

    print("🧪 RIPER-Ω EVOLUTION COMPARISON TEST")
    print("="*60)

    # Test parameters based on actual system
    population_size = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    results = {}

    # Test 1: Current 7 generations (actual system behavior)
    print("\n🔬 TEST 1: Current 7 Generations (Actual System)")
    print("-" * 40)

    evo_7 = NeuroEvolutionEngine(
        population_size=population_size,
        mutation_rate=0.05,
        crossover_rate=0.7,
        device=device
    )
    
    start_time = time.time()
    fitness_history_7 = []
    
    for gen in range(7):
        fitness_scores = evo_7.evaluate_population()
        best_fitness = max(fitness_scores)
        fitness_history_7.append(best_fitness)
        
        print(f"   Generation {gen+1}: Fitness = {best_fitness:.4f}")
        
        if best_fitness >= 0.70:
            print(f"   ✅ Target fitness achieved in generation {gen+1}")
            break
            
        evo_7.evolve()
    
    time_7 = time.time() - start_time
    final_fitness_7 = max(fitness_history_7)
    
    results['7_gen'] = {
        'time': time_7,
        'final_fitness': final_fitness_7,
        'generations': len(fitness_history_7),
        'fitness_history': fitness_history_7
    }
    
    print(f"   📊 Results: {final_fitness_7:.4f} fitness in {time_7:.2f}s")
    
    # Test 2: Extended 70 generations
    print("\n🔬 TEST 2: Extended 70 Generations")
    print("-" * 40)
    
    evo_70 = NeuroEvolution(
        population_size=population_size,
        input_size=10,
        hidden_size=20,
        output_size=5,
        device=device
    )
    
    start_time = time.time()
    fitness_history_70 = []
    
    for gen in range(70):
        fitness_scores = evo_70.evaluate_population()
        best_fitness = max(fitness_scores)
        fitness_history_70.append(best_fitness)
        
        if gen % 10 == 0 or gen < 10:
            print(f"   Generation {gen+1}: Fitness = {best_fitness:.4f}")
        
        if best_fitness >= 0.95:  # Higher target for extended run
            print(f"   🎯 High fitness achieved in generation {gen+1}")
            
        evo_70.evolve()
    
    time_70 = time.time() - start_time
    final_fitness_70 = max(fitness_history_70)
    
    results['70_gen'] = {
        'time': time_70,
        'final_fitness': final_fitness_70,
        'generations': len(fitness_history_70),
        'fitness_history': fitness_history_70
    }
    
    print(f"   📊 Results: {final_fitness_70:.4f} fitness in {time_70:.2f}s")
    
    # Analysis
    print("\n📈 COMPARATIVE ANALYSIS")
    print("="*60)
    
    improvement = final_fitness_70 - final_fitness_7
    time_ratio = time_70 / time_7
    efficiency_7 = final_fitness_7 / time_7
    efficiency_70 = final_fitness_70 / time_70
    
    print(f"📊 PERFORMANCE METRICS:")
    print(f"   7 Gen:  {final_fitness_7:.4f} fitness in {time_7:.2f}s")
    print(f"   70 Gen: {final_fitness_70:.4f} fitness in {time_70:.2f}s")
    print(f"   Improvement: +{improvement:.4f} fitness ({improvement/final_fitness_7*100:.1f}%)")
    print(f"   Time Cost: {time_ratio:.1f}x longer")
    print(f"   Efficiency (fitness/second):")
    print(f"     7 Gen:  {efficiency_7:.4f}")
    print(f"     70 Gen: {efficiency_70:.4f}")
    
    # Recommendations
    print(f"\n🎯 RECOMMENDATIONS:")
    
    if improvement > 0.1 and efficiency_70 > efficiency_7 * 0.5:
        print("   ✅ RECOMMEND 70 GENERATIONS:")
        print("     - Significant fitness improvement")
        print("     - Acceptable efficiency trade-off")
        print("     - Better for production optimization")
    elif improvement > 0.05:
        print("   ⚖️ CONDITIONAL RECOMMENDATION:")
        print("     - Moderate improvement, consider use case")
        print("     - Use 70 gen for offline optimization")
        print("     - Use 7 gen for real-time applications")
    else:
        print("   ❌ STICK WITH 7 GENERATIONS:")
        print("     - Minimal improvement for 10x cost")
        print("     - Diminishing returns evident")
        print("     - Current approach is optimal")
    
    # Bakery-specific analysis
    print(f"\n🏭 BAKERY SIMULATION IMPACT:")
    
    if final_fitness_70 > 0.85:
        print("   🍞 High fitness (>0.85) could enable:")
        print("     - More precise production scheduling")
        print("     - Better resource allocation")
        print("     - Reduced waste and higher profits")
        print("     - Advanced workflow coordination")
    
    if time_70 > 60:  # More than 1 minute
        print("   ⏱️ Extended time considerations:")
        print("     - May impact real-time decision making")
        print("     - Better for overnight optimization runs")
        print("     - Consider parallel processing")
    
    return results

if __name__ == '__main__':
    try:
        results = test_evolution_comparison()
        print("\n🎉 Evolution comparison test completed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
