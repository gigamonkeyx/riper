"""
DGM Evolution Cycle with OpenRouter + Ollama Integration
100+ generations with hybrid fitness evaluation
"""

import os
import sys
import time
import json

sys.path.insert(0, "D:/pytorch")

# Set OpenRouter API key
os.environ[
    "OPENROUTER_API_KEY"
] = "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"

from evo_core import NeuroEvolutionEngine, EvolutionaryMetrics
from openrouter_client import get_openrouter_client

print("=== DGM EVOLUTION CYCLE (100+ GENERATIONS) ===")

# Initialize evolution engine
engine = NeuroEvolutionEngine(population_size=20, gpu_accelerated=True)
metrics = EvolutionaryMetrics()
client = get_openrouter_client()

print(f"✅ Evolution engine initialized: {len(engine.population)} individuals")
print(f"✅ GPU acceleration: {engine.gpu_accelerated}")

# Evolution parameters
MAX_GENERATIONS = 100
FITNESS_TARGET = 0.70
HYBRID_ANALYSIS_INTERVAL = 10  # Analyze every 10 generations

best_fitness = 0.0
generation_data = []
hybrid_analyses = []

print(f"\nStarting evolution cycle:")
print(f"Target: {FITNESS_TARGET} fitness in {MAX_GENERATIONS} generations")
print(f"Hybrid analysis every {HYBRID_ANALYSIS_INTERVAL} generations")

start_time = time.time()

for generation in range(MAX_GENERATIONS):
    gen_start = time.time()

    # Evolve generation
    fitness = engine.evolve_generation()
    best_fitness = max(best_fitness, fitness)

    # Record metrics
    metrics.add_fitness_score(fitness)

    gen_time = time.time() - gen_start
    generation_data.append(
        {
            "generation": generation + 1,
            "fitness": fitness,
            "best_fitness": best_fitness,
            "execution_time": gen_time,
        }
    )

    # Progress output
    if (generation + 1) % 10 == 0 or fitness >= FITNESS_TARGET:
        print(
            f"Gen {generation + 1:3d}: {fitness:.4f} (best: {best_fitness:.4f}) [{gen_time:.2f}s]"
        )

    # Hybrid analysis at intervals
    if (generation + 1) % HYBRID_ANALYSIS_INTERVAL == 0:
        try:
            analysis_data = {
                "generation": generation + 1,
                "current_fitness": fitness,
                "best_fitness": best_fitness,
                "fitness_trend": [g["fitness"] for g in generation_data[-10:]],
                "population_size": len(engine.population),
                "gpu_accelerated": engine.gpu_accelerated,
            }

            # Get OpenRouter analysis
            response = client.qwen3_fitness_analysis(analysis_data, generation + 1)

            if response.success:
                hybrid_analyses.append(
                    {
                        "generation": generation + 1,
                        "analysis": response.content,
                        "execution_time": response.execution_time,
                        "source": "openrouter",
                    }
                )
                print(
                    f"    ✅ Qwen3 analysis completed ({response.execution_time:.1f}s)"
                )
            else:
                print(f"    ⚠️ Qwen3 analysis failed: {response.error_message}")

        except Exception as e:
            print(f"    ❌ Hybrid analysis error: {e}")

    # DGM self-modification every 20 generations
    if (generation + 1) % 20 == 0:
        print(f"    🔄 Applying DGM self-modification...")
        modifications = engine.dgm_self_modify()

        if "error" not in modifications:
            mod_count = (
                sum(modifications.values()) if isinstance(modifications, dict) else 0
            )
            print(f"    ✅ DGM applied {mod_count} modifications")
        else:
            print(f'    ⚠️ DGM: {modifications.get("error", "Unknown error")}')

    # Check if target achieved
    if best_fitness >= FITNESS_TARGET:
        print(f"\n🎉 FITNESS TARGET ACHIEVED!")
        print(f"Generation {generation + 1}: {best_fitness:.4f} >= {FITNESS_TARGET}")
        break

total_time = time.time() - start_time

# Final results
print("\n=== EVOLUTION CYCLE RESULTS ===")
print(f"Generations completed: {len(generation_data)}")
print(f"Final best fitness: {best_fitness:.4f}")
print(f'Target achieved: {"✅ YES" if best_fitness >= FITNESS_TARGET else "❌ NO"}')
print(f"Total execution time: {total_time:.1f}s")
print(f"Average time per generation: {total_time/len(generation_data):.2f}s")

# Fitness analysis
if len(generation_data) >= 10:
    recent_fitness = [g["fitness"] for g in generation_data[-10:]]
    improvement = (
        recent_fitness[-1] - recent_fitness[0] if len(recent_fitness) > 1 else 0
    )
    print(f"Recent improvement (last 10 gen): {improvement:+.4f}")

# Hybrid analysis summary
print(f"\nHybrid analyses performed: {len(hybrid_analyses)}")
if hybrid_analyses:
    avg_analysis_time = sum(a["execution_time"] for a in hybrid_analyses) / len(
        hybrid_analyses
    )
    print(f"Average analysis time: {avg_analysis_time:.2f}s")
    print("✅ OpenRouter integration functional throughout evolution")

# GPU metrics check
try:
    import subprocess

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        memory_info = result.stdout.strip().split(", ")
        used_mb, total_mb = int(memory_info[0]), int(memory_info[1])
        print(
            f"GPU VRAM: {used_mb}MB/{total_mb}MB ({used_mb/1024:.1f}GB/{total_mb/1024:.1f}GB)"
        )

        if used_mb / 1024 < 8.0:
            print("✅ VRAM usage <8GB requirement met")
        else:
            print("⚠️ VRAM usage exceeds 8GB limit")
except:
    print("⚠️ Could not check GPU memory")

# Save detailed results
results = {
    "evolution_summary": {
        "generations": len(generation_data),
        "best_fitness": best_fitness,
        "target_achieved": best_fitness >= FITNESS_TARGET,
        "total_time": total_time,
        "avg_time_per_gen": total_time / len(generation_data),
    },
    "generation_data": generation_data,
    "hybrid_analyses": hybrid_analyses,
    "engine_config": {
        "population_size": len(engine.population),
        "gpu_accelerated": engine.gpu_accelerated,
        "fitness_target": FITNESS_TARGET,
    },
}

with open("dgm_evolution_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\n✅ Results saved to dgm_evolution_results.json")
print(
    f'✅ DGM Evolution Cycle: {"SUCCESSFUL" if best_fitness >= FITNESS_TARGET else "PARTIAL SUCCESS"}'
)
