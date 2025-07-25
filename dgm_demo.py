"""
DGM Self-Modification Demo for RIPER-Ω System
"""
import sys
sys.path.insert(0, 'D:/pytorch')
from evo_core import NeuroEvolutionEngine
import time

print('=== DGM SELF-MODIFICATION DEMO ===')

# Create evolution engine
engine = NeuroEvolutionEngine(population_size=5, gpu_accelerated=False)

# Get initial fitness
print('Before DGM modification:')
initial_fitness = engine.evolve_generation()
print(f'Initial fitness: {initial_fitness:.4f}')

# Apply DGM self-modification
print('\nApplying DGM self-modification...')
modifications = engine.dgm_self_modify()
print(f'Modifications applied: {modifications}')

# Get post-modification fitness
print('\nAfter DGM modification:')
final_fitness = engine.evolve_generation()
print(f'Final fitness: {final_fitness:.4f}')

# Calculate improvement
improvement = ((final_fitness - initial_fitness) / initial_fitness) * 100 if initial_fitness > 0 else 0
print(f'Fitness improvement: {improvement:.2f}%')

threshold_met = final_fitness >= 0.70
status = "✅ MET" if threshold_met else "❌ NOT MET"
print(f'Fitness threshold (≥70%): {status} ({final_fitness:.1%})')

print('DGM Demo: ✅ COMPLETE')
