"""
Ollama Specialists Demo for RIPER-Ω System
"""
import sys
sys.path.insert(0, 'D:/pytorch')
from agents import FitnessScorer, TTSHandler, SwarmCoordinator

print('=== OLLAMA SPECIALISTS DEMO ===')

# Test FitnessScorer
print('\n1. Testing FitnessScorer:')
scorer = FitnessScorer()
task_data = {'generation': 5, 'fitness': 0.65}
result = scorer.process_task(task_data, generation=5, current_fitness=0.65)
print(f'FitnessScorer result: {result.success}')
print(f'Fitness score: {result.data.get("fitness_score", "N/A")}')
print(f'GPU utilized: {result.gpu_utilized}')

# Test TTSHandler  
print('\n2. Testing TTSHandler:')
tts = TTSHandler()
text_input = 'RIPER-Omega evolutionary algorithms optimize neural networks'
result = tts.process_task(text_input, text=text_input)
print(f'TTSHandler result: {result.success}')
print(f'Audio generated: {result.data.get("audio_generated", False)}')
print(f'Optimized text length: {len(result.data.get("optimized_text", ""))}')

# Test SwarmCoordinator
print('\n3. Testing SwarmCoordinator:')
coordinator = SwarmCoordinator()
swarm_task = {'objective': 'parallel_fitness_evaluation'}
result = coordinator.process_task(swarm_task, task_type='fitness', parallel_agents=2)
print(f'SwarmCoordinator result: {result.success}')
print(f'Success rate: {result.data.get("success_rate", 0):.1%}')
print(f'Parallel agents: {result.data.get("parallel_agents", 0)}')

print('\nOllama Specialists Demo: ✅ COMPLETE')
