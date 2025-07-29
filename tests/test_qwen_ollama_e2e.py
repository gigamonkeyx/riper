import unittest
from orchestration import qwen_ollama_handoff, Observer, Builder
from agents import FitnessScorer
from protocol import builder_output_fitness

class TestQwenOllamaE2E(unittest.TestCase):
    def test_e2e_handoff(self):
        task = "E2E test task"
        result = qwen_ollama_handoff(task)
        self.assertTrue(result["success"])
        
        observer = Observer()
        builder = Builder()
        
        # Simulate handoff chain
        specialist = FitnessScorer()
        handoff_result = specialist.receive_a2a_handoff(result["handoff"])
        self.assertTrue(handoff_result["success"])
        
        # Simulate observer coordination
        coordination_result = observer.coordinate_evolution(builder, None)  # None for evo_engine in test
        
        fitness = builder_output_fitness(str(handoff_result) + str(coordination_result))
        self.assertGreater(fitness, 0.70)

    def test_e2e_low_fitness(self):
        low_handoff = {"checklist": "Low fitness task", "fitness_requirement": 0.70}
        specialist = FitnessScorer()
        result = specialist.receive_a2a_handoff(low_handoff)
        self.assertFalse(result["success"])
        self.assertEqual(result["halt_reason"], "low_fitness")

if __name__ == "__main__":
    unittest.main()
