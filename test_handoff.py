import unittest
from orchestration import qwen_ollama_handoff
from agents import FitnessScorer
from protocol import builder_output_fitness

class TestHandoff(unittest.TestCase):
    def test_handoff(self):
        task = "Test task for handoff"
        result = qwen_ollama_handoff(task)
        self.assertTrue(result["success"])
        
        specialist = FitnessScorer()
        handoff_result = specialist.receive_a2a_handoff(result["handoff"])
        self.assertTrue(handoff_result["success"])
        
        fitness = builder_output_fitness(str(handoff_result))
        self.assertGreater(fitness, 0.70)

    def test_low_fitness_handoff(self):
        low_fitness_handoff = {"checklist": "Failed task", "fitness_requirement": 0.70}
        specialist = FitnessScorer()
        result = specialist.receive_a2a_handoff(low_fitness_handoff)
        self.assertFalse(result["success"])
        self.assertEqual(result["halt_reason"], "low_fitness")

if __name__ == "__main__":
    unittest.main()
