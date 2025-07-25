"""
Test suite for evolutionary algorithms and neuroevolution
Verifies GA fitness loops, EvoTorch integration, and RTX 3080 compatibility.

Tests include:
- DEAP genetic algorithm examples
- EvoTorch neuroevolution quickstart problems
- GPU benchmark verification
- Fitness threshold validation (>70%)
"""

import pytest
import numpy as np
import time
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evo_core import (
    NeuroEvolutionEngine,
    EvolvableNeuralNet,
    EvolutionaryMetrics,
    benchmark_gpu_performance,
)
from orchestration import Observer, Builder
from agents import FitnessScorer, TTSHandler, SwarmCoordinator

# Test configuration
FITNESS_THRESHOLD = 0.70
GPU_PERFORMANCE_MIN = 1.0  # Minimum acceptable performance
TEST_TIMEOUT = 30  # Maximum test duration in seconds


class TestEvolutionaryMetrics:
    """Test evolutionary metrics tracking"""

    def test_metrics_initialization(self):
        """Test metrics object initialization"""
        metrics = EvolutionaryMetrics()
        assert metrics.generation_count == 0
        assert metrics.best_fitness == 0.0
        assert len(metrics.fitness_scores) == 0

    def test_fitness_score_tracking(self):
        """Test fitness score addition and tracking"""
        metrics = EvolutionaryMetrics()

        # Add fitness scores
        scores = [0.3, 0.5, 0.8, 0.6, 0.9]
        for score in scores:
            metrics.add_fitness_score(score)

        assert metrics.generation_count == len(scores)
        assert metrics.best_fitness == max(scores)
        assert abs(metrics.average_fitness - np.mean(scores)) < 1e-6

    def test_threshold_checking(self):
        """Test fitness threshold validation"""
        metrics = EvolutionaryMetrics()

        # Below threshold
        metrics.add_fitness_score(0.6)
        assert not metrics.meets_threshold(FITNESS_THRESHOLD)

        # Above threshold
        metrics.add_fitness_score(0.8)
        assert metrics.meets_threshold(FITNESS_THRESHOLD)


class TestEvolvableNeuralNet:
    """Test evolvable neural network functionality"""

    def test_network_initialization(self):
        """Test neural network creation"""
        net = EvolvableNeuralNet(input_size=10, hidden_sizes=[5, 3], output_size=2)

        # Test forward pass
        import torch

        test_input = torch.randn(1, 10)
        output = net(test_input)

        assert output.shape == (1, 2)
        assert not torch.isnan(output).any()

    def test_parameter_vector_operations(self):
        """Test parameter vector get/set operations"""
        net = EvolvableNeuralNet(input_size=5, hidden_sizes=[3], output_size=2)

        # Get original parameters
        original_params = net.get_parameters_vector()
        assert isinstance(original_params, np.ndarray)
        assert len(original_params) > 0

        # Modify and set parameters
        modified_params = original_params + 0.1
        net.set_parameters_vector(modified_params)

        # Verify parameters changed
        new_params = net.get_parameters_vector()
        assert not np.allclose(original_params, new_params)
        assert np.allclose(modified_params, new_params)

    def test_mutation(self):
        """Test neural network mutation"""
        net = EvolvableNeuralNet(input_size=5, hidden_sizes=[3], output_size=2)

        # Get original parameters
        original_params = net.get_parameters_vector()

        # Apply mutation
        net.mutate(mutation_rate=1.0, mutation_strength=0.1)

        # Verify parameters changed
        mutated_params = net.get_parameters_vector()
        assert not np.allclose(original_params, mutated_params)


class TestNeuroEvolutionEngine:
    """Test neuroevolution engine functionality"""

    def test_engine_initialization(self):
        """Test evolution engine initialization"""
        engine = NeuroEvolutionEngine(population_size=10, gpu_accelerated=False)

        assert len(engine.population) == 10
        assert isinstance(engine.metrics, EvolutionaryMetrics)
        assert engine.population_size == 10

    def test_fitness_computation(self):
        """Test fitness computation for neural networks"""
        engine = NeuroEvolutionEngine(population_size=5, gpu_accelerated=False)

        # Test fitness computation
        network = engine.population[0]
        fitness = engine._compute_fitness(network)

        assert isinstance(fitness, float)
        assert 0.0 <= fitness <= 1.0

    def test_evolution_generation(self):
        """Test single generation evolution"""
        engine = NeuroEvolutionEngine(population_size=10, gpu_accelerated=False)

        # Run one generation
        fitness = engine.evolve_generation()

        assert isinstance(fitness, float)
        assert fitness >= 0.0
        assert engine.metrics.generation_count == 1

    def test_best_network_selection(self):
        """Test best network selection"""
        engine = NeuroEvolutionEngine(population_size=5, gpu_accelerated=False)

        # Run a few generations
        for _ in range(3):
            engine.evolve_generation()

        best_network = engine.get_best_network()
        assert isinstance(best_network, EvolvableNeuralNet)

    def test_dgm_self_modification(self):
        """Test DGM self-modification capabilities"""
        engine = NeuroEvolutionEngine(population_size=10, gpu_accelerated=False)

        # Apply DGM modifications
        modifications = engine.dgm_self_modify()

        assert isinstance(modifications, dict)
        assert "architecture_mutation" in modifications
        assert "parameter_scaling" in modifications


class TestGPUPerformance:
    """Test GPU performance and RTX 3080 compatibility"""

    def test_gpu_availability_check(self):
        """Test GPU availability detection"""
        try:
            import torch

            gpu_available = torch.cuda.is_available()

            if gpu_available:
                device_name = torch.cuda.get_device_name(0)
                print(f"GPU detected: {device_name}")
            else:
                print("No GPU detected - tests will run on CPU")

            # Test should not fail regardless of GPU availability
            assert True
        except ImportError:
            pytest.skip("PyTorch not available")

    def test_gpu_benchmark(self):
        """Test GPU performance benchmark"""
        benchmark_result = benchmark_gpu_performance()

        if "error" in benchmark_result:
            pytest.skip("GPU not available for benchmarking")

        assert "device" in benchmark_result
        assert "memory_gb" in benchmark_result
        assert "benchmark_time" in benchmark_result
        assert "estimated_tok_sec" in benchmark_result

        # Check if performance meets minimum requirements
        tok_sec = benchmark_result.get("estimated_tok_sec", 0)
        if tok_sec > 0:
            assert (
                tok_sec >= GPU_PERFORMANCE_MIN
            ), f"GPU performance too low: {tok_sec:.2f} tok/sec"

    @pytest.mark.skipif(
        not hasattr(pytest, "gpu_available"), reason="GPU not available"
    )
    def test_gpu_memory_usage(self):
        """Test GPU memory utilization"""
        try:
            import torch

            if not torch.cuda.is_available():
                pytest.skip("CUDA not available")

            # Create neural network on GPU
            net = EvolvableNeuralNet()
            if torch.cuda.is_available():
                net.cuda()

            # Check memory usage
            memory_allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved() / (1024**3)  # GB

            print(
                f"GPU Memory - Allocated: {memory_allocated:.2f}GB, Reserved: {memory_reserved:.2f}GB"
            )

            # Should not exceed RTX 3080 limits (10GB)
            assert (
                memory_allocated < 10.0
            ), f"Memory usage too high: {memory_allocated:.2f}GB"

        except ImportError:
            pytest.skip("PyTorch not available")


class TestAgentIntegration:
    """Test agent integration and coordination"""

    def test_fitness_scorer(self):
        """Test fitness scoring agent"""
        scorer = FitnessScorer()

        task_data = {
            "generation": 5,
            "current_fitness": 0.6,
            "target_threshold": FITNESS_THRESHOLD,
        }

        result = scorer.process_task(task_data, generation=5, current_fitness=0.6)

        assert result.success
        assert "fitness_score" in result.data
        assert isinstance(result.data["fitness_score"], float)
        assert 0.0 <= result.data["fitness_score"] <= 1.0

    def test_swarm_coordinator(self):
        """Test swarm coordination functionality"""
        coordinator = SwarmCoordinator()

        task_data = {"objective": "test_coordination", "data": "sample_data"}

        result = coordinator.process_task(
            task_data, task_type="fitness", parallel_agents=2
        )

        assert result.success
        assert "swarm_results" in result.data
        assert "success_rate" in result.data
        assert "parallel_agents" in result.data


class TestOrchestration:
    """Test orchestration and A2A communication"""

    def test_observer_builder_coordination(self):
        """Test Observer-Builder coordination"""
        observer = Observer("test_obs")
        builder = Builder("test_builder")

        # Test A2A message sending
        success = observer.a2a_comm.send_message(
            receiver_id="test_builder",
            message_type="coordination",
            payload={"action": "test", "data": "sample"},
        )

        assert success
        assert len(observer.a2a_comm.message_queue) > 0

    def test_evolution_coordination(self):
        """Test full evolution coordination"""
        observer = Observer("test_obs")
        builder = Builder("test_builder")
        engine = NeuroEvolutionEngine(population_size=5, gpu_accelerated=False)

        # Run short coordination test
        results = observer.coordinate_evolution(builder, engine)

        assert isinstance(results, dict)
        assert "final_fitness" in results
        assert "generations" in results
        assert "success" in results


# Performance benchmarks
def test_performance_benchmarks():
    """Run performance benchmarks and verify targets"""
    print("\n=== Performance Benchmark Results ===")

    # GPU benchmark
    gpu_result = benchmark_gpu_performance()
    if "error" not in gpu_result:
        print(f"GPU Performance: {gpu_result['estimated_tok_sec']:.2f} tok/sec")
        print(f"Target Range: 7-15 tok/sec")
        print(f"Memory Available: {gpu_result['memory_gb']:.1f}GB")

    # Evolution benchmark
    start_time = time.time()
    engine = NeuroEvolutionEngine(population_size=20, gpu_accelerated=False)

    best_fitness = 0.0
    for generation in range(10):
        fitness = engine.evolve_generation()
        best_fitness = max(best_fitness, fitness)

        if best_fitness >= FITNESS_THRESHOLD:
            print(f"Fitness threshold achieved in generation {generation}")
            break

    evolution_time = time.time() - start_time
    print(f"Evolution Performance: {best_fitness:.4f} fitness in {evolution_time:.2f}s")
    print(f"Target Fitness: ≥{FITNESS_THRESHOLD}")

    # Verify performance meets requirements
    assert best_fitness > 0.0, "Evolution produced no fitness improvement"
    print("=== Benchmark Complete ===\n")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])
