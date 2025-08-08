import os
import pytest
import torch

from evo_core import NeuroEvolutionEngine


def test_engine_device_consistency_cpu():
    # Force CPU
    engine = NeuroEvolutionEngine(population_size=3, gpu_accelerated=False)
    # Assert dataset tensors are on CPU
    assert not engine.gpu_accelerated
    assert engine._fitness_input.device.type == 'cpu'
    assert engine._fitness_target.device.type == 'cpu'
    # Assert model params/device match CPU for at least one network
    net = engine.population[0]
    assert next(net.parameters()).device.type == 'cpu'
    # Forward should not raise due to device mismatch
    import torch as T
    x = T.randn(2, net.input_size)
    _ = net(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_engine_device_consistency_cuda():
    engine = NeuroEvolutionEngine(population_size=3, gpu_accelerated=True)
    assert engine.gpu_accelerated
    assert engine._fitness_input.device.type == 'cuda'
    assert engine._fitness_target.device.type == 'cuda'
    net = engine.population[0]
    assert next(net.parameters()).device.type == 'cuda'
    import torch as T
    x = T.randn(2, net.input_size).cuda()
    _ = net(x)

