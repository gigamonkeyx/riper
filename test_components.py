#!/usr/bin/env python3

# Test individual optimized components
from economy_rewards import EconomyRewards
from orchestration import AsyncSubAgentCoordinator
import asyncio

def test_pgpe_fitness():
    print("Testing PGPE fitness optimization...")
    rewards = EconomyRewards()
    fitness = rewards.evotorch_fitness_calculation({'test': 'optimization'})
    print(f"PGPE fitness result: {fitness:.3f}")
    return fitness

async def test_priority_system():
    print("Testing priority optimization system...")
    coordinator = AsyncSubAgentCoordinator()
    await coordinator.async_initialize()
    
    print(f"Initialization successful: {coordinator.initialized}")
    print(f"Max concurrent tasks: {coordinator.max_concurrent}")
    return coordinator.initialized

def main():
    print("=== Component Optimization Tests ===")
    
    # Test PGPE
    pgpe_fitness = test_pgpe_fitness()
    
    # Test priority system
    priority_result = asyncio.run(test_priority_system())
    
    print(f"\n=== Results Summary ===")
    print(f"PGPE fitness: {pgpe_fitness:.3f}")
    print(f"Priority system: {'OK' if priority_result else 'FAILED'}")
    
    # Overall assessment
    if pgpe_fitness >= 0.5 and priority_result:
        print("✅ Component optimizations working")
        return True
    else:
        print("❌ Component optimizations need attention")
        return False

if __name__ == "__main__":
    main()
