import asyncio
from orchestration import Builder

async def test_priority_optimization():
    print("Testing priority-optimized sub-agent delegation...")
    
    builder = Builder("priority_test")
    
    # Mixed priority tasks to test optimization
    tasks = [
        {"type": "coordination", "data": {"task": "low_priority"}},
        {"type": "grant_calculations", "data": {"grants": ["USDA 2501"], "priority": "high"}},
        {"type": "fitness_evaluation", "data": {"target": 1.0, "priority": "high"}},
        {"type": "coordination", "data": {"task": "coordination_support"}},
        {"type": "grant_calculations", "data": {"grants": ["We Feed WA"], "priority": "high"}},
    ]
    
    result = await builder.delegate_balanced_tasks(tasks)
    
    print(f'Priority optimization test:')
    print(f'Success: {result["success"]}')
    print(f'High priority tasks: {result["high_priority_tasks"]}/{len(tasks)}')
    print(f'Execution time: {result["execution_time"]:.2f}s')
    print(f'Agent usage: {result["agent_usage"]}')
    print(f'Priority optimized: {result["priority_optimized"]}')
    
    return result

if __name__ == "__main__":
    asyncio.run(test_priority_optimization())
