import asyncio
from orchestration import Builder

async def test_load_balancing():
    builder = Builder("test_builder")
    
    tasks = [
        {"type": "grant_calculations", "data": {"grants": ["USDA 2501", "We Feed WA"]}},
        {"type": "coordination", "data": {"agents": ["observer", "builder"]}},
        {"type": "fitness_evaluation", "data": {"population": 50, "generation": 5}},
        {"type": "grant_calculations", "data": {"grants": ["TEFAP", "CSFP"]}},
    ]
    
    result = await builder.delegate_balanced_tasks(tasks)
    
    print(f'Load balancing test:')
    print(f'Success: {result["success"]}')
    print(f'Execution time: {result["execution_time"]:.2f}s')
    print(f'Agent usage: {result["agent_usage"]}')
    print(f'Tasks processed: {len(result["results"])}')
    
    return result

if __name__ == "__main__":
    asyncio.run(test_load_balancing())
