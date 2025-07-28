import asyncio
from orchestration import AsyncSubAgentCoordinator

async def test_async_coordination():
    coord = AsyncSubAgentCoordinator()
    
    tasks = [
        {'agent': 'grant-modeler', 'data': {'test': 1}},
        {'agent': 'swarm-coordinator', 'data': {'test': 2}},
        {'agent': 'fitness-evaluator', 'data': {'test': 3}}
    ]
    
    results = await coord.delegate_multiple_tasks(tasks)
    print('Async results:', len(results), 'tasks completed')
    
    successful = sum(1 for r in results if r.get('success', False))
    print(f'Success rate: {successful}/{len(tasks)}')
    
    return results

if __name__ == "__main__":
    asyncio.run(test_async_coordination())
