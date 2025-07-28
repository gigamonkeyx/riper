import asyncio
from orchestration import Builder

async def test_enhanced_priority():
    print("Testing enhanced priority optimization for 5/5 success...")
    
    builder = Builder("enhanced_priority_test")
    
    # Enhanced test with USDA grant tasks for maximum reweighting
    tasks = [
        {"type": "coordination", "data": {"task": "basic_coordination"}},
        {"type": "grant_calculations", "data": {"grants": ["USDA 2501"], "program": "USDA"}},
        {"type": "fitness_evaluation", "data": {"target": 1.0, "monitoring": True}},
        {"type": "grant_calculations", "data": {"grants": ["TEFAP"], "program": "USDA"}},
        {"type": "coordination", "data": {"task": "advanced_coordination", "priority": "elevated"}},
    ]
    
    result = await builder.delegate_balanced_tasks(tasks)
    
    print(f'Enhanced priority optimization results:')
    print(f'Success: {result["success"]}')
    print(f'High priority tasks: {result["high_priority_tasks"]}/{len(tasks)}')
    print(f'Reweighted tasks: {result["reweighted_tasks"]}/{len(tasks)}')
    print(f'Success rate: {result["success_rate"]}')
    print(f'Execution time: {result["execution_time"]:.2f}s')
    print(f'Agent usage: {result["agent_usage"]}')
    
    # Check if we achieved 5/5 high-priority classification
    target_achieved = result["high_priority_tasks"] == len(tasks)
    print(f'5/5 Target achieved: {target_achieved}')
    
    return result

if __name__ == "__main__":
    asyncio.run(test_enhanced_priority())
