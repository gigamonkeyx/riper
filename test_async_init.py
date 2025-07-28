import asyncio
import time
from orchestration import AsyncSubAgentCoordinator

async def test_async_initialization():
    print("Testing async initialization optimization...")
    
    start_time = time.time()
    coordinator = AsyncSubAgentCoordinator()
    
    # Test async initialization
    await coordinator.async_initialize()
    
    init_time = time.time() - start_time
    
    print(f"Initialization completed in {init_time:.2f}s")
    print(f"Initialized: {coordinator.initialized}")
    print(f"Max concurrent: {coordinator.max_concurrent}")
    print(f"YAML parser available: {coordinator.yaml_parser is not None}")
    
    # Test simple task delegation
    if coordinator.initialized:
        test_task = {
            "agent": "swarm-coordinator",
            "data": {"test": "async_init", "context_limit": 8192}
        }
        
        result = await coordinator.delegate_task_async("swarm-coordinator", test_task["data"])
        print(f"Test delegation success: {result.get('success', False)}")
        
        if result.get('success'):
            print(f"Execution time: {result.get('execution_time', 0):.2f}s")
            print(f"Context limit: {result.get('context_limit', 'N/A')}")
    
    return coordinator

if __name__ == "__main__":
    asyncio.run(test_async_initialization())
