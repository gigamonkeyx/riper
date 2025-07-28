import asyncio
import time
from orchestration import tonasket_underserved_swarm

async def test_optimized_integration():
    print("Testing optimized integration with async init and context limits...")
    
    start_time = time.time()
    
    try:
        # Run optimized swarm with timeout protection
        result = await asyncio.wait_for(
            asyncio.to_thread(tonasket_underserved_swarm),
            timeout=120.0  # 2 minute timeout
        )
        
        execution_time = time.time() - start_time
        
        print(f"Integration test completed in {execution_time:.2f}s")
        print(f"Success: {result.get('success', False)}")
        print(f"Final fitness: {result.get('final_fitness', 'N/A')}")
        print(f"YAML enhanced: {result.get('yaml_enhanced', False)}")
        
        if 'coordination_results' in result:
            successful_coords = sum(1 for r in result['coordination_results'] if r.get('success', False))
            total_coords = len(result['coordination_results'])
            print(f"Coordination success: {successful_coords}/{total_coords}")
        
        return result
        
    except asyncio.TimeoutError:
        print("Integration test timed out after 120s")
        return {"success": False, "error": "timeout", "execution_time": 120.0}
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"Integration test failed: {e}")
        return {"success": False, "error": str(e), "execution_time": execution_time}

if __name__ == "__main__":
    asyncio.run(test_optimized_integration())
