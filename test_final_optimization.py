import time
from orchestration import tonasket_underserved_swarm

def test_final_optimization():
    print("Testing final optimization with PGPE tuning and priority sub-agents...")
    
    start = time.time()
    result = tonasket_underserved_swarm()
    elapsed = time.time() - start
    
    print(f'Optimized swarm results:')
    print(f'Success: {result["success"]}')
    print(f'Final fitness: {result.get("final_fitness", "N/A")}')
    print(f'Execution time: {elapsed:.2f}s')
    print(f'YAML enhanced: {result.get("yaml_enhanced", False)}')
    
    if 'coordination_results' in result:
        successful_coords = sum(1 for r in result['coordination_results'] if r.get('success', False))
        total_coords = len(result['coordination_results'])
        print(f'Coordination success: {successful_coords}/{total_coords}')
    
    return result

if __name__ == "__main__":
    test_final_optimization()
