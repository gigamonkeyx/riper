from orchestration import tonasket_underserved_swarm

def test_optimized_swarm():
    result = tonasket_underserved_swarm()
    
    print(f'Optimized swarm success: {result["success"]}')
    print(f'Final fitness: {result.get("final_fitness", "N/A")}')
    
    return result

if __name__ == "__main__":
    test_optimized_swarm()
