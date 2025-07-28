from pre_sim_validation import PreSimValidator
import time

def test_async_timeouts():
    start = time.time()
    validator = PreSimValidator()
    result = validator.validate_pre_sim()
    elapsed = time.time() - start
    
    print(f'Async timeout test: {elapsed:.2f}s')
    print(f'Compliance: {result["compliance"]}')
    print(f'Completed cycles: {result["completed_cycles"]}')
    print(f'Average fitness: {result["average_fitness"]:.3f}')
    
    return result

if __name__ == "__main__":
    test_async_timeouts()
