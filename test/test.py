import mcts_tsp
import numpy as np
import time


def generate_test_data(num_cities, num_instances):
    pos = np.random.rand(num_instances, num_cities, 2)
    distances = np.linalg.norm(pos[:, np.newaxis, :] - pos[:, :, np.newaxis], axis=-1)
    opt_solutions = np.array([np.random.permutation(num_cities) + 1 for _ in range(num_instances)])
    heatmaps = np.random.rand(num_instances, num_cities, num_cities)
    return distances, opt_solutions, heatmaps


def test_parallel_mcts_solve(distances_list, opt_solutions, heatmaps, city_num, num_threads):
    print("\nTesting parallel_mcts_solve:")
    start_time = time.time()
    
    concorde_distances, mcts_distances, gaps, times, solutions, lengths_times = mcts_tsp.parallel_mcts_solve(
        city_num=city_num,
        distances_list=distances_list,
        opt_solutions=opt_solutions,
        heatmaps=heatmaps,
        num_threads=num_threads,
        alpha=1,
        beta=10,
        param_h=10,
        param_t=10./city_num,
        max_candidate_num=5,
        candidate_use_heatmap=1,
        max_depth=10,
        debug=False
    )
    
    total_time = time.time() - start_time
    
    print(f"Average Concorde Distance: {np.mean(concorde_distances):.2f}")
    print(f"Average MCTS Distance: {np.mean(mcts_distances):.2f}")
    print(f"Average Gap: {np.mean(gaps) * 100:.2f}%")
    print(f"Average Time per Instance: {np.mean(times):.2f} seconds")
    print(f"Total Time: {total_time:.2f} seconds")
    print(f"Length Time: {len(lengths_times)} instances")
    print(lengths_times)
    for i, length_time in enumerate(lengths_times):
        print(f"Instance {i}: {len(length_time)} records")

    return concorde_distances, mcts_distances, gaps, times, solutions

def main():
    np.random.seed(42)  # For reproducibility
    
    # Test parameters
    num_cities = 1000
    num_instances = 4
    num_threads = 4
    
    # Generate test data
    distances, opt_solutions, heatmaps = generate_test_data(num_cities, num_instances)
  
    # Test parallel_mcts_solve
    parallel_results = test_parallel_mcts_solve(distances, opt_solutions, heatmaps, num_cities, num_threads)    
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()