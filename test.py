import mcts_tsp
import numpy as np
import time


def generate_test_data(num_cities, num_instances):
    pos = np.random.rand(num_instances, num_cities, 2)
    distances = np.linalg.norm(pos[:, np.newaxis, :] - pos[:, :, np.newaxis], axis=-1)
    opt_solutions = np.array([np.random.permutation(num_cities) + 1 for _ in range(num_instances)])
    heatmaps = np.random.rand(num_instances, num_cities, num_cities)
    return distances, opt_solutions, heatmaps

def test_solve_one_instance(distances, opt_solution, heatmap, city_num):
    print("\nTesting solve_one_instance:")
    result = mcts_tsp.solve_one_instance(
        city_num=city_num,
        alpha=1,
        beta=10,
        param_h=10,
        param_t=0.1,
        max_candidate_num=5,
        candidate_use_heatmap=1,
        max_depth=10,
        distances=distances,
        opt_solution=opt_solution,
        heatmap=heatmap,
        debug=True
    )
    
    print(f"Concorde Distance: {result.Concorde_Distance}")
    print(f"MCTS Distance: {result.MCTS_Distance}")
    print(f"Gap: {result.Gap * 100:.2f}%")
    print(f"Time: {result.Time:.2f} seconds")
    print(f"Solution: {result.Solution}")
    
    return result

def test_parallel_mcts_solve(distances_list, opt_solutions, heatmaps, city_num, num_threads):
    print("\nTesting parallel_mcts_solve:")
    start_time = time.time()
    
    concorde_distances, mcts_distances, gaps, times, solutions = mcts_tsp.parallel_mcts_solve(
        city_num=city_num,
        distances_list=distances_list,
        opt_solutions=opt_solutions,
        heatmaps=heatmaps,
        num_threads=num_threads,
        alpha=1,
        beta=10,
        param_h=10,
        param_t=0.1,
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
    
    return concorde_distances, mcts_distances, gaps, times, solutions

def main():
    np.random.seed(42)  # For reproducibility
    
    # Test parameters
    num_cities = 20
    num_instances = 5
    num_threads = 2
    
    # Generate test data
    distances, opt_solutions, heatmaps = generate_test_data(num_cities, num_instances)
    
    # Test solve_one_instance
    single_result = test_solve_one_instance(distances[0], opt_solutions[0], heatmaps[0], num_cities)
    
    # Test parallel_mcts_solve
    parallel_results = test_parallel_mcts_solve(distances, opt_solutions, heatmaps, num_cities, num_threads)
    
    print("\nTest completed successfully!")

if __name__ == "__main__":
    main()