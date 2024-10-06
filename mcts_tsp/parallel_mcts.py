import concurrent.futures
from .mcts_wrapper import solve_one_instance

def parallel_mcts_solve(city_num, coordinates_list, opt_solutions, heatmaps, num_threads, alpha=1, beta=10, param_h=10, param_t=0.1,
                        max_candidate_num=5, candidate_use_heatmap=1, max_depth=10, log_len_time=False, debug=False):

    args = [
        (coordinates, solution, heatmap, city_num, alpha, beta, param_h, param_t, max_candidate_num, candidate_use_heatmap, max_depth, log_len_time, debug)
        for coordinates, solution, heatmap in zip(coordinates_list, opt_solutions, heatmaps)
    ]

    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(solve_one_instance, *arg) for arg in args]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())

    concorde_distances = [result.Concorde_Distance for result in results if result is not None]
    mcts_distances = [result.MCTS_Distance for result in results if result is not None]
    gaps = [result.Gap for result in results if result is not None]
    times = [result.Time for result in results if result is not None]
    solutions = [result.Solution for result in results if result is not None]
    lengths_times = [result.Length_Time for result in results if result is not None]

    return concorde_distances, mcts_distances, gaps, times, solutions, lengths_times
