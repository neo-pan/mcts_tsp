import concurrent.futures
from functools import partial
from .mcts_wrapper import solve_one_instance


def parallel_mcts_solve(city_num, distances_list, opt_solutions, heatmaps, num_threads, alpha=1, beta=10, param_h=10, param_t=0.1,
                        max_candidate_num=5, candidate_use_heatmap=1, max_depth=10, log_len_time=False, debug=False):
    solve_with_params = partial(solve_one_instance, city_num=city_num, alpha=alpha, beta=beta, param_h=param_h,
                                param_t=param_t, max_candidate_num=max_candidate_num,
                                candidate_use_heatmap=candidate_use_heatmap, max_depth=max_depth,
                                log_len_time=log_len_time, debug=debug)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(solve_with_params, distances_list, opt_solutions, heatmaps))

    concorde_distances = [result.Concorde_Distance for result in results]
    mcts_distances = [result.MCTS_Distance for result in results]
    gaps = [result.Gap for result in results]
    times = [result.Time for result in results]
    solutions = [result.Solution for result in results]
    lengths_times = [result.Length_Time for result in results]

    return concorde_distances, mcts_distances, gaps, times, solutions, lengths_times