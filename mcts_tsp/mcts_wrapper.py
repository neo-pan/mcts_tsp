import numpy as np
from . import _mcts_cpp as mcts
from .mcts_types import TSP_Result

def solve_one_instance(
    coordinates: np.ndarray,
    opt_solution: np.ndarray,
    heatmap: np.ndarray,
    city_num: int,
    alpha: float,
    beta: float,
    param_h: float,
    param_t: float,
    max_candidate_num: int,
    candidate_use_heatmap: int,
    max_depth: int,
    log_len_time: bool = False,
    debug: bool = False
) -> TSP_Result:
    if (2 * max_depth > city_num):
        raise ValueError("max_depth should be less than city_num/2")
    return mcts.solve(
        city_num,
        alpha,
        beta,
        param_h,
        param_t,
        max_candidate_num,
        candidate_use_heatmap,
        max_depth,
        coordinates,
        opt_solution,
        heatmap,
        log_len_time,
        debug
    )