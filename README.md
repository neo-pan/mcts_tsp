# MCTS TSP Solver

This is a Python wrapper for the MCTS TSP solver. The MCTS TSP solver is a parallelized Monte Carlo Tree Search solver for the Traveling Salesman Problem (TSP). It is implemented in C++ and wrapped with Pybind11 for easy use in Python.

## Installation

```bash
pip install .
```

## Usage

This package provides one main function:

### parallel_mcts_solve

Solves multiple TSP instances in parallel using MCTS.
> **Notice**  
> It is recommended to normalize the coordinates to the range [0, 1] for better performance.  
> Please note that the indices of `opt_solutions` and the returned `solutions` are 0-based (i.e., start from 0).

```python
import numpy as np
from mcts_tsp import parallel_mcts_solve

coordinates = np.random.rand(4, 20, 2) # 4 instances of 20 cities with 2 dimsional coordinates
opt_solutions = np.array([np.random.permutation(20) for _ in range(4)]) # fake optimal solutions
heatmaps = np.random.rand(4, 20, 20) # fake heatmaps

concorde_distances, mcts_distances, gaps, times, overall_times, solutions, lengths_times = parallel_mcts_solve(
    city_num=20,
    coordinates_list=coordinates,
    opt_solutions=opt_solutions,
    heatmaps=heatmaps,
    num_threads=4,
    alpha=1,
    beta=50,
    param_h=2,
    param_t=0.1,
    max_candidate_num=5,
    candidate_use_heatmap=1,
    max_depth=10,
    log_len_time=True, # Record the length-time record during the MCTS search
    debug=False, # Print debug information
)

# The output values from parallel_mcts_solve can now be accessed
print(concorde_distances)   # distances based on input opt_solutions, which may not be the optimal solution
print(mcts_distances)       # distances based on the MCTS solution
print(gaps)                 # gap between the concorde and mcts distances
print(times)                # time taken by MCTS to solve each instance
print(overall_times)        # total time taken to solve each instance (including the IO and initialization time)
print(solutions)            # MCTS solutions
print(lengths_times)        # detailed length-time record during the MCTS search
```

## Credit

This project is based on the original work of [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP), which is licensed under the MIT License.