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

```python
from mcts_tsp import parallel_mcts_solve

concorde_distances, mcts_distances, gaps, times, solutions, lengths_times = parallel_mcts_solve(
    city_num=3,
    distances_list=[
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ],
    opt_solutions=[
        np.array([1, 2, 3]),
        np.array([1, 3, 2]),
        np.array([2, 1, 3])
    ],
    heatmaps=[
        np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5]
        ]),
        np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5]
        ]),
        np.array([
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5]
        ])
    ],
    num_threads=4,
    alpha=1,
    beta=50,
    param_h=2,
    param_t=10,
    max_candidate_num=5,
    candidate_use_heatmap=1,
    max_depth=100,
)
```

## Credit

This project is based on the original work of [Spider-scnu/TSP](https://github.com/Spider-scnu/TSP), which is licensed under the MIT License.