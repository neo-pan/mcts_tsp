# MCTS TSP Solver

This is a Python wrapper for the MCTS TSP solver. The MCTS TSP solver is a parallelized Monte Carlo Tree Search solver for the Traveling Salesman Problem (TSP). It is implemented in C++ and wrapped with Pybind11 for easy use in Python.

## Installation

```bash
pip install .
```

## Usage

This package provides two main functions:

### 1. solve_one_instance

Solves a single TSP instance.

```python
from mcts_tsp import solve_one_instance

result = solve_one_instance(
    distances=np.array([
        [0, 1, 2],
        [1, 0, 3],
        [2, 3, 0]
    ]),
    opt_solution=np.array([1, 2, 3]),
    heatmap=np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5]
    ]),
    city_num=3,
    alpha=1,
    beta=50,
    param_h=2,
    param_t=10,
    max_candidate_num=5,
    candidate_use_heatmap=1,
    max_depth=100,
)
```

### 2. parallel_mcts_solve

Solves multiple TSP instances in parallel using MCTS.

```python
from mcts_tsp import parallel_mcts_solve

results = parallel_mcts_solve(
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
