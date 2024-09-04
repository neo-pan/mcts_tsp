import mcts
import numpy as np

pos = np.random.rand(10, 2)
distances = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2)
opt_solution = np.arange(10)
heatmap = np.ones((10, 10))

mcts.solve(10, 1, 10, 10, 0.1, 5, 1, 10, distances, opt_solution, heatmap, True)
