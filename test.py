import mcts
import numpy as np

# 假设我们已经有了这些数据
pos = np.random.rand(10, 2)
distances = np.linalg.norm(pos[:, np.newaxis] - pos, axis=2)
opt_solution = np.arange(10) + 1  # 最优解（这里只是一个示例）
heatmap = np.ones((10, 10))  # 热力图（这里使用全1矩阵作为示例）

result = mcts.solve(
    city_num=10,
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

print("求解结果：")
print(f"Concorde 距离: {result.Concorde_Distance}")
print(f"MCTS 距离: {result.MCTS_Distance}")
print(f"差距: {result.Gap * 100:.2f}%")
print(f"耗时: {result.Time:.2f} 秒")
print("解决方案:")
print(result.Solution)

# 如果你想要更详细的输出，可以直接打印整个 result 对象
print("\n完整结果对象：")
print(result)
