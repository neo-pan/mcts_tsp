from dataclasses import dataclass
from typing import List

@dataclass
class TSP_Result:
    Concorde_Distance: float
    MCTS_Distance: float
    Gap: float
    Time: float
    Solution: List[int]