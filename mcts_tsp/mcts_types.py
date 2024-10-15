from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TSP_Result:
    Concorde_Distance: float
    MCTS_Distance: float
    Gap: float
    Time: float
    Overall_Time: float
    Solution: List[int]
    Length_Time: List[Tuple[float, float]]