#ifndef TSP_MARKOV_DECISION_H
#define TSP_MARKOV_DECISION_H

#include "TSP_MCTS.h"

// Jump to a new state by randomly generating a solution
void Jump_To_Random_State()
{
    Generate_Initial_Solution();
}

Distance_Type Markov_Decision_Process()
{
    MCTS_Init();                 // Initialize MCTS parameters
    Generate_Initial_Solution(); // State initialization of MDP
    Local_Search_by_2Opt_Move(); // 2-opt based local search within small
                                 // neighborhood
    MCTS(); // Targeted sampling via MCTS within enlarged neighborhood

    // Repeat the following process until termination
    while (Get_Elapsed_Time(Current_Instance_Begin_Time) < Param_T * Virtual_City_Num)
    {
        Jump_To_Random_State();
        Local_Search_by_2Opt_Move();
        MCTS();
        // Max_Depth = 10 + (rand() % 80);
    }

    // Copy information of the best found solution (stored in Struct_Node
    // *Best_All_Node ) to Struct_Node *All_Node
    Restore_Best_Solution();

    if (Check_Solution_Feasible())
        return Get_Solution_Total_Distance();
    else
        return Inf_Cost;
}

#endif // TSP_MARKOV_DECISION_H