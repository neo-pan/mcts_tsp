
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

    if (Log_Length_Time)
    {
        Length_Time.push_back(
            std::make_pair(Current_Instance_Best_Distance, Get_Elapsed_Time(Current_Instance_Begin_Time)));
    }

    MCTS(); // Targeted sampling via MCTS within enlarged neighborhood

    if (Log_Length_Time)
    {
        Length_Time.push_back(
            std::make_pair(Current_Instance_Best_Distance, Get_Elapsed_Time(Current_Instance_Begin_Time)));
    }

    // Repeat the following process until termination
    while (Get_Elapsed_Time(Current_Instance_Begin_Time) < Param_T * Virtual_City_Num)
    {
        Jump_To_Random_State();
        Local_Search_by_2Opt_Move();

        if (Log_Length_Time)
        {
            Length_Time.push_back(
                std::make_pair(Current_Instance_Best_Distance, Get_Elapsed_Time(Current_Instance_Begin_Time)));
        }

        MCTS();

        if (Log_Length_Time)
        {
            Length_Time.push_back(
                std::make_pair(Current_Instance_Best_Distance, Get_Elapsed_Time(Current_Instance_Begin_Time)));
        }

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