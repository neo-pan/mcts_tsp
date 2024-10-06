
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
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed_seconds = std::chrono::duration<float>(now - Current_Instance_Begin_Time).count();
        Length_Time.push_back(std::make_pair(Current_Instance_Best_Distance, elapsed_seconds));
    }

    MCTS(); // Targeted sampling via MCTS within enlarged neighborhood

    if (Log_Length_Time)
    {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed_seconds = std::chrono::duration<float>(now - Current_Instance_Begin_Time).count();
        Length_Time.push_back(std::make_pair(Current_Instance_Best_Distance, elapsed_seconds));
    }

    // Repeat the following process until termination
    while (
        std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - Current_Instance_Begin_Time).count() <
        Param_T * Virtual_City_Num)
    {
        Jump_To_Random_State();
        Local_Search_by_2Opt_Move();

        if (Log_Length_Time)
        {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed_seconds = std::chrono::duration<float>(now - Current_Instance_Begin_Time).count();
            Length_Time.push_back(std::make_pair(Current_Instance_Best_Distance, elapsed_seconds));
        }

        MCTS();

        if (Log_Length_Time)
        {
            auto now = std::chrono::high_resolution_clock::now();
            float elapsed_seconds = std::chrono::duration<float>(now - Current_Instance_Begin_Time).count();
            Length_Time.push_back(std::make_pair(Current_Instance_Best_Distance, elapsed_seconds));
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