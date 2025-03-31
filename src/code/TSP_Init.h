#ifndef TSP_INIT_H
#define TSP_INIT_H

#include "TSP_Basic_Functions.h"

// Estimate the potential of each edge by upper bound confidence function
float Temp_Get_Potential(int First_City, int Second_City)
{
    // double Potential=Weight[First_City][Second_City]/Avg_Weight+Alpha*sqrt(
    // log(Total_Simulation_Times+1) / (
    // log(2.718)*(Chosen_Times[First_City][Second_City]+1) ) );

    return pow(2.718, 1 * Weight[First_City][Second_City]);
}

// Indentify the promising cities as candidates which are possible to connect
// to Cur_City
void Temp_Identify_Promising_City()
{
    Promising_City_Num = 0;
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        if (If_City_Selected[i] == true)
            continue;

        Promising_City[Promising_City_Num++] = i;
    }
}

// Set the probability (stored in Probabilistic[]) of selecting each candidate
// city (proportion to the potential of the corresponding edge)
bool Temp_Get_Probabilistic(int Cur_City)
{
    if (Promising_City_Num == 0)
        return false;

    float Total_Potential = 0;
    for (int i = 0; i < Promising_City_Num; i++)
        Total_Potential += Temp_Get_Potential(Cur_City, Promising_City[i]);

    Probabilistic[0] = (int)(1000 * Temp_Get_Potential(Cur_City, Promising_City[0]) / Total_Potential);
    for (int i = 1; i < Promising_City_Num - 1; i++)
        Probabilistic[i] =
            Probabilistic[i - 1] + (int)(1000 * Temp_Get_Potential(Cur_City, Promising_City[i]) / Total_Potential);
    Probabilistic[Promising_City_Num - 1] = 1000;

    return true;
}

// Probabilistically choose a city, controled by the values stored in
// Probabilistic[]
int Temp_Probabilistic_Get_City_To_Connect()
{
    int Random_Num = Get_Random_Int(1000);
    for (int i = 0; i < Promising_City_Num; i++)
        if (Random_Num < Probabilistic[i])
            return Promising_City[i];

    return Null;
}

// The whole process of choosing a city (a_{i+1} in the paper) to connect
// Cur_City (b_i in the paper)
int Temp_Choose_City_To_Connect(int Cur_City)
{
    // Avg_Weight=Get_Avg_Weight(Cur_City);
    Temp_Identify_Promising_City();
    Temp_Get_Probabilistic(Cur_City);

    return Temp_Probabilistic_Get_City_To_Connect();
}

bool Generate_Initial_Solution()
{
    if (MCTS_Debug)
        cout << "Generate_Initial_Solution() begin" << endl;
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        Solution[i] = Null;
        If_City_Selected[i] = false;
    }

    int Selected_City_Num = 0;
    int Cur_City = Start_City;
    int Next_City;

    Solution[Selected_City_Num++] = Cur_City;
    If_City_Selected[Cur_City] = true;
    do
    {
        // Next_City=Select_Random_City(Cur_City);
        Next_City = Temp_Choose_City_To_Connect(Cur_City);
        if (Next_City != Null)
        {
            // if (MCTS_Debug)
            //     cout << "Initial_Solution_Cur_City: " << Cur_City << " " << endl;
            Solution[Selected_City_Num++] = Next_City;
            If_City_Selected[Next_City] = true;
            Cur_City = Next_City;
        }
    } while (Next_City != Null);

    Convert_Solution_To_All_Node();
    if (MCTS_Debug)
        cout << "Generate_Initial_Solution() end" << endl;

    if (Check_Solution_Feasible() == false)
    {
        cout << "\nError! The constructed solution is unfeasible" << endl;
        getchar();
        return false;
    }

    return true;
}

// Generates an initial solution using the random insertion method
bool Generate_Initial_Solution_Random_Insert()
{
    if (MCTS_Debug)
        cout << "Generate_Initial_Solution_Random_Insert() begin" << endl;
    
    // Reset solution and city selection markers
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        Solution[i] = Null;
        If_City_Selected[i] = false;
    }
    
    // Start with Start_City
    Solution[0] = Start_City;
    If_City_Selected[Start_City] = true;
    
    // Find a second city (could be random, but using closest city for better results)
    int second_city = Null;
    Distance_Type min_dist = Inf_Cost;
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        if (i != Start_City && Distance[Start_City][i] < min_dist)
        {
            min_dist = Distance[Start_City][i];
            second_city = i;
        }
    }
    
    // Add second city to form initial subtour
    Solution[1] = second_city;
    If_City_Selected[second_city] = true;
    int current_tour_size = 2;
    
    // Random insertion for each remaining city
    for (int i = 2; i < Virtual_City_Num; i++)
    {
        // Select a random unvisited city
        Promising_City_Num = 0;
        for (int j = 0; j < Virtual_City_Num; j++)
        {
            if (!If_City_Selected[j])
                Promising_City[Promising_City_Num++] = j;
        }
        
        if (Promising_City_Num == 0)
            break;  // No more cities to insert
        
        int random_index = Get_Random_Int(Promising_City_Num);
        int next_city = Promising_City[random_index];
        
        // Find the best position to insert the new city
        int best_pos = -1;
        Distance_Type min_increase = Inf_Cost;
        
        for (int pos = 0; pos < current_tour_size; pos++)
        {
            int prev = Solution[pos];
            int next;
            if (pos == current_tour_size - 1)
                next = Solution[0]; // Close the loop for the last position
            else
                next = Solution[pos + 1];
            
            // Calculate increase in tour length if we insert next_city between prev and next
            Distance_Type increase = Distance[prev][next_city] + Distance[next_city][next] - Distance[prev][next];
            
            if (increase < min_increase)
            {
                min_increase = increase;
                best_pos = pos;
            }
        }
        
        // Insert the city at the best position
        if (best_pos == current_tour_size - 1)
        {
            // Insert at the end
            Solution[current_tour_size] = next_city;
        }
        else
        {
            // Shift cities to make room
            for (int j = current_tour_size; j > best_pos + 1; j--)
            {
                Solution[j] = Solution[j - 1];
            }
            Solution[best_pos + 1] = next_city;
        }
        
        If_City_Selected[next_city] = true;
        current_tour_size++;
    }
    
    // Convert the solution to the linked structure
    Convert_Solution_To_All_Node();
    
    if (MCTS_Debug)
        cout << "Generate_Initial_Solution_Random_Insert() end" << endl;
    
    // Verify solution
    if (Check_Solution_Feasible() == false)
    {
        cout << "\nError! The constructed solution is unfeasible" << endl;
        getchar();
        return false;
    }
    
    return true;
}

#endif // TSP_INIT_H
