#ifndef TSP_IO_H
#define TSP_IO_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

int Total_Instance_Num = 1;

#define Null -1
#define Inf_Cost 1000000000
#define Magnify_Rate 100000
#define Coord_Dim 2
// Hyper parameters
thread_local double Alpha = 1;              // used in estimating the potential of each edge
thread_local double Beta = 10;              // used in back propagation
thread_local double Param_H = 10;           // used to control the number of sampling actions
thread_local double Param_T = 0.10;         // used to control the termination condition
thread_local int Max_Candidate_Num = 5;     // used to control the number of candidate neighbors of each city
thread_local int Candidate_Use_Heatmap = 1; // used to control whether to use the heatmap information
thread_local int Max_Depth = 10;            // used to control the depth of the search tree
thread_local bool Log_Length_Time = false;  // used to control whether to log the length-time information

thread_local bool MCTS_Debug = false;

#define Default_Random_Seed 489663920;
unsigned Random_Seed = Default_Random_Seed;

typedef int Distance_Type;

/* 2020-02-11 */
thread_local int Temp_City_Num;

// Used to store the input information of a given instance
thread_local int City_Num;
thread_local int Start_City;
thread_local int Salesman_Num; // This program was proposed for the multiple TSP. If
                               // Salesman_Num=1, it reduces to the TSP
thread_local int Virtual_City_Num;
thread_local double *Coordinate_X;
thread_local double *Coordinate_Y;
thread_local Distance_Type **Distance;
thread_local int *Opt_Solution;

// Store the length-time information
thread_local vector<std::pair<double, double>> Length_Time;

thread_local std::chrono::steady_clock::time_point Current_Instance_Begin_Time;
thread_local Distance_Type Current_Instance_Best_Distance;

// Used to store a solution in double link
struct Struct_Node
{
    int Pre_City;
    int Next_City;
    int Salesman;
};

thread_local Struct_Node *All_Node;      // Store the incumbent tour
thread_local Struct_Node *Best_All_Node; // Store the best found tour

// Used to store a solution in an array
thread_local int *Solution;

// Used to store a set of candidate neighbors of each city
thread_local int *Candidate_Num;
thread_local int **Candidate;
thread_local bool *If_City_Selected;

// Used to store the information of an action
thread_local int Pair_City_Num;
thread_local int Temp_Pair_Num;
thread_local int *City_Sequence;
thread_local int *Temp_City_Sequence;
thread_local Distance_Type *Gain;
thread_local Distance_Type *Real_Gain;

// Used in MCTS
thread_local float **Edge_Heatmap;
thread_local float **Weight;
thread_local float Avg_Weight;
thread_local int **Chosen_Times;
thread_local int *Promising_City;
thread_local int *Probabilistic;
thread_local int Promising_City_Num;
thread_local int Total_Simulation_Times;

Distance_Type Get_Solution_Total_Distance();
void Convert_Solution_To_All_Node();

void Allocate_Memory(int City_Num)
{
    Coordinate_X = new double[City_Num];
    Coordinate_Y = new double[City_Num];

    Distance = new Distance_Type *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Distance[i] = new Distance_Type[City_Num];

    Opt_Solution = new int[City_Num];

    All_Node = new Struct_Node[City_Num];
    Best_All_Node = new Struct_Node[City_Num];
    Solution = new int[City_Num];

    Candidate_Num = new int[City_Num];
    Candidate = new int *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Candidate[i] = new int[Max_Candidate_Num];
    If_City_Selected = new bool[City_Num];

    City_Sequence = new int[2 * City_Num];
    Temp_City_Sequence = new int[2 * City_Num];
    Gain = new Distance_Type[2 * City_Num];
    Real_Gain = new Distance_Type[2 * City_Num];

    Edge_Heatmap = new float *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Edge_Heatmap[i] = new float[City_Num];

    Weight = new float *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Weight[i] = new float[City_Num];

    Chosen_Times = new int *[City_Num];
    for (int i = 0; i < City_Num; i++)
        Chosen_Times[i] = new int[City_Num];

    Promising_City = new int[City_Num];
    Probabilistic = new int[City_Num];
}

void Release_Memory(int City_Num)
{
    for (int i = 0; i < City_Num; i++)
        delete[] Distance[i];
    delete[] Distance;

    delete[] Opt_Solution;

    delete[] All_Node;
    delete[] Best_All_Node;
    delete[] Solution;

    delete[] Candidate_Num;
    for (int i = 0; i < City_Num; i++)
        delete[] Candidate[i];
    delete[] Candidate;
    delete[] If_City_Selected;

    delete[] City_Sequence;
    delete[] Temp_City_Sequence;
    delete[] Gain;
    delete[] Real_Gain;

    for (int i = 0; i < City_Num; i++)
        delete[] Edge_Heatmap[i];
    delete[] Edge_Heatmap;

    for (int i = 0; i < City_Num; i++)
        delete[] Weight[i];
    delete[] Weight;

    for (int i = 0; i < City_Num; i++)
        delete[] Chosen_Times[i];
    delete[] Chosen_Times;

    delete[] Promising_City;
    delete[] Probabilistic;
}

// Print the cities of a solution one by one
void Print_TSP_Tour(int Begin_City)
{
    cout << "\nThe current tour is:" << endl;
    int Cur_City = Begin_City;
    do
    {
        printf("%d ", Cur_City + 1);
        Cur_City = All_Node[Cur_City].Next_City;
    } while (Cur_City != Null && Cur_City != Begin_City);
}

double Get_Elapsed_Time(std::chrono::steady_clock::time_point Begin_Time)
{
    auto Current_Time = std::chrono::steady_clock::now();

    std::chrono::duration<double> elapsed_seconds = Current_Time - Begin_Time;

    return elapsed_seconds.count();
}

#endif // TSP_IO_H