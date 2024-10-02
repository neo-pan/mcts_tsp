#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "TSP_2Opt.h"
#include "TSP_Basic_Functions.h"
#include "TSP_IO.h"
#include "TSP_Init.h"
#include "TSP_MCTS.h"
#include "TSP_Markov_Decision.h"

namespace py = pybind11;

struct TSP_Result
{
    double Concorde_Distance;
    double MCTS_Distance;
    double Gap;
    double Time;
    py::list Solution;
    py::list Length_Time;
};

TSP_Result solve(int city_num, double alpha, double beta, double param_h, double param_t, int max_candidate_num,
                 int candidate_use_heatmap, int max_depth, py::array_t<double> distances, py::array_t<int> opt_solution,
                 py::array_t<double> heatmap, bool log_len_time, bool debug)
{
    double Overall_Begin_Time = (double)clock();
    srand(Random_Seed);

    // Initialize parameters
    Temp_City_Num = city_num;
    Alpha = alpha;
    Beta = beta;
    Param_H = param_h;
    Param_T = param_t;
    Max_Candidate_Num = max_candidate_num;
    Candidate_Use_Heatmap = candidate_use_heatmap;
    Max_Depth = max_depth;
    Log_Length_Time = log_len_time;

    if (debug)
    {
        std::cout << "求解器创建，参数如下：" << std::endl;
        std::cout << "Temp_City_Num: " << Temp_City_Num << std::endl;
        std::cout << "Alpha: " << Alpha << std::endl;
        std::cout << "Beta: " << Beta << std::endl;
        std::cout << "Param_H: " << Param_H << std::endl;
        std::cout << "Param_T: " << Param_T << std::endl;
        std::cout << "Max_Candidate_Num: " << Max_Candidate_Num << std::endl;
        std::cout << "Candidate_Use_Heatmap: " << Candidate_Use_Heatmap << std::endl;
        std::cout << "Max_Depth: " << Max_Depth << std::endl;
    }

    City_Num = Temp_City_Num;
    Start_City = 0;
    Salesman_Num = 1;
    Virtual_City_Num = City_Num + Salesman_Num - 1;

    Allocate_Memory(Virtual_City_Num);
    // Assert that distances has the correct shape
    auto distances_shape = distances.shape();
    auto solution_shape = opt_solution.shape();
    auto heatmap_shape = heatmap.shape();

    // Size check and dimension check for numpy arrays
    if (distances.ndim() != 2 || distances_shape[0] != distances_shape[1] || distances_shape[0] != Virtual_City_Num)
    {
        throw std::runtime_error("Invalid distances array shape or dimensions");
    }

    if (opt_solution.ndim() != 1 || solution_shape[0] != Virtual_City_Num)
    {
        throw std::runtime_error("Invalid solution array shape or dimensions");
    }

    if (heatmap.ndim() != 2 || heatmap_shape[0] != Virtual_City_Num || heatmap_shape[1] != Virtual_City_Num)
    {
        throw std::runtime_error("Invalid heatmap array shape or dimensions");
    }

    // Fill in the matrix
    auto distances_r = distances.unchecked<2>();
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        for (int j = 0; j < Virtual_City_Num; j++)
        {
            if (i != j)
            {
                Distance[i][j] = distances_r(i, j) * Magnify_Rate;
                DoubleDistance[i][j] = distances_r(i, j) * Magnify_Rate;
            }
            else
            {
                Distance[i][j] = Inf_Cost;
                DoubleDistance[i][j] = Inf_Cost;
            }
        }
    }

    auto solution_r = opt_solution.unchecked<1>();
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        Opt_Solution[i] = solution_r(i) - 1;
    }

    auto heatmap_r = heatmap.unchecked<2>();
    for (int i = 0; i < Virtual_City_Num; i++)
    {
        for (int j = 0; j < Virtual_City_Num; j++)
        {
            Edge_Heatmap[i][j] = heatmap_r(i, j);
        }
    }

    for (int i = 0; i < City_Num; i++)
    {
        for (int j = i + 1; j < City_Num; j++)
        {
            Edge_Heatmap[i][j] = (Edge_Heatmap[i][j] + Edge_Heatmap[j][i]) / 2;
            Edge_Heatmap[j][i] = Edge_Heatmap[i][j];
        }
    }

    Current_Instance_Begin_Time = (double)clock();
    Current_Instance_Best_Distance = Inf_Cost;

    Identify_Candidate_Set();
    Markov_Decision_Process();

    double Stored_Solution_Double_Distance = Get_Stored_Solution_Double_Distance();
    double Current_Solution_Double_Distance = Get_Current_Solution_Double_Distance();
    double Concorde_Distance = Stored_Solution_Double_Distance / Magnify_Rate;
    double MCTS_Distance = Current_Solution_Double_Distance / Magnify_Rate;
    double Gap = (Current_Solution_Double_Distance - Stored_Solution_Double_Distance) / Stored_Solution_Double_Distance;
    double Time = ((double)clock() - Current_Instance_Begin_Time) / CLOCKS_PER_SEC;
    double Overall_Time = ((double)clock() - Overall_Begin_Time) / CLOCKS_PER_SEC;

    vector<int> Solution;
    int Cur_City = Start_City;
    do
    {
        Solution.push_back(Cur_City);
        Cur_City = All_Node[Cur_City].Next_City;
    } while (Cur_City != Null && Cur_City != Start_City);

    for (auto &pair : Length_Time)
    {
        pair.first /= Magnify_Rate;
        pair.second /= CLOCKS_PER_SEC;
    }

    if (debug)
    {
        std::cout << "求解完成，结果如下：" << std::endl;
        std::cout << "Stored_Solution_Double_Distance: " << Stored_Solution_Double_Distance << std::endl;
        std::cout << "Current_Solution_Double_Distance: " << Current_Solution_Double_Distance << std::endl;
        std::cout << "Concorde_Distance: " << Concorde_Distance << std::endl;
        std::cout << "MCTS_Distance: " << MCTS_Distance << std::endl;
        std::cout << "Gap: " << Gap * 100 << "%" << std::endl;
        std::cout << "Time: " << Time << " seconds" << std::endl;
        std::cout << "Overall_Time: " << Overall_Time << " seconds" << std::endl;
        std::cout << "Solution: ";
        for (int i = 0; i < Solution.size(); ++i)
        {
            std::cout << Solution[i] << " ";
        }
        for (auto &pair : Length_Time)
        {
            std::cout << "Length: " << pair.first << " Time: " << pair.second << std::endl;
        }
        std::cout << std::endl;
    }

    Release_Memory(Virtual_City_Num);

    return TSP_Result{Concorde_Distance, MCTS_Distance, Gap, Time, py::cast(Solution), py::cast(Length_Time)};
}

PYBIND11_MODULE(_mcts_cpp, m)
{
    m.def("solve", &solve, "A function to solve TSP using MCTS", py::arg("city_num"), py::arg("alpha"), py::arg("beta"),
          py::arg("param_h"), py::arg("param_t"), py::arg("max_candidate_num"), py::arg("candidate_use_heatmap"),
          py::arg("max_depth"), py::arg("distances"), py::arg("opt_solution"), py::arg("heatmap"),
          py::arg("log_len_time") = false, py::arg("debug") = false);

    py::class_<TSP_Result>(m, "TSP_Result")
        .def(py::init<>())
        .def_readonly("Concorde_Distance", &TSP_Result::Concorde_Distance)
        .def_readonly("MCTS_Distance", &TSP_Result::MCTS_Distance)
        .def_readonly("Gap", &TSP_Result::Gap)
        .def_readonly("Time", &TSP_Result::Time)
        .def_readonly("Solution", &TSP_Result::Solution)
        .def_readonly("Length_Time", &TSP_Result::Length_Time)
        .def("__repr__",
             [](const TSP_Result &r) {
                 std::string solution_str = py::str(py::tuple(r.Solution)).cast<std::string>();
                 std::string length_time_str = py::str(py::tuple(r.Length_Time)).cast<std::string>();
                 return "TSP_Result(Concorde_Distance=" + std::to_string(r.Concorde_Distance) +
                        ", MCTS_Distance=" + std::to_string(r.MCTS_Distance) + ", Gap=" + std::to_string(r.Gap) +
                        ", Time=" + std::to_string(r.Time) + ", solution=" + solution_str +
                        ", length_time=" + length_time_str + ")";
             })
        .def(py::pickle(
            [](const TSP_Result &r) { // __getstate__
                return py::make_tuple(r.Concorde_Distance, r.MCTS_Distance, r.Gap, r.Time, r.Solution, r.Length_Time);
            },
            [](py::tuple t) { // __setstate__
                if (t.size() != 6)
                    throw std::runtime_error("Invalid state!");
                TSP_Result r;
                r.Concorde_Distance = t[0].cast<double>();
                r.MCTS_Distance = t[1].cast<double>();
                r.Gap = t[2].cast<double>();
                r.Time = t[3].cast<double>();
                r.Solution = t[4].cast<py::list>();
                r.Length_Time = t[5].cast<py::list>();
                return r;
            }));
}