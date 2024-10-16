import concurrent.futures
import numpy as np
from multiprocessing import shared_memory
from .mcts_wrapper import solve_one_instance

# Helper function for creating shared memory for one instance
def create_shared_memory_for_one_instance(data):
    shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
    shm_arr = np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)
    shm_arr[:] = data[:]  # Copy data into shared memory
    return shm, shm_arr

# Helper function for accessing shared memory in child processes
def access_shared_memory(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm

# Function that processes shared memory access and performs calculations
def solve_one_instance_with_shared_memory(shm_coords_name, shm_solutions_name, shm_heatmaps_name, 
                                          coords_shape, sol_shape, heatmap_shape,
                                          coords_dtype, sol_dtype, heatmap_dtype, 
                                          city_num, alpha, beta, param_h, param_t,
                                          max_candidate_num, candidate_use_heatmap, 
                                          max_depth, log_len_time, debug):
    # Access shared memory in child process
    shm_coords_arr, shm_coords = access_shared_memory(shm_coords_name, coords_shape, coords_dtype)
    shm_solutions_arr, shm_solutions = access_shared_memory(shm_solutions_name, sol_shape, sol_dtype)
    shm_heatmaps_arr, shm_heatmaps = access_shared_memory(shm_heatmaps_name, heatmap_shape, heatmap_dtype)

    try:
        # Call the original solve function with the shared memory arrays
        result = solve_one_instance(shm_coords_arr, shm_solutions_arr, shm_heatmaps_arr, city_num, alpha, beta, param_h, param_t,
                                    max_candidate_num, candidate_use_heatmap, max_depth, log_len_time, debug)
    finally:
        # Cleanup shared memory references in child process
        shm_coords.close()
        shm_coords.unlink()
        shm_solutions.close()
        shm_solutions.unlink()
        shm_heatmaps.close()
        shm_heatmaps.unlink()

    return result

def parallel_mcts_solve(city_num, coordinates_list, opt_solutions, heatmaps, num_threads, alpha=1, beta=10, param_h=10, param_t=0.1,
                        max_candidate_num=5, candidate_use_heatmap=1, max_depth=10, log_len_time=False, debug=False):

    results = []
    futures = []

    # Step 1: Use ProcessPoolExecutor to submit tasks as soon as data is ready
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:

        for i in range(len(coordinates_list)):
            # Step 2: Create shared memory for this instance
            shm_coords, shm_coords_arr = create_shared_memory_for_one_instance(coordinates_list[i])
            shm_solutions, shm_solutions_arr = create_shared_memory_for_one_instance(opt_solutions[i])
            shm_heatmaps, shm_heatmaps_arr = create_shared_memory_for_one_instance(heatmaps[i])

            # Step 3: Submit the task immediately after copying the data to shared memory
            future = executor.submit(
                solve_one_instance_with_shared_memory,
                shm_coords.name, shm_solutions.name, shm_heatmaps.name,
                coordinates_list[i].shape, opt_solutions[i].shape, heatmaps[i].shape,
                coordinates_list[i].dtype, opt_solutions[i].dtype, heatmaps[i].dtype,
                city_num, alpha, beta, param_h, param_t, 
                max_candidate_num, candidate_use_heatmap, max_depth, log_len_time, debug
            )
            futures.append(future)

            # Step 4: Continue to copy the next instance while the previous one is being processed
            # Clean up the shared memory in the main process for the current instance
            shm_coords.close()
            shm_solutions.close()
            shm_heatmaps.close()

        # Collect the results as they are completed
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    # Step 5: Gather the results
    concorde_distances = [result.Concorde_Distance for result in results if result is not None]
    mcts_distances = [result.MCTS_Distance for result in results if result is not None]
    gaps = [result.Gap for result in results if result is not None]
    times = [result.Time for result in results if result is not None]
    overall_times = [result.Overall_Time for result in results if result is not None]
    solutions = [result.Solution for result in results if result is not None]
    lengths_times = [result.Length_Time for result in results if result is not None]

    return concorde_distances, mcts_distances, gaps, times, overall_times, solutions, lengths_times
