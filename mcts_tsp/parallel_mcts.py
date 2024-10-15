import concurrent.futures
import numpy as np
from multiprocessing import shared_memory
from .mcts_wrapper import solve_one_instance

# Helper function to create shared memory for ndarray
def create_shared_memory(arr):
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]  # Copy data into shared memory
    return shm, shm_arr

# Helper function for accessing a slice of shared memory in child processes
def access_shared_memory_slice(shm_name, shape, dtype, index):
    shm = shared_memory.SharedMemory(name=shm_name)
    full_arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return full_arr[index], shm  # Return the slice corresponding to the index

def parallel_mcts_solve(city_num, coordinates_list, opt_solutions, heatmaps, num_threads, alpha=1, beta=10, param_h=10, param_t=0.1,
                        max_candidate_num=5, candidate_use_heatmap=1, max_depth=10, log_len_time=False, debug=False):

    # Step 1: Create shared memory for large data arrays
    shm_coords, shm_coords_arr = create_shared_memory(np.array(coordinates_list))
    shm_solutions, shm_solutions_arr = create_shared_memory(np.array(opt_solutions))
    shm_heatmaps, shm_heatmaps_arr = create_shared_memory(np.array(heatmaps))

    try:
        # Step 2: Prepare arguments for each process, passing shared memory names
        args = [
            (shm_coords.name, shm_solutions.name, shm_heatmaps.name, coordinates_list.shape, opt_solutions.shape, heatmaps.shape,
             coordinates_list.dtype, opt_solutions.dtype, heatmaps.dtype, i, city_num, alpha, beta, param_h, param_t, 
             max_candidate_num, candidate_use_heatmap, max_depth, log_len_time, debug)
            for i in range(len(coordinates_list))
        ]

        results = []

        # Step 3: Use ProcessPoolExecutor, passing shared memory names and child-specific index
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(solve_one_instance_with_shared_memory, *arg) for arg in args]

        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

        # Step 4: Gather the results
        concorde_distances = [result.Concorde_Distance for result in results if result is not None]
        mcts_distances = [result.MCTS_Distance for result in results if result is not None]
        gaps = [result.Gap for result in results if result is not None]
        times = [result.Time for result in results if result is not None]
        overall_times = [result.Overall_Time for result in results if result is not None]
        solutions = [result.Solution for result in results if result is not None]
        lengths_times = [result.Length_Time for result in results if result is not None]

    finally:
        # Step 5: Cleanup shared memory
        shm_coords.close()
        shm_coords.unlink()
        shm_solutions.close()
        shm_solutions.unlink()
        shm_heatmaps.close()
        shm_heatmaps.unlink()

    return concorde_distances, mcts_distances, gaps, times, overall_times, solutions, lengths_times

# Wrapper function that processes shared memory access in each process, based on index
def solve_one_instance_with_shared_memory(shm_coords_name, shm_solutions_name, shm_heatmaps_name, coords_shape, sol_shape, heatmap_shape,
                                          coords_dtype, sol_dtype, heatmap_dtype, index, city_num, alpha, beta, param_h, param_t,
                                          max_candidate_num, candidate_use_heatmap, max_depth, log_len_time, debug):
    # Step 6: Access the slice of shared memory in the child process for the corresponding index
    shm_coords_arr, shm_coords = access_shared_memory_slice(shm_coords_name, coords_shape, coords_dtype, index)
    shm_solutions_arr, shm_solutions = access_shared_memory_slice(shm_solutions_name, sol_shape, sol_dtype, index)
    shm_heatmaps_arr, shm_heatmaps = access_shared_memory_slice(shm_heatmaps_name, heatmap_shape, heatmap_dtype, index)

    # Call the original solving function with the sliced shared memory arrays
    result = solve_one_instance(shm_coords_arr, shm_solutions_arr, shm_heatmaps_arr, city_num, alpha, beta, param_h, param_t,
                                max_candidate_num, candidate_use_heatmap, max_depth, log_len_time, debug)

    # Cleanup the shared memory in the child process
    shm_coords.close()
    shm_solutions.close()
    shm_heatmaps.close()

    return result
