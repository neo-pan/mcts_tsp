import concurrent.futures
import numpy as np
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from .mcts_wrapper import solve_one_instance

# Helper function for creating shared memory for one instance using SharedMemoryManager
def create_shared_memory_for_one_instance(smm, data):
    shm = smm.SharedMemory(size=data.nbytes)
    # Create temporary array view to copy data, no need to return it
    np.ndarray(data.shape, dtype=data.dtype, buffer=shm.buf)[:] = data[:]
    return shm

# Helper function for accessing shared memory in child processes
def access_shared_memory(shm_name, shape, dtype):
    shm = shared_memory.SharedMemory(name=shm_name)
    return np.ndarray(shape, dtype=dtype, buffer=shm.buf), shm

# Function that processes shared memory access and performs calculations
def solve_one_instance_with_shared_memory(shm_coords_name, shm_solutions_name, shm_heatmaps_name, shm_knn_edges_name,
                                          coords_shape, sol_shape, heatmap_shape, knn_edges_shape,
                                          coords_dtype, sol_dtype, heatmap_dtype, knn_edges_dtype,
                                          city_num, alpha, beta, param_h, param_t,
                                          max_candidate_num, candidate_use_heatmap, 
                                          max_depth, log_len_time, debug):
    # Access shared memory in child process
    shm_coords_arr, shm_coords = access_shared_memory(shm_coords_name, coords_shape, coords_dtype)
    shm_solutions_arr, shm_solutions = access_shared_memory(shm_solutions_name, sol_shape, sol_dtype)
    shm_heatmaps_arr, shm_heatmaps = access_shared_memory(shm_heatmaps_name, heatmap_shape, heatmap_dtype)
    shm_knn_edges_arr, shm_knn_edges = access_shared_memory(shm_knn_edges_name, knn_edges_shape, knn_edges_dtype)

    try:
        # Call the original solve function with the shared memory arrays
        result = solve_one_instance(
            city_num, alpha, beta, param_h, param_t,
            max_candidate_num, candidate_use_heatmap, max_depth,
            shm_coords_arr, shm_solutions_arr, shm_heatmaps_arr, shm_knn_edges_arr,
            log_len_time, debug
        )
    finally:
        # Cleanup shared memory references in child process
        shm_coords.close()
        shm_solutions.close()
        shm_heatmaps.close()
        shm_knn_edges.close()

    return result

def parallel_mcts_solve(city_num, num_threads, coordinates_list, opt_solutions, heatmaps, knn_edges_list, alpha=1, beta=10, param_h=10, param_t=0.1,
                        max_candidate_num=5, candidate_use_heatmap=1, max_depth=10, batch_size=None, log_len_time=False, debug=False):

    results = []
    total_instances = len(coordinates_list)
    
    # Determine batch size if not specified (default to processing all instances at once if total is small)
    if batch_size is None:
        batch_size = min(total_instances, max(1, num_threads * 2))
    
    # Process instances in batches to limit memory consumption
    for batch_start in range(0, total_instances, batch_size):
        batch_end = min(batch_start + batch_size, total_instances)
        batch_results = []
        
        # Use SharedMemoryManager to automatically handle cleanup
        with SharedMemoryManager() as smm:
            futures = []  # Track futures for later collection
            
            # Step 1: Use ProcessPoolExecutor for parallel processing
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
                
                for i in range(batch_start, batch_end):
                    # Step 2: Create shared memory for this instance - only keep the shared memory objects
                    shm_coords = create_shared_memory_for_one_instance(smm, coordinates_list[i])
                    shm_solutions = create_shared_memory_for_one_instance(smm, opt_solutions[i])
                    shm_heatmaps = create_shared_memory_for_one_instance(smm, heatmaps[i])
                    shm_knn_edges = create_shared_memory_for_one_instance(smm, knn_edges_list[i])
                    
                    # Step 3: Submit the task
                    future = executor.submit(
                        solve_one_instance_with_shared_memory,
                        shm_coords.name, shm_solutions.name, shm_heatmaps.name, shm_knn_edges.name,
                        coordinates_list[i].shape, opt_solutions[i].shape, heatmaps[i].shape, knn_edges_list[i].shape,
                        coordinates_list[i].dtype, opt_solutions[i].dtype, heatmaps[i].dtype, knn_edges_list[i].dtype,
                        city_num, alpha, beta, param_h, param_t, 
                        max_candidate_num, candidate_use_heatmap, max_depth, log_len_time, debug
                    )
                    futures.append(future)
                    
                # Collect the results as they are completed
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    batch_results.append(result)
                    
            # The SharedMemoryManager context will automatically clean up all shared memory
        
        # Add batch results to overall results
        results.extend(batch_results)

    # Gather the results
    concorde_distances = [result.Concorde_Distance for result in results if result is not None]
    mcts_distances = [result.MCTS_Distance for result in results if result is not None]
    gaps = [result.Gap for result in results if result is not None]
    times = [result.Time for result in results if result is not None]
    overall_times = [result.Overall_Time for result in results if result is not None]
    solutions = [result.Solution for result in results if result is not None]
    lengths_times = [result.Length_Time for result in results if result is not None]

    return concorde_distances, mcts_distances, gaps, times, overall_times, solutions, lengths_times
