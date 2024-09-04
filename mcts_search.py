import re
import time
import numpy as np
import os
import subprocess
import concurrent.futures
import tempfile
from dataclasses import dataclass

@dataclass
class Config:
    current_file_path: str = os.path.abspath(__file__)
    current_dir_path: str = os.path.dirname(current_file_path)
    total_instance_num: int = -1
    thread_num: int = 1
    inst_num_per_batch: int = 1
    mcts_dir: str = ''

def read_default_heatmap(file_name):
    with open(file_name, "r") as file:
        lines = file.readlines()
    N = int(lines[0].strip())
    assert len(lines) == N + 1
    
    heatmap = np.zeros((N, N))
    for i, line in enumerate(lines[1:]):
        heatmap[i] = np.array([float(x) for x in line.split()])

    return heatmap

def get_heatmap_list(config, name, num_nodes, num_instances):
    folder_name = os.path.join(config.current_dir_path, "all_heatmap", name, "heatmap", f"tsp{num_nodes}")
    
    def read_single_heatmap(i):
        return read_default_heatmap(os.path.join(folder_name, f"heatmaptsp{num_nodes}_{i}.txt"))
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=config.thread_num) as executor:
        heatmap_list = list(executor.map(read_single_heatmap, range(num_instances)))
    
    return np.array(heatmap_list)


def read_concorde_file(config, file_name):
    # Read the entire file content at once
    with open(file_name, "r") as file:
        content = file.read()

    # Split the content into lines
    lines = content.strip().split('\n')
    assert len(lines) >= config.total_instance_num

    # Split each line into coordinates and solution parts
    parts = [line.split("output") for line in lines]

    # Process coordinates
    coords = np.fromstring(' '.join([p[0].strip() for p in parts]), sep=' ')
    pos = coords.reshape(config.total_instance_num, -1, 2)

    # Process optimal solutions
    opt_sols = np.fromstring(' '.join([p[1].strip() for p in parts]), sep=' ', dtype=int)
    opt_sols = opt_sols.reshape(config.total_instance_num, -1)

    return pos, opt_sols

def read_utsp_heatmap(heatmap_file = "1kTraning_TSP500Instance_128.txt", num_of_nodes = 500, topk = 500):
    with open(heatmap_file, "r") as file:
        lines = file.readlines()
    assert len(lines) == 128
    Saved_indices = np.zeros((128, num_of_nodes, topk))
    Saved_Values = np.zeros((128, num_of_nodes, topk))
    Saved_sol = np.zeros((128, num_of_nodes + 1))
    Saved_pos = np.zeros((128, num_of_nodes, 2))
    for i, line in enumerate(lines):
        coords, rest = line.split("output")
        coords = coords.strip().split(" ")
        opt_sol, rest = rest.split("indices")
        opt_sol = opt_sol.strip().split(" ")
        indices, heatmap = rest.split("value")
        indices = indices.strip().split(" ")
        heatmap = heatmap.strip().split(" ")
        coords = np.array([float(x) for x in coords])
        coords = coords.reshape((num_of_nodes, 2))
        Saved_pos[i] = coords
        opt_sol = np.array([int(x) for x in opt_sol])
        Saved_sol[i] = opt_sol
        indices = np.array([int(x) for x in indices])
        indices = indices.reshape((num_of_nodes, topk))
        Saved_indices[i] = indices
        heatmap = np.array([float(x) for x in heatmap])
        heatmap = heatmap.reshape((num_of_nodes, topk))
        Saved_Values[i] = heatmap

    return Saved_indices, Saved_Values, Saved_sol, Saved_pos

def default_write_heatmap_to_file(
    config, heatmap_list, opt_sols, distance_matrix, heatmap_filename_list, concorde_filename_list
):
    assert config.total_instance_num == len(heatmap_filename_list) == heatmap_list.shape[0], f"total_instance_num: {config.total_instance_num}, heatmap_filename_list: {len(heatmap_filename_list)}, heatmap_list: {heatmap_list.shape[0]}"
    num_nodes = distance_matrix.shape[1]

    def write_concorde_file(args):
        i, concorde_filename = args
        with open(concorde_filename, "w") as f:
            for j in range(distance_matrix.shape[1]):
                for k in range(distance_matrix.shape[2]):
                    f.write(f"{distance_matrix[i][j][k]} ")
            f.write("output ")
            for j in range(opt_sols.shape[1]):
                f.write(f"{int(opt_sols[i][j])} ")
            f.write("\n")

    def write_heatmap_file(args):
        heatmap, heatmap_filename = args
        with open(heatmap_filename, "w") as f:
            f.write(f"{num_nodes}\n")
            for i in range(heatmap.shape[0]):
                f.write(" ".join(map(str, heatmap[i])) + "\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=config.thread_num) as executor:
        executor.map(write_concorde_file, enumerate(concorde_filename_list))
        executor.map(write_heatmap_file, zip(heatmap_list, heatmap_filename_list))

def utsp_write_heatmap_to_file(
    Saved_indices, Saved_Values, Saved_sol, Saved_pos, heatmap_filename, topk
):
    Q = Saved_pos
    A = Saved_sol
    C = Saved_indices
    V = Saved_Values
    with open(heatmap_filename, "w") as f:
        for i in range(Q.shape[0]):
            for j in range(Q.shape[1]):
                f.write(str(Q[i][j][0]) + " " + str(Q[i][j][1]) + " ")
            f.write("output ")
            for j in range(A.shape[1]):
                f.write(str(int(A[i][j] + 1)) + " ")
            f.write("indices ")
            for j in range(C.shape[1]):
                for k in range(topk):
                    if C[i][j][k] == j:
                        f.write("-1" + " ")
                    else:
                        f.write(str(int(C[i][j][k] + 1)) + " ")
            f.write("value ")
            for j in range(V.shape[1]):
                for k in range(topk):
                    f.write(str(V[i][j][k]) + " ")
            f.write("\n")
            if i == Saved_indices.shape[0] - 1:
                break

def make_binary(config):
    cwd = os.getcwd()
    os.chdir(config.mcts_dir)
    print("MCTS directory:", os.getcwd())
    if os.path.exists("./code/TSP.o"):
        os.remove("./code/TSP.o")
    if os.path.exists("./test"):
        os.remove("./test")
    subprocess.run(["make"])
    os.chdir(cwd)

def run_single_utsp_mcts(i, result_file, heatmap_file, num_of_nodes, mcts_dir):
    subprocess.run(
        [
            os.path.join(mcts_dir, "test"),
            str(i),
            result_file,
            heatmap_file,
            str(num_of_nodes),
            "1",
            "0",
            "5",
            "100",
            "0",
            "50",
            "2",
            "1",
            "1",
        ]
    )

def run_single_default_mcts(
    result_file,
    heatmap_file,
    concorde_file,
    num_of_nodes,
    mcts_dir,
    alpha,
    beta,
    H,
    T,
    max_candidate_num,
    use_heatmap,
    max_depth,  # New parameter
):
    subprocess.run(
        [
            os.path.join(mcts_dir, "test"),
            "0",
            result_file,
            concorde_file,
            str(num_of_nodes),
            "1",
            heatmap_file,
            str(alpha),
            str(beta),
            str(H),
            str(T),
            str(max_candidate_num),
            str(use_heatmap),
            str(max_depth),  # Add max_depth to subprocess call
        ]
    )

def run_mcts(
    config,
    heatmap_file_list,
    result_file_list,
    concorde_file_list,
    num_of_nodes,
    alpha,
    beta,
    H,
    T,
    max_candidate_num,
    use_heatmap,
    max_depth,  # New parameter
):
    assert len(result_file_list) == config.total_instance_num

    with concurrent.futures.ProcessPoolExecutor(max_workers=config.thread_num) as executor:
        futures = [
            executor.submit(
                run_single_default_mcts,
                result_file_list[i],
                heatmap_file_list[i],
                concorde_file_list[i],
                num_of_nodes,
                config.mcts_dir,
                alpha,
                beta,
                H,
                T,
                max_candidate_num,
                use_heatmap,
                max_depth,  # Pass max_depth to run_single_default_mcts
            )
            for i in range(config.total_instance_num)
        ]

        for future in concurrent.futures.as_completed(futures):
            future.result()

def analyze(result_file_list, config):
    total_concorde = []
    total_mcts = []
    total_time = []
    total_gap = []
    inst_index_count = 0

    pattern = re.compile(
        r"Avg_Concorde_Distance:\s*(-?\d+\.\d+)\s+"
        r"Avg_MCTS_Distance:\s*(-?\d+\.\d+)\s+"
        r"Avg_Gap:\s*(-?\d+\.\d+)\s+"
        r"Total_Time:\s*(-?\d+\.\d+)"
    )
    for result_file in result_file_list:
        with open(result_file, "r") as file:
            content = file.read()
            matches = pattern.finditer(content)
            for match in matches:
                inst_index_count += 1
                total_concorde.append(float(match.group(1)))
                total_mcts.append(float(match.group(2)))
                total_gap.append(float(match.group(3)) * 100)
                total_time.append(float(match.group(4)))
    assert inst_index_count == config.total_instance_num, f"inst_index_count: {inst_index_count}, total_instance_num: {config.total_instance_num}"

    return total_concorde, total_mcts, total_time, total_gap

def mcts(
    config,
    heatmap_list,
    opt_sols,
    distance_matrix,
    alpha=1,
    beta=10,
    H=10,
    T=0.1,
    max_candidate_num=1000,
    use_heatmap=1,
    use_temp_dir=True,
    max_depth=10,  # New parameter with default value
):
    assert heatmap_list.shape[0] == opt_sols.shape[0] == distance_matrix.shape[0]
    assert config.total_instance_num == heatmap_list.shape[0]
    num_nodes = distance_matrix.shape[1]

    def process_in_directory(directory):
        heatmap_file_list = [
            os.path.join(directory, f"heatmap_{i}.txt")
            for i in range(config.total_instance_num)
        ]
        result_file_list = [
            os.path.join(directory, f"result_{i}.txt") for i in range(config.total_instance_num)
        ]
        concorde_file_list = [
            os.path.join(directory, f"concorde_{i}.txt")
            for i in range(config.total_instance_num)
        ]

        default_write_heatmap_to_file(
            config, heatmap_list, opt_sols, distance_matrix, heatmap_file_list, concorde_file_list
        )
        run_mcts(
            config,
            heatmap_file_list,
            result_file_list,
            concorde_file_list,
            num_nodes,
            alpha,
            beta,
            H,
            T,
            max_candidate_num,
            use_heatmap,
            max_depth,  # Pass max_depth to run_mcts
        )
        return analyze(result_file_list, config)

    if use_temp_dir:
        with tempfile.TemporaryDirectory(dir="/dev/shm") as temp_dir:
        # with tempfile.TemporaryDirectory(dir="/mnt/data3/xhpan/tmp") as temp_dir:
            total_concorde, total_mcts, total_time, total_gap = process_in_directory(temp_dir)
    else:
        debug_dir = os.path.join(config.current_dir_path, "tmp")
        os.makedirs(debug_dir, exist_ok=True)
        total_concorde, total_mcts, total_time, total_gap = process_in_directory(debug_dir)

    return total_concorde, total_mcts, total_time, total_gap

if __name__ == "__main__":
    config = Config()
    config.thread_num = 1
    config.total_instance_num = 1
    config.mcts_dir = "ours_mcts"
    # make_binary(config)

    num_nodes = 1000
    pos, opt_sols = read_concorde_file(config, f"ours_mcts/tsp{num_nodes}_test_concorde.txt")
    pos = pos[0:1]
    opt_sols = opt_sols[0:1]
    distance_matrix = np.sqrt(np.sum((pos[:, np.newaxis, :] - pos[:, :, np.newaxis])**2, axis=-1))
    print("Position shape:", pos.shape, "Optimal solutions shape:", opt_sols.shape, "Distance matrix shape:", distance_matrix.shape)
    heatmap_list = get_heatmap_list(config, "difusco", num_nodes, config.total_instance_num)

    # opt_sols = np.arange(num_nodes) + 1
    # opt_sols = np.tile(opt_sols, (config.total_instance_num, 1))
    # heatmap_list = np.random.rand(config.total_instance_num, num_nodes, num_nodes)
    # pos = np.random.rand(config.total_instance_num, num_nodes, 2)
    # distance_matrix = np.sqrt(np.sum((pos[:, np.newaxis, :] - pos[:, :, np.newaxis])**2, axis=-1))
    # # distance_matrix = np.random.rand(config.total_instance_num, num_nodes, num_nodes)
    # print(f"heatmap_list shape: {heatmap_list.shape}, opt_sols shape: {opt_sols.shape}, distance_matrix shape: {distance_matrix.shape}")
    # print(distance_matrix[0])

    # start_time = time.time()
    # gaps = []
    # gaps.append(
    #     mcts(
    #         config,
    #         heatmap_list[0:1],
    #         opt_sols[0:1],
    #         distance_matrix[0:1],
    #         # alpha=1,
    #         # beta=50,
    #         # H=2,
    #         T=100./num_nodes,
    #         # max_candidate_num=5,
    #         # use_heatmap=1,
    #         use_temp_dir=False,  # Set to False for debugging
    #         # max_depth=100,  # Add max_depth parameter
    #     )[-1]
    # )
    # print(f"Gap: \t{np.average(gaps[0])}")
    # print(f"Time: \t{time.time() - start_time}")

    import mcts
    start_time = time.time()
    mcts.solve(10000, 1, 10, 10, 100./num_nodes, 1000, 1, 10, distance_matrix[0], opt_sols[0], heatmap_list[0], debug=True)
    print(f"Time: \t{time.time() - start_time}")

