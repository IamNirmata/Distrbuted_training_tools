import torch.distributed as dist
import os
import time
import torch

# Function to generate pairwise configurations for a given number of participants
def generate_pairwise_config(n):
    def rotate(lst):
        return lst[1:] + lst[0:1]

    if n <= 0:
        raise ValueError("n must be a positive integer")

    config = []
    participants = list(range(n))

    # Add a dummy participant if the number of participants is odd
    if n % 2 == 1:
        participants.append(-1)

    fixed_participant = participants[0]
    rotating_participants = participants[1:]

    # Generate pairwise configurations by rotating participants
    for _ in range(len(participants) - 1):
        pairs = [
            [participants[i], participants[-i - 1]]
            for i in range(len(participants) // 2)
            if participants[i] != -1 and participants[-i - 1] != -1
        ]
        config.append(pairs)
        rotating_participants = rotate(rotating_participants)
        participants = [fixed_participant] + rotating_participants

    return config

# Function to map ranks to nodes based on the given ranks lists and ranks per node
def map_ranks_to_nodes(ranks_lists, ranks_per_node=8):

    if len(ranks_lists) == 0 or len(ranks_lists[0]) == 0:
        return []

    tests = []

    for i in range(len(ranks_lists)):
        test = []
        for j in range(len(ranks_lists[i])):
            # Calculate the ranks for the left and right nodes
            left = [(ranks_lists[i][j][0] * ranks_per_node) + k for k in range(ranks_per_node)]
            right = [(ranks_lists[i][j][1] * ranks_per_node) + k for k in range(ranks_per_node)]
            test.append(left + right)
        tests.append(test)

    return tests

# Retrieve environment variables for distributed training configuration
local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))                            
local_rank = int(os.environ.get("LOCAL_RANK", 0))                                        
world_size = int(os.environ.get("WORLD_SIZE", 1))                                        
world_rank = int(os.environ.get("RANK", 0))                                              
node_rank = int(os.environ.get("NODE_RANK", 0))                                                
node_world_size = world_size // local_world_size                                              

# Create local groups for each node
local_groups = [ [ i * local_world_size + j  for j in range(local_world_size)] for i in range(node_world_size)]                                                                        

# Print the configuration for debugging purposes
print(f"local_world_size: {local_world_size}, local_rank: {local_rank}, world_size: {world_size}, world_rank: {world_rank}, node_rank: {node_rank}, node_world: {node_world_size}")

# Initialize the process group for distributed training
dist.init_process_group("nccl", init_method="env://", rank=world_rank, world_size=world_size)
world_group = dist.group.WORLD 

# Create a new group for the current node
local_group = dist.new_group(ranks=local_groups[node_rank], use_local_synchronization=True)  

# Set the current CUDA device to the local rank
torch.cuda.set_device(local_rank)

# Initialize the all-pair table with default values
allpair_table = [[-1 for j in range(node_world_size)] for i in range(node_world_size) ]

# Generate pairwise configurations for nodes
allpairs_node = generate_pairwise_config(node_world_size)
allpairs = map_ranks_to_nodes(allpairs_node, ranks_per_node=local_world_size)

# Retrieve the number of pairs to test from the environment variable
n_pairs = os.getenv('NPAIRS')
if n_pairs is None:
    n_pairs = len(allpairs)
n_pairs = allpairs[:n_pairs]

g = 1024*1024*1024  # Define 1 GB in bytes
size = 8*g*2  # Total size of data to be reduced

# Allocate memory for the data to be reduced
npair_data = torch.zeros(8*g, dtype=torch.bfloat16).to('cuda')

# Create tensors for gathering results
tensor_list = [torch.zeros(len(n_pairs), dtype=torch.float32).to('cuda') for i in range(world_size)]
tensor = torch.zeros(len(n_pairs), dtype=torch.float32).to('cuda')

# Perform tests for each pair configuration
for test in range(len(n_pairs)):
    my_group = None
    my_group_id = -1
    for group in range(len(n_pairs[test])):
        # Create a new group for the current pair
        new_group = dist.new_group(ranks=n_pairs[test][group])
        if world_rank in n_pairs[test][group]:
            my_group = new_group
            my_group_id = group
    
    # Synchronize all processes before starting the test
    dist.barrier(group=world_group)

    if my_group is not None:
        # Perform all-reduce operation and measure bandwidth
        dist.all_reduce(npair_data, group=my_group)
        torch.cuda.synchronize()
        pre = time.perf_counter()
        dist.all_reduce(npair_data, group=my_group)
        torch.cuda.synchronize()
        duration = time.perf_counter() - pre
        busbw = ((size/g) / (duration)) *(2 * (16 - 1) / 16)
        tensor[test] = busbw

# Gather results from all processes
dist.all_gather(tensor_list=tensor_list, tensor=tensor, group=world_group)
del npair_data

# Calculate average bandwidth for each node
res = []
for i in range(node_world_size):
    tmp = torch.zeros(len(n_pairs), dtype=torch.float32).to('cuda')
    cnt = 0
    for j in range(local_world_size):
        tmp += tensor_list[(i * local_world_size) + j]
        cnt += 1
    tmp /= cnt
    res.append(tmp.tolist())

# Update the all-pair table with the maximum bandwidth values
for test in range(len(n_pairs)):
    for group in range(len(n_pairs[test])):
        allpair_table[allpairs_node[test][group][0]][allpairs_node[test][group][1]] = max(res[allpairs_node[test][group][0]][test], res[allpairs_node[test][group][1]][test])
        allpair_table[allpairs_node[test][group][1]][allpairs_node[test][group][0]] = max(res[allpairs_node[test][group][0]][test], res[allpairs_node[test][group][1]][test])

# Print the all-pair table if the current process is the root
if world_rank == 0:
    for i in range(node_world_size):
        print(f"\t\t node-{i}", end="\t")
    print("\n")

    for i in range(node_world_size):
        print(f"node-{i}", end="\t")
        for j in range(node_world_size):
            print(f"{allpair_table[i][j]:.2f}", end="\t")
        print("\n")
dist.destroy_process_group()
