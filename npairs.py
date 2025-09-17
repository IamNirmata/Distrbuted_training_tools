import torch.distributed as dist
import os
import time
import torch

# Retrieve environment variables for distributed training configuration        
local_world_size = int(os.environ.get("LOCAL_WORLD", 1))     
local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK", 0))                                        
world_size = int(os.environ.get("OMPI_COMM_WORLD_SIZE", 1))                                        
world_rank = int(os.environ.get("OMPI_COMM_WORLD_RANK", 0))                                              
node_rank = world_rank // local_world_size         
node_world_size = world_size // local_world_size      
ITERATIONS = 7

# Create local groups for each node

# Initialize the process group for distributed training
dist.init_process_group("nccl", init_method="env://", rank=world_rank, world_size=world_size)
world_group = dist.group.WORLD 

# Set the current CUDA device to the local rank
torch.cuda.set_device(local_rank)

g = 1024*1024*1024  # Define 1 GB in bytes
size = 8*g*2  # Total size of data to be reduced

# Allocate memory for the data to be reduced
npair_data = torch.zeros(8*g, dtype=torch.bfloat16).to('cuda')

dist.all_reduce(npair_data)
torch.cuda.synchronize()
pre = time.perf_counter()
for _ in range(ITERATIONS):
    dist.all_reduce(npair_data)
torch.cuda.synchronize()
duration = (time.perf_counter() - pre) / ITERATIONS
busbw = ((size/g) / (duration)) *(2 * (world_size - 1) / world_size)
print(f"latency: {duration} busbw: {busbw}")


dist.destroy_process_group()
