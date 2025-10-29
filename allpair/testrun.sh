# Define your variables first
node1="llama3-finetune-sleep-job-master-0"
node2="llama3-finetune-sleep-job-worker-0"
NPERNODE=8
master_port=29500 # Example: PyTorch default port
NET_INTERFACE="eth0" # IMPORTANT: Check your pod's interface name with `ip a`

# Construct the mpirun command array
mp_cmd=(
  mpirun
  --allow-run-as-root
  --tag-output
  --display-map
  
  # Process mapping
  -np 16
  -H "${node1}:${NPERNODE},${node2}:${NPERNODE}"
  
  # --- Recommended Flags for GPU/Network ---
  
  # Prevent MPI from binding processes to specific CPU cores,
  # which can interfere with GPU affinity.
  --bind-to none
  
  # Tell MPI which network interface to use for its own communication
  --mca btl_tcp_if_include "${NET_INTERFACE}"
  --mca oob_tcp_if_include "${NET_INTERFACE}"

  # --- Environment Variables (-x) ---
  
  # Set NCCL debug level
  -x NCCL_DEBUG=INFO
  
  # Tell NCCL to use the same network interface
  -x NCCL_SOCKET_IFNAME="${NET_INTERFACE}"
  
  # Corrected syntax for MASTER_ADDR
  # Use ${node1} as the master
  -x MASTER_ADDR="${node1}"
  
  # Corrected syntax for MASTER_PORT
  -x MASTER_PORT="${master_port}"
  
  # --- Your Executable ---
  
  # It's safer to explicitly call python
  python ./npairs.py
)

# --- To run the command ---
#
# echo "Executing command: ${mp_cmd[@]}"
# "${mp_cmd[@]}"