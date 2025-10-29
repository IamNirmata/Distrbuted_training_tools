set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends openssh-server openssh-client ca-certificates \
ibverbs-utils rdmacm-utils perftest infiniband-diags
mkdir -p /run/sshd && ssh-keygen -A
/usr/sbin/sshd -D -e &
for host in ${HOSTNAME//,/ }; do echo "$host slots=8"; done > /opt/hostfile
for host in ${MASTER_ADDR//,/ }; do echo "$host slots=8"; done >> /opt/hostfile

# export gcrnode=$(cat /opt/gcrnode.txt)
# echo "GCR Node name: $gcrnode"




pip install -U datasets
pip install -U wandb transformers peft bitsandbytes accelerate huggingface_hub trl
python -m pip install --upgrade pip

chmod +x ./launch.sh



# Logging & safer error surfacing
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV,NET
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Fabric detection / plugin toggles (common H100 fixes)
export NCCL_NVLS_ENABLE=0           # disable NVLink-SHARP plugin
export NCCL_SHARP_DISABLE=1         # disable SHARP if IB switch doesn't support it
export NCCL_P2P_DISABLE=0           # keep P2P unless you suspect NVLink issues; set to 1 if still crashing
export NCCL_NET_GDR_LEVEL=PHB       # be conservative with GPUDirect-RDMA
# export NCCL_IB_DISABLE=1          # try this only to fall back to sockets as a last resort

# Interface hints (adjust if your pods use a different NIC name)
export NCCL_SOCKET_IFNAME=eth0
# If you use IB inside pods and device is named e.g. mlx5_0:
# export NCCL_IB_HCA=mlx5_0

# Rendezvous sanity
export MASTER_PORT=23456
export OMP_NUM_THREADS=1












# REPO_DIR=/workspace/distrbuted_training_tools
# git clone https://github.com/IamNirmata/distrbuted_training_tools.git "$REPO_DIR"


bash data.sh
bash model.sh
chmod +x ./launch.sh
