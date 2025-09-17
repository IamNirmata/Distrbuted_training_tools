# HOSTFILE=/home/hari/b200/Distrbuted_training_tools/test_files/hostfile \
# NPERNODE=1 \
# NP_TOTAL=2 \
# LOGDIR=/tmp/allpairs_test \
# MASTER_PORT_BASE=49000 \
# GEN_SCRIPT=/home/hari/b200/Distrbuted_training_tools/generate_permutations.py \
# LOGDIR=/home/hari/b200/Distrbuted_training_tools/allpairs_logs \
# EXTRA_MPI_ARGS="--oversubscribe" \
# bash /home/hari/b200/Distrbuted_training_tools/run_allpair_test.sh

#git clone https://github.com/IamNirmata/Distrbuted_training_tools.git /opt/Distrbuted_training_tools
HOSTFILE=/opt/hostfile \
NPERNODE=8 \
NP_TOTAL=16 \
LOGDIR=/data/allpairs_logs \
MASTER_PORT_BASE=45567 \
GEN_SCRIPT=/opt/Distrbuted_training_tools/generate_permutations.py \
LOGDIR=/home/hari/b200/Distrbuted_training_tools/allpairs_logs \
bash /opt/Distrbuted_training_tools/run_allpair_test.sh