set -eo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends openssh-server openssh-client ca-certificates \
ibverbs-utils rdmacm-utils perftest infiniband-diags
mkdir -p /run/sshd && ssh-keygen -A
/usr/sbin/sshd -D -e &
for host in ${VC_SERVER_HOSTS//,/ }; do echo "$host slots=8"; done > /opt/hostfile
for host in ${VC_CLIENT_HOSTS//,/ }; do echo "$host slots=8"; done >> /opt/hostfile