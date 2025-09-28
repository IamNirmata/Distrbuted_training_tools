mkdir -p /data/cluster_validation/dltest # create if not exists

today=$(date +%Y%m%d)
if [ ! -d /data/cluster_validation/dltest/"$today" ]; then
  mkdir /data/cluster_validation/dltest/"$today"
fi
ln -s /data/cluster_validation/dltest/"$today" /data/cluster_validation/dltest/latest

