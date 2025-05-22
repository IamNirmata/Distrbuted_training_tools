#!/usr/bin/env bash
# ib_bw_watch_tab.sh   --  show per-device TX bandwidth in MB/s as a table
# Usage:  ./ib_bw_watch_tab.sh <highest_idx> [interval]
# Example: ./ib_bw_watch_tab.sh 7          # watch mlx5_ib0 … mlx5_ib7 every 1 s
#          ./ib_bw_watch_tab.sh 3 2        # …every 2 s

N=${1:-0}          # highest mlx5_ib index to watch
INT=${2:-1}        # sampling interval in seconds
COLW=12            # column width (chars)

# ---------- helper ----------
read_ctr () { cat "/host/sys/class/infiniband/mlx5_ib$1/ports/1/counters/port_xmit_data"; }

# ---------- print header ----------
printf "%-${COLW}s" "Time"
for i in $(seq 0 "$N"); do
    printf "%-${COLW}s" "mlx5_ib$i"
done
printf "\n"

# ---------- prime the “old” array ----------
for i in $(seq 0 "$N"); do
    old[$i]=$(read_ctr "$i")
done

# ---------- main loop ----------
while true; do
    sleep "$INT"
    printf "%-${COLW}s" "$(date +%H:%M:%S)"
    for i in $(seq 0 "$N"); do
        new=$(read_ctr "$i")
        delta_mb=$(( (new - old[$i]) / 262144 ))   # words→bytes→MB
        printf "%-${COLW}s" "${delta_mb}MB/s"
        old[$i]=$new
    done
    printf "\n"
done
