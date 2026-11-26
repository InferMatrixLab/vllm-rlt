#!/usr/bin/env bash
# queue.sh JOBFILE : each line = arguments to run_job.sh minus GPU. Runs jobs on any GPU
# whose memory use is below 2 GiB, one job per GPU, polling every 60 s. Never touches busy GPUs.
W=$HOME/code/david/b_workspace/rlt-plugin; JOBS=$1; mkdir -p $W/logs/q
declare -A BUSY
mapfile -t LINES < <(grep -v '^#' $JOBS | grep -v '^$')
i=0
while [ $i -lt ${#LINES[@]} ] || [ ${#BUSY[@]} -gt 0 ]; do
  for g in "${!BUSY[@]}"; do kill -0 ${BUSY[$g]} 2>/dev/null || unset BUSY[$g]; done
  while read -r g mem; do
    [ $i -ge ${#LINES[@]} ] && break
    [ -n "${BUSY[$g]:-}" ] && continue
    [ "$mem" -gt 2048 ] && continue
    name=$(echo ${LINES[$i]} | awk '{print $2}')
    echo "$(date -u +%FT%TZ) start gpu$g ${LINES[$i]}" >> $W/logs/q/queue.log
    bash $W/bench/run_job.sh $g ${LINES[$i]} > $W/logs/q/$name.log 2>&1 &
    BUSY[$g]=$!; i=$((i+1)); sleep 20
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr -d ',')
  sleep 60
done
echo "$(date -u +%FT%TZ) all done" >> $W/logs/q/queue.log
