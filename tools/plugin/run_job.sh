#!/usr/bin/env bash
# run_job.sh GPU ENGINE NAME KEY CONC PROMPTS [RLT_CHUNK] [EXTRA_RLT_FLAGS...]
# Starts one server on one GPU, runs the benchmark, stops the server.
set -u
GPU=$1 ENGINE=$2 NAME=$3 KEY=$4 CONC=$5 PROMPTS=$6 CHUNK=${7:-1024}; shift 7 || shift $#
W=$HOME/code/david/b_workspace/rlt-plugin; PY=$W/.venv/bin/python; M=/dev/shm/zjy-inkling2-mini/varloop
PORT=$((19600 + GPU)); LOG=$W/logs/$NAME.server.log; OUT=$W/results/$NAME.json
mkdir -p $W/results; export CUDA_VISIBLE_DEVICES=$GPU
PLUG=""
case $ENGINE in
  plugin) PLUG="--scheduler-cls vllm_rlt.plugin.scheduler.LoopScheduler --worker-cls vllm_rlt.plugin.worker.LoopWorker" ;;
  plugin-split) PLUG="--scheduler-cls vllm_rlt.plugin.scheduler.LoopScheduler --worker-cls vllm_rlt.plugin.worker.LoopWorker --additional-config {\"rlt_split_ops\":true}" ;;
  plugin-graphs) PLUG="--scheduler-cls vllm_rlt.plugin.scheduler.LoopScheduler --worker-cls vllm_rlt.plugin.worker.LoopWorker --additional-config {\"rlt_graphs\":true}" ;;
esac
if [ "$ENGINE" = vllm ] || [ -n "$PLUG" ]; then
  VLLM_KV_CACHE_LAYOUT=LBHNC $W/.venv/bin/vllm serve $M --language-model-only --tensor-parallel-size 1 \
    --max-model-len 8192 --no-enable-prefix-caching --enable-chunked-prefill --generation-config vllm \
    --dtype bfloat16 --max-num-seqs 32 --max-num-batched-tokens 1024 --async-scheduling \
    --hf-overrides '{"text_config":{"loop_num_passes":3,"loop_share_cache":false}}' \
    --block-size 16 --kv-cache-dtype bfloat16 --disable-hybrid-kv-cache-manager \
    --num-gpu-blocks-override 8449 --served-model-name inkling --host 127.0.0.1 --port $PORT $PLUG > $LOG 2>&1 &
else
  $PY -m vllm_rlt.entrypoints.serve --model $M --max-loops 3 --device cuda --dtype bfloat16 \
    --mode mixed --packed-prefill --prefill-chunk-size $CHUNK --attention-backend triton \
    --num-blocks 25344 --block-size 16 --max-num-seqs 32 --max-num-batched-tokens 1024 \
    --request-timeout 600 --cuda-graphs --cuda-graph-max-batch-size 96 --cuda-graph-max-graphs 96 \
    --packed-graph-buckets --cuda-graph-warmup --min-coda-batch-size 1 --async-scheduling \
    --served-model-name inkling --host 127.0.0.1 --port $PORT "$@" > $LOG 2>&1 &
fi
SRV=$!
for i in $(seq 1 360); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $SRV 2>/dev/null || { echo "server died"; tail -20 $LOG; exit 1; }
  sleep 5
done
IGN=1; MAXT=128; [ "$KEY" = W7 ] && { IGN=0; MAXT=512; }
$PY $W/bench/bench.py $PORT $W/bench/$PROMPTS $KEY $CONC $MAXT $IGN 3 $OUT 2>&1 | tail -3
pkill -P $SRV 2>/dev/null; kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; sleep 5
