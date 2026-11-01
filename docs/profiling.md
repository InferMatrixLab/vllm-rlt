# PyTorch profiling

Profiling is disabled by default. Collection options are independent and use PyTorch Profiler. Each rank exports its own trace. A background thread parses the completed files, creates a verified `.tar.gz` archive, and removes the standalone raw files. This implements the core interface proposed in [RFC #71](https://github.com/InferMatrixLab/vllm-rlt/issues/71).

## Offline CLI

```bash
vllm-rlt --model /path/to/model --device cuda --attention-backend triton \
  --prompt "Explain matrix multiplication." --max-tokens 128 \
  --profile --profile-dir /tmp/rlt-profiles \
  --profile-activities cpu,cuda \
  --profile-record-shapes --profile-with-stack --profile-memory \
  --profile-wait 2 --profile-warmup 1 --profile-active 10 --profile-repeat 1
```

The same profiling flags are accepted by `vllm-rlt-serve` and `vllm-rlt-pd-serve`. Startup collection starts after model initialization. The offline command waits for artifact processing before exit and writes profiling status to stderr, leaving generated outputs on stdout.

| Option | Default | Purpose |
| --- | --- | --- |
| `--profile` | disabled | Enable startup collection |
| `--profile-dir` | required when enabled | Output root |
| `--profile-activities` | `cpu` on CPU; `cpu,cuda` on CUDA | Activity collection, not model placement |
| `--profile-record-shapes` | off | Operator input shapes |
| `--profile-with-stack` | off | Source stacks |
| `--profile-memory` | off | Tensor allocations and frees |
| `--profile-with-flops` | off | FLOP estimates for supported operators |
| `--profile-wait` | 0 | Wait steps per cycle |
| `--profile-warmup` | 1 | Profiler warmup steps per cycle |
| `--profile-active` | 10 | Recorded steps per cycle |
| `--profile-repeat` | 1 | Cycle count; 0 repeats until explicitly stopped |

An ordinary engine step is one `LLMEngine.step()` invocation, not necessarily one token. PD prefill advances after an iteration that submits work; decode advances after its engine step. Idle polling does not advance the schedule. Finite schedules automatically stop collection after their final cycle. A short workload can leave an incomplete or empty window; this is recorded in the result.

## Python control

`LLM` starts a spawned worker process. In Python scripts, create it inside
`main()` and use the `if __name__ == "__main__":` guard shown below. The context
manager closes the worker when inference finishes or raises.

```python
from vllm_rlt import LLM, ProfileConfig


def main():
    config = ProfileConfig(
        output_dir="/tmp/rlt-profiles",
        activities=("cpu", "cuda"),
        record_shapes=True,
    )
    with LLM("/path/to/model", device="cuda", attention_backend="triton") as llm:
        llm.start_profile(config)  # Continuous collection until stop_profile().
        outputs = llm.generate(["Explain matrix multiplication."])
        llm.stop_profile()  # Enqueue artifacts; do not wait for parsing/compression.
        status = llm.wait_for_profile_artifacts()


if __name__ == "__main__":
    main()
```

`start_profile(config, scheduled=True)` uses the configured step schedule. Plain `start_profile(config)` uses an explicitly controlled window. `profile_status()` reports the current session and per-cycle jobs. `wait_for_profile_artifacts(timeout=None)` waits after collection stops. Inspect errors and job states as well as `artifacts_complete`: processing can finish with a failed artifact. A new session cannot overlap an active session or unfinished processing from the previous session.

The engine owns the profiler, and start/stop/step must execute on its owner thread. The serving layer handles this routing. `close()` stops collection and joins the artifact worker. The context-manager form ensures cleanup when inference raises.

## Serving and PD

After the server is ready, start a continuous session:

```bash
curl -X POST http://localhost:8000/profile/start \
  -H 'Content-Type: application/json' \
  -d '{"output_dir":"/tmp/rlt-profiles","activities":["cpu","cuda"],"record_shapes":true}'
```

Submit requests after this call returns. For PD, success means all P/D workers have acknowledged starting their local profiler. When the target requests and related transfers finish, stop collection and wait for artifacts:

```bash
curl -X POST http://localhost:8000/profile/stop
curl http://localhost:8000/profile/status
curl -X POST http://localhost:8000/profile/wait \
  -H 'Content-Type: application/json' -d '{"timeout":60}'
```

The HTTP start operation enables profiling by default. Add `"scheduled":true` to use `wait`, `warmup`, `active` and `repeat`; explicit schedule fields without it are rejected. Startup CLI collection is always step-scheduled.

PD workers use a shared session ID and distinct ranks. Control acknowledgments are handled independently from request/transfer messages, including while workers are idle. Partial start failures trigger stop commands for all participants and are retained in the session manifest. Equal local step numbers do not represent the same moment across ranks. To cover the same requests, use the explicit start/stop sequence above. On a busy service, other concurrent work may appear in the traces.

## Artifacts

```text
<output-dir>/
  manifest-20260930T021015Z.json
  rank-00000-20260930T021015Z/
    status.json
    cycle-00000.tar.gz
  rank-00001-20260930T021015Z/
    status.json
    cycle-00000.tar.gz
```

Directory names use the rank and UTC capture timestamp to the second. Captures started in the same second receive a collision suffix (`-2`, `-3`, etc.) to prevent overwrites. PD workers share the coordinator's timestamp and suffix. Session IDs remain internal control identifiers and are not used in paths. Each capture has its own timestamped manifest.

Each archive contains:

- `trace.json`: native Chrome trace of the configured CPU/CUDA activities.
- `events.jsonl`: immutable native event snapshots captured before the profiler releases the cycle.
- `operators.csv`: trace event counts and inclusive durations, with available process/thread/device/stream attribution.
- `operator_metrics.csv`: native CPU/device inclusive and self times, shapes, stacks and FLOPs, aggregated in the background.
- `summary.json` and `metadata.json`: collection settings, rank/host/device identity, timing coverage and limitations.
- `inventory.json`: archived file sizes and SHA-256 checksums.
- Stack exports and summaries when stacks are enabled; memory timeline and summary when memory, shapes and stacks are all enabled.

FLOPs and other unavailable fields are not inferred for unsupported operators. Trace event durations can overlap and must not be summed as wall-clock time. Memory summaries describe allocations visible to PyTorch, not total device memory. CUDA Graph and transport visibility depends on the native profiler/backend; the native profiler does not provide a complete NIXL/RDMA trace.

Parsing and compression run on one background thread per rank. Native profiler finalization, export and event snapshotting still run on the owner thread. Background work shares CPU, memory and the Python GIL; it does not eliminate resource contention. Raw files are deleted only after archive contents have been verified and the final archive published. On failure, raw files remain in that cycle's staging directory and status includes the error. Archives retain the original data for inspection and reprocessing.
