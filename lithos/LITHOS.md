# LithOS Guide (libsmctrl)

This document explains the LithOS prototype in this repository: how it works, how to run it, and how to debug it.

It is intentionally separate from README.md.

## 1. What LithOS is in this repo

LithOS here is implemented as a CUDA Driver API interposition runtime layered on top of the `libcuda.so.1` wrapper approach used by `libsmctrl`.

High-level goals:

- Interpose kernel launches (`cuLaunchKernel`) and stream/sync entry points.
- Defer launches through a runtime queue.
- Apply TPC partitioning masks per launch (global mode) or per stream (local mode).
- Coordinate partition assignment across processes via a daemon (`lithosd`).

Main components:

- `libsmctrl.c`:
  - Existing mask machinery (global/stream/next masks).
  - `LIBSMCTRL_WRAPPER` constructor that initializes mask callback + LithOS runtime.
- `lithos/lithos_runtime.c`:
  - Interposed CUDA Driver runtime.
  - Deferred launch queue, dispatcher thread, launch-level allocation/free protocol.
- `lithos/lithosd.c`:
  - Global scheduler daemon.
  - Launch-keyed allocation table.
  - Status/profiling client mode (`--status`).
- `lithos/lithos_ipc.h`:
  - Runtime-daemon IPC protocol.

## 2. Execution model

### 2.1 Wrapper and interposition

`libcuda.so.1` in this repo is a wrapper shared object. CUDA Runtime and Driver calls route through this wrapper when it is found first in loader search path (for example, `LD_LIBRARY_PATH=.`).

### 2.2 Deferred dispatch

When LithOS is enabled:

- `cuLaunchKernel` calls are transformed to packed args and enqueued.
- A dispatcher thread dequeues items and launches kernels.
- For global mode, each launch asks `lithosd` for a partition just before dispatch.
- The assigned mask is applied via `libsmctrl_set_next_mask(...)` for that launch.

### 2.3 Completion and reclamation

Launch allocations are released at completion boundaries:

- Preferred path: event-based tracking when event symbols are available.
- Fallback path: reclaimed on stream/context synchronization boundaries.

This ensures global scheduler state is not leaked and allows reuse.

## 3. Scheduler modes

### 3.1 Local mode (single process policy)

Enable local scheduler policy:

- `LIBSMCTRL_LITHOS_ENABLE=1`
- `LIBSMCTRL_LITHOS_SCHED_ENABLE=1`

Optional quota controls:

- `LIBSMCTRL_LITHOS_TPC_QUOTA_DEFAULT=<n>`
- `LIBSMCTRL_LITHOS_TPC_QUOTAS=<comma-separated per-stream quotas>`

### 3.2 Global mode (cross-process policy)

Enable global daemon scheduling:

- `LIBSMCTRL_LITHOS_ENABLE=1`
- `LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE=1`
- `LIBSMCTRL_LITHOSD_SOCK=<unix socket path>`

Defaults when LithOS is enabled:

- Global scheduler is enabled by default.
- If `LIBSMCTRL_LITHOSD_SOCK` is unset, default socket is `/tmp/lithosd.sock`.

To force local mode explicitly:

- `LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE=0`
- `LIBSMCTRL_LITHOS_SCHED_ENABLE=1`

In this mode, wrapper clients request partitions from `lithosd` per launch.

## 4. MPS behavior

Process co-running on NVIDIA GPUs usually requires MPS.

Current ownership is daemon-side:

- `lithosd` attempts to ensure MPS is running at startup.
- This avoids each wrapper process trying to bootstrap MPS independently.

Daemon MPS controls:

- `LIBSMCTRL_LITHOS_MPS_AUTOSTART=0`
  - Disable daemon auto-start of MPS.
- `LIBSMCTRL_LITHOS_MPS_VISIBLE_DEVICES=<list>`
  - If `CUDA_VISIBLE_DEVICES` is unset, daemon can set it before starting MPS.

Related standard CUDA envs:

- `CUDA_VISIBLE_DEVICES`
- `CUDA_MPS_PIPE_DIRECTORY`

Note: if a shared MPS daemon was started with all visible devices, it may appear on all GPUs. To isolate by GPU, use per-GPU visibility and (optionally) per-pipe directories.

## 5. Build and basic run

### 5.1 Build core binaries

```bash
make libcuda.so.1 lithosd
```

### 5.2 Start daemon

```bash
./lithosd /tmp/lithosd.sock 40
```

Here `40` is number of TPCs exposed to scheduler policy for that GPU context.

### 5.3 Run an app in global mode

```bash
LIBSMCTRL_LITHOS_ENABLE=1 \
LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE=1 \
LIBSMCTRL_LITHOSD_SOCK=/tmp/lithosd.sock \
LD_LIBRARY_PATH=. \
./your_driver_api_or_runtime_app
```

## 6. Profiling/status interface

`lithosd` also supports status client mode:

```bash
./lithosd --status /tmp/lithosd.sock
```

Output includes:

- daemon header with `num_tpcs`, `used_tpcs`, `active_allocs`
- per-slot active allocation rows with:
  - `pid`, `launch`, `stream`, `quota`, `start_tpc`, `disable_mask`

This is the authoritative scheduler-side view.

## 7. Tests

### 7.1 LithOS suite

```bash
make run_lithos_tests
```

Covers:

- pass-through launch behavior
- packed/deferred path
- scheduler quota path
- kernelParams scheduling
- terminal-style global scheduler test
- framework smoke target

### 7.2 Long-running overlap stress

Script:

- `lithos/tests/lithos_stress_long_overlap.sh`

Usage:

```bash
./lithos/tests/lithos_stress_long_overlap.sh [apps] [tpcs] [cycles] [outdir]
```

Defaults:

- `apps=40`
- `tpcs=40`
- `cycles=20000000000`
- `outdir=/tmp/lithos_stress_<timestamp>`

Useful env toggles:

- `LITHOS_STRESS_VERBOSE=1|0`
- `LITHOS_STRESS_MONITOR_TIMEOUT_SEC=<seconds>`

Artifacts written to output dir:

- `daemon.log`
- `status.log`
- `status_peak.txt`
- `nvidia_smi.log`
- `app_*.out`, `app_*.err`
- `summary.txt`

## 8. Interpreting disjointness results

There are two disjointness signals in stress output:

- `pairwise_disjoint`:
  - Computed from app-reported SMID min/max ranges.
  - Can report false overlap depending on measurement characteristics.
- `peak_status_disjoint`:
  - Computed from scheduler slot rows at peak (`start_tpc`, `quota`).
  - This is the scheduler-ground-truth metric.

Prefer `peak_status_disjoint` for allocation correctness.

## 9. Common troubleshooting

### 9.1 No confinement effect

Check:

- wrapper in use (`LD_LIBRARY_PATH=.` for local runs)
- LithOS envs set
- daemon reachable (`LIBSMCTRL_LITHOSD_SOCK`)

### 9.2 Daemon unreachable fallback

Runtime may log fallback to unpartitioned launches if daemon RPC fails.

Validate socket path and daemon liveness:

```bash
./lithosd --status /tmp/lithosd.sock
```

### 9.3 MPS warnings

If daemon cannot start MPS automatically, workloads still run but may time-slice.

Check:

- `nvidia-cuda-mps-control` availability
- permissions for MPS pipe/log directories
- `CUDA_VISIBLE_DEVICES` and `CUDA_MPS_PIPE_DIRECTORY`

### 9.4 Stress script appears stuck

The script now treats zombie children as finished and has monitor timeout.

If needed, set shorter timeout:

```bash
LITHOS_STRESS_MONITOR_TIMEOUT_SEC=180 ./lithos/tests/lithos_stress_long_overlap.sh ...
```

## 10. Key files map

- `lithos/lithos_runtime.c`: runtime interposition, queue, dispatch, completion/reclaim
- `lithos/lithosd.c`: global scheduler daemon and status interface
- `lithos/lithos_ipc.h`: IPC request/response protocol
- `lithos/tests/lithos_stress_long_overlap.sh`: long-running overlap stress harness
- `lithos/tests/lithos_test_terminal_arbitrary.c`: terminal-style two-app confinement/disjoint test
- `lithos/tests/lithos_test_arbitrary_app.c`: parseable app used by terminal and stress tests

## 11. Notes on scope

This is a prototype implementation focused on deterministic behavior and validation of launch-level partition control. It does not yet implement advanced dynamic stealing/prediction policies from full production schedulers.
