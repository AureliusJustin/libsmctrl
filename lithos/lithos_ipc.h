/*
 * IPC protocol between libsmctrl LithOS wrapper clients and lithosd.
 */
#pragma once

#include <stdint.h>

#define LITHOSD_OP_ALLOC_LAUNCH 1u
#define LITHOSD_OP_FREE_LAUNCH  2u
#define LITHOSD_OP_GET_STATS    3u
#define LITHOSD_OP_GET_STATUS   4u

struct lithosd_req {
	uint32_t op;
	uint32_t launch_id;
	uint32_t stream_id;
	uint32_t quota;
	uint32_t grid_x;
	uint32_t grid_y;
	uint32_t grid_z;
	uint32_t block_x;
	uint32_t block_y;
	uint32_t block_z;
	uint32_t shared_mem;
	uint32_t reserved;
	uint64_t pid;
};

struct lithosd_resp {
	int32_t status;
	uint32_t has_partition;
	uint64_t disable_mask;
	uint32_t active_allocs;
	uint32_t num_tpcs;
	uint64_t used_tpcs;
	uint32_t alloc_in_use;
	uint64_t alloc_pid;
	uint32_t alloc_launch_id;
	uint32_t alloc_stream_id;
	uint32_t alloc_quota;
	uint32_t alloc_start_tpc;
	uint64_t alloc_disable_mask;
};
