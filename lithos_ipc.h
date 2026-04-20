/*
 * IPC protocol between libsmctrl LithOS wrapper clients and lithosd.
 */
#pragma once

#include <stdint.h>

#define LITHOSD_OP_ALLOC_STREAM 1u
#define LITHOSD_OP_FREE_STREAM  2u
#define LITHOSD_OP_GET_STATS    3u

struct lithosd_req {
	uint32_t op;
	uint32_t stream_id;
	uint32_t quota;
	uint32_t reserved;
	uint64_t pid;
};

struct lithosd_resp {
	int32_t status;
	uint32_t has_partition;
	uint64_t disable_mask;
	uint32_t active_allocs;
	uint32_t reserved;
};
