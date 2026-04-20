/*
 * Copyright 2026 Justin and contributors
 *
 * LithOS runtime prototype layered on top of libsmctrl's libcuda.so.1 wrapper.
 *
 * This intentionally keeps policy simple and deterministic so behavior can be
 * validated before adding dynamic stealing and prediction logic.
 */
#include "lithos_runtime.h"

// #ifdef LIBSMCTRL_WRAPPER

#include "libsmctrl.h"
#include "lithos_ipc.h"

#include <cuda.h>

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

/*** Runtime State ***/

struct stream_state {
	CUstream stream;
	uint32_t stream_id;
	bool has_partition;
	uint64_t disable_mask;
	uint64_t pending;
	uint64_t inflight;
	struct stream_state* next;
};

struct launch_item {
	uint32_t launch_id;
	uint32_t stream_id;
	CUfunction f;
	unsigned int gx, gy, gz;
	unsigned int bx, by, bz;
	unsigned int shared_mem;
	CUstream stream;
	void* arg_buffer;
	size_t arg_buffer_size;
	struct launch_item* next;
};

struct active_launch {
	uint32_t launch_id;
	CUstream stream;
	CUevent done_event;
	bool has_done_event;
	bool has_partition;
	uint64_t disable_mask;
	struct active_launch* next;
};

static struct {
	pthread_once_t once;
	bool initialized;
	bool real_ready;
	bool enabled;
	bool shutting_down;
	void* real_cuda;
	CUresult (*real_cuStreamCreate)(CUstream*, unsigned int);
	CUresult (*real_cuStreamCreateWithPriority)(CUstream*, unsigned int, int);
	CUresult (*real_cuStreamDestroy)(CUstream);
	CUresult (*real_cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
	                                unsigned int, unsigned int, unsigned int,
	                                unsigned int, CUstream, void**, void**);
	CUresult (*real_cuFuncGetParamInfo)(CUfunction, size_t, size_t*, size_t*);
	CUresult (*real_cuEventCreate)(CUevent*, unsigned int);
	CUresult (*real_cuEventRecord)(CUevent, CUstream);
	CUresult (*real_cuEventQuery)(CUevent);
	CUresult (*real_cuEventDestroy)(CUevent);
	CUresult (*real_cuStreamSynchronize)(CUstream);
	CUresult (*real_cuCtxSynchronize)(void);
	CUresult (*real_cuStreamQuery)(CUstream);
	CUresult (*real_cuStreamIsCapturing)(CUstream, CUstreamCaptureStatus*);

	pthread_t dispatcher;
	pthread_mutex_t mu;
	pthread_cond_t cv;
	struct launch_item* q_head;
	struct launch_item* q_tail;
	struct stream_state* streams;
	struct active_launch* active_launches;
	CUresult dispatch_error;

	bool scheduler_enabled;
	uint32_t num_tpcs;
	uint32_t default_quota;
	uint32_t next_tpc_cursor;
	uint32_t next_stream_id;
	uint32_t next_launch_id;
	bool oversubscribe_warned;
	uint32_t quota_list[128];
	uint32_t quota_count;
	bool daemon_enabled;
	bool daemon_warned;
	bool launch_track_warned;
	bool capture_bypass_warned;
	bool capture_passthrough_mode;
	char daemon_sock_path[108];
} g_lithos = {
	.once = PTHREAD_ONCE_INIT,
	.initialized = false,
	.real_ready = false,
	.enabled = false,
	.shutting_down = false,
	.real_cuda = NULL,
	.real_cuStreamCreate = NULL,
	.real_cuStreamCreateWithPriority = NULL,
	.real_cuStreamDestroy = NULL,
	.real_cuLaunchKernel = NULL,
	.real_cuFuncGetParamInfo = NULL,
	.real_cuEventCreate = NULL,
	.real_cuEventRecord = NULL,
	.real_cuEventQuery = NULL,
	.real_cuEventDestroy = NULL,
	.real_cuStreamSynchronize = NULL,
	.real_cuCtxSynchronize = NULL,
	.real_cuStreamQuery = NULL,
	.real_cuStreamIsCapturing = NULL,
	.dispatcher = 0,
	.mu = PTHREAD_MUTEX_INITIALIZER,
	.cv = PTHREAD_COND_INITIALIZER,
	.q_head = NULL,
	.q_tail = NULL,
	.streams = NULL,
	.active_launches = NULL,
	.dispatch_error = CUDA_SUCCESS,
	.scheduler_enabled = false,
	.num_tpcs = 0,
	.default_quota = 0,
	.next_tpc_cursor = 0,
	.next_stream_id = 0,
	.next_launch_id = 1,
	.oversubscribe_warned = false,
	.quota_list = {0},
	.quota_count = 0,
	.daemon_enabled = false,
	.daemon_warned = false,
	.launch_track_warned = false,
	.capture_bypass_warned = false,
	.capture_passthrough_mode = false,
	.daemon_sock_path = {0},
};

/*** Baseline Scheduler Helpers ***/

static uint64_t valid_tpc_bits_mask(void) {
	if (g_lithos.num_tpcs >= 64)
		return ~0ull;
	if (g_lithos.num_tpcs == 0)
		return 0;
	return (1ull << g_lithos.num_tpcs) - 1ull;
}

static uint32_t quota_for_stream_id(uint32_t stream_id) {
	if (stream_id < g_lithos.quota_count)
		return g_lithos.quota_list[stream_id];
	return g_lithos.default_quota;
}

static bool daemon_rpc(const struct lithosd_req* req, struct lithosd_resp* resp) {
	int fd;
	struct sockaddr_un addr;
	ssize_t n;

	// Best-effort RPC: failures are handled by fallback to unrestricted launches.
	if (!g_lithos.daemon_enabled || !g_lithos.daemon_sock_path[0])
		return false;

	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd == -1)
		return false;

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, g_lithos.daemon_sock_path, sizeof(addr.sun_path) - 1);
	if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
		close(fd);
		return false;
	}

	n = write(fd, req, sizeof(*req));
	if (n != (ssize_t)sizeof(*req)) {
		close(fd);
		return false;
	}
	n = read(fd, resp, sizeof(*resp));
	close(fd);
	if (n != (ssize_t)sizeof(*resp))
		return false;
	return true;
}

static bool parse_quota_list(const char* s, uint32_t* out, uint32_t* out_count) {
	char* tmp;
	char* tok;
	char* save = NULL;
	uint32_t count = 0;
	if (!s || !*s)
		return true;
	tmp = strdup(s);
	if (!tmp)
		return false;
	for (tok = strtok_r(tmp, ",", &save); tok; tok = strtok_r(NULL, ",", &save)) {
		char* end = NULL;
		unsigned long val;
		if (count >= 128) {
			free(tmp);
			return false;
		}
		val = strtoul(tok, &end, 10);
		if (!end || *end != '\0') {
			free(tmp);
			return false;
		}
		out[count++] = (uint32_t)val;
	}
	free(tmp);
	*out_count = count;
	return true;
}

static void configure_scheduler_from_env(void) {
	const char* g_en = getenv("LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE");
	const char* sock = getenv("LIBSMCTRL_LITHOSD_SOCK");
	const char* en = getenv("LIBSMCTRL_LITHOS_SCHED_ENABLE");
	const char* q_default = getenv("LIBSMCTRL_LITHOS_TPC_QUOTA_DEFAULT");
	const char* q_list = getenv("LIBSMCTRL_LITHOS_TPC_QUOTAS");
	bool global_requested = false;
	uint32_t tpcs = 0;
	int rc;

	// Scheduler policy is opt-in.
	g_lithos.scheduler_enabled = false;
	g_lithos.daemon_enabled = false;
	g_lithos.daemon_warned = false;
	g_lithos.daemon_sock_path[0] = '\0';
	g_lithos.default_quota = 0;
	g_lithos.quota_count = 0;
	if (q_default && *q_default)
		g_lithos.default_quota = (uint32_t)strtoul(q_default, NULL, 10);
	if (!parse_quota_list(q_list, g_lithos.quota_list, &g_lithos.quota_count)) {
		fprintf(stderr, "libsmctrl-lithos: scheduler disabled (invalid LIBSMCTRL_LITHOS_TPC_QUOTAS).\n");
		return;
	}

	// Mode selection precedence:
	// 1) Explicit global toggle (LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE)
	// 2) Explicit local toggle (LIBSMCTRL_LITHOS_SCHED_ENABLE)
	// 3) Default to global mode for LithOS-enabled runs
	if (g_en)
		global_requested = strcmp(g_en, "0") != 0;
	else if (en && strcmp(en, "1") == 0)
		global_requested = false;
	else
		global_requested = true;

	if (global_requested) {
		const char* resolved_sock = (sock && *sock) ? sock : "/tmp/lithosd.sock";
		strncpy(g_lithos.daemon_sock_path, resolved_sock, sizeof(g_lithos.daemon_sock_path) - 1);
		g_lithos.daemon_enabled = true;
		g_lithos.scheduler_enabled = true;
		fprintf(stderr, "libsmctrl-lithos: global scheduler mode enabled (socket=%s).\n", g_lithos.daemon_sock_path);
		return;
	}

	if (!en || strcmp(en, "1") != 0)
		return;

	rc = libsmctrl_get_tpc_info_cuda(&tpcs, 0);
	if (rc != 0 || tpcs == 0) {
		fprintf(stderr, "libsmctrl-lithos: scheduler disabled (unable to query TPC count).\n");
		return;
	}
	// The baseline scheduler currently applies masks through next-mask, which
	// in this implementation is limited to 64-bit masks.
	if (tpcs > 64) {
		fprintf(stderr, "libsmctrl-lithos: scheduler disabled (only <=64 TPCs currently supported by next-mask path).\n");
		return;
	}
	g_lithos.num_tpcs = tpcs;

	g_lithos.scheduler_enabled = true;
	fprintf(stderr,
	        "libsmctrl-lithos:  Scheduler enabled (%u TPCs, default quota=%u, explicit quotas=%u).\n",
	        g_lithos.num_tpcs, g_lithos.default_quota, g_lithos.quota_count);
}

static void assign_stream_partition_locked(struct stream_state* st) {
	uint32_t quota;
	uint64_t enable_mask;
	uint64_t valid_mask;
	if (!st || !g_lithos.scheduler_enabled)
		return;

	quota = quota_for_stream_id(st->stream_id);

	st->has_partition = false;
	st->disable_mask = 0;
	if (quota == 0)
		return;

	// This baseline policy statically slices TPCs in stream creation order.
	if (quota > g_lithos.num_tpcs || g_lithos.next_tpc_cursor + quota > g_lithos.num_tpcs) {
		if (!g_lithos.oversubscribe_warned) {
			fprintf(stderr,
			        "libsmctrl-lithos: quota oversubscription detected; falling back to unrestricted launches for excess streams.\n");
			g_lithos.oversubscribe_warned = true;
		}
		return;
	}

	if (quota >= 64)
		enable_mask = ~0ull;
	else
		enable_mask = ((1ull << quota) - 1ull) << g_lithos.next_tpc_cursor;
	valid_mask = valid_tpc_bits_mask();
	enable_mask &= valid_mask;
	// libsmctrl masks use "bit set = disabled" semantics.
	st->disable_mask = (~enable_mask) & valid_mask;
	st->has_partition = true;
	g_lithos.next_tpc_cursor += quota;
}

static bool daemon_alloc_launch(const struct launch_item* item, bool* has_partition, uint64_t* disable_mask) {
	struct lithosd_req req;
	struct lithosd_resp resp;
	uint32_t quota;

	if (!g_lithos.daemon_enabled)
		return false;

	quota = quota_for_stream_id(item->stream_id);
	memset(&req, 0, sizeof(req));
	memset(&resp, 0, sizeof(resp));
	req.op = LITHOSD_OP_ALLOC_LAUNCH;
	req.pid = (uint64_t)getpid();
	req.launch_id = item->launch_id;
	req.stream_id = item->stream_id;
	req.quota = quota;
	req.grid_x = item->gx;
	req.grid_y = item->gy;
	req.grid_z = item->gz;
	req.block_x = item->bx;
	req.block_y = item->by;
	req.block_z = item->bz;
	req.shared_mem = item->shared_mem;

	if (!daemon_rpc(&req, &resp)) {
		if (!g_lithos.daemon_warned) {
			fprintf(stderr, "libsmctrl-lithos: failed to contact global scheduler daemon; unpartitioned fallback will be used.\n");
			g_lithos.daemon_warned = true;
		}
		return false;
	}
	if (resp.status != 0)
		return false;
	*has_partition = resp.has_partition != 0;
	*disable_mask = resp.disable_mask;
	return true;
}

static void daemon_free_launch(uint32_t launch_id) {
	struct lithosd_req req;
	struct lithosd_resp resp;

	if (!g_lithos.daemon_enabled)
		return;
	memset(&req, 0, sizeof(req));
	memset(&resp, 0, sizeof(resp));
	req.op = LITHOSD_OP_FREE_LAUNCH;
	req.pid = (uint64_t)getpid();
	req.launch_id = launch_id;
	(void)daemon_rpc(&req, &resp);
}

static void fail_missing_symbol(const char* sym) {
	fprintf(stderr, "libsmctrl-lithos: missing CUDA symbol %s; disabling runtime.\n", sym);
	g_lithos.real_ready = false;
}

static void* resolve_real_symbol(const char* name) {
	void* ptr = dlsym(g_lithos.real_cuda, name);
	if (!ptr)
		fail_missing_symbol(name);
	return ptr;
}

/*** Stream and Queue Bookkeeping ***/

static struct stream_state* find_stream_state_locked(CUstream stream) {
	struct stream_state* it = g_lithos.streams;
	while (it) {
		if (it->stream == stream)
			return it;
		it = it->next;
	}
	return NULL;
}

static struct stream_state* get_or_create_stream_state_locked(CUstream stream) {
	struct stream_state* found = find_stream_state_locked(stream);
	if (found)
		return found;
	struct stream_state* st = calloc(1, sizeof(*st));
	if (!st)
		return NULL;
	st->stream = stream;
	// Stream IDs are process-local and used for quota lookup and daemon hints.
	st->stream_id = g_lithos.next_stream_id++;
	if (!g_lithos.daemon_enabled)
		assign_stream_partition_locked(st);
	st->next = g_lithos.streams;
	g_lithos.streams = st;
	return st;
}

static bool any_pending_or_inflight_locked(void) {
	if (g_lithos.q_head)
		return true;
	for (struct stream_state* st = g_lithos.streams; st; st = st->next) {
		if (st->pending || st->inflight)
			return true;
	}
	return false;
}

static bool has_active_launches_for_stream_locked(CUstream stream) {
	for (struct active_launch* it = g_lithos.active_launches; it; it = it->next) {
		if (it->stream == stream)
			return true;
	}
	return false;
}

static void add_active_launch_locked(uint32_t launch_id,
	                                 CUstream stream,
	                                 CUevent done_event,
	                                 bool has_done_event,
	                                 bool has_partition,
	                                 uint64_t disable_mask) {
	struct active_launch* node = calloc(1, sizeof(*node));
	if (!node) {
		daemon_free_launch(launch_id);
		if (!g_lithos.launch_track_warned) {
			fprintf(stderr, "libsmctrl-lithos: launch completion tracking OOM; reclaiming launch allocation eagerly.\n");
			g_lithos.launch_track_warned = true;
		}
		if (has_done_event && g_lithos.real_cuEventDestroy)
			(void)g_lithos.real_cuEventDestroy(done_event);
		return;
	}
	node->launch_id = launch_id;
	node->stream = stream;
	node->done_event = done_event;
	node->has_done_event = has_done_event;
	node->has_partition = has_partition;
	node->disable_mask = disable_mask;
	node->next = g_lithos.active_launches;
	g_lithos.active_launches = node;
}

static void reap_completed_launches_locked(CUstream stream, bool only_one_stream, bool force_complete) {
	struct active_launch** pp = &g_lithos.active_launches;
	while (*pp) {
		bool finished = false;
		struct active_launch* node = *pp;
		if (only_one_stream && node->stream != stream) {
			pp = &(*pp)->next;
			continue;
		}
		if (force_complete) {
			finished = true;
		} else if (node->has_done_event && g_lithos.real_cuEventQuery) {
			CUresult q = g_lithos.real_cuEventQuery(node->done_event);
			if (q == CUDA_SUCCESS)
				finished = true;
		}
		if (!finished) {
			pp = &(*pp)->next;
			continue;
		}
		if (node->has_done_event && g_lithos.real_cuEventDestroy)
			(void)g_lithos.real_cuEventDestroy(node->done_event);
		daemon_free_launch(node->launch_id);
		*pp = node->next;
		free(node);
	}
}

/*** Launch Argument Handling ***/

static bool parse_packed_args(void** extra, void** buf_out, size_t* sz_out) {
	void* buf = NULL;
	size_t sz = 0;
	if (!extra)
		return false;
	for (size_t i = 0;; i += 2) {
		void* tag = extra[i];
		if (tag == CU_LAUNCH_PARAM_END)
			break;
		if (!extra[i + 1])
			return false;
		if (tag == CU_LAUNCH_PARAM_BUFFER_POINTER) {
			buf = extra[i + 1];
		} else if (tag == CU_LAUNCH_PARAM_BUFFER_SIZE) {
			sz = *(size_t*)extra[i + 1];
		}
	}
	if (!buf || !sz)
		return false;
	*buf_out = buf;
	*sz_out = sz;
	return true;
}

// Convert kernelParams (array of pointers to argument values) into the packed
// launch buffer form so launches can be deferred safely.
static bool pack_kernel_params(CUfunction f, void** kernelParams, void** packed_out, size_t* packed_size_out) {
	const size_t max_params = 256;
	size_t max_end = 0;
	size_t num_params = 0;
	void* packed;

	if (!g_lithos.real_cuFuncGetParamInfo || !kernelParams)
		return false;

	for (size_t i = 0; i < max_params; i++) {
		size_t off = 0;
		size_t sz = 0;
		CUresult res = g_lithos.real_cuFuncGetParamInfo(f, i, &off, &sz);
		if (res == CUDA_ERROR_INVALID_VALUE)
			break;
		if (res != CUDA_SUCCESS)
			return false;
		if (!kernelParams[i])
			return false;
		if (off + sz > max_end)
			max_end = off + sz;
		num_params++;
	}

	if (num_params == 0 || max_end == 0)
		return false;

	packed = calloc(1, max_end);
	if (!packed)
		return false;

	for (size_t i = 0; i < num_params; i++) {
		size_t off = 0;
		size_t sz = 0;
		CUresult res = g_lithos.real_cuFuncGetParamInfo(f, i, &off, &sz);
		if (res != CUDA_SUCCESS) {
			free(packed);
			return false;
		}
		memcpy((char*)packed + off, kernelParams[i], sz);
	}

	*packed_out = packed;
	*packed_size_out = max_end;
	return true;
}

/*** Real Driver Callthrough ***/

static CUresult direct_launch(CUfunction f,
                              unsigned int gx,
                              unsigned int gy,
                              unsigned int gz,
                              unsigned int bx,
                              unsigned int by,
                              unsigned int bz,
                              unsigned int shared_mem,
                              CUstream stream,
                              void** kernelParams,
                              void** extra) {
	return g_lithos.real_cuLaunchKernel(f, gx, gy, gz, bx, by, bz, shared_mem,
	                                    stream, kernelParams, extra);
}

static bool stream_is_capturing(CUstream stream) {
	CUstreamCaptureStatus status = CU_STREAM_CAPTURE_STATUS_NONE;

	if (!g_lithos.real_cuStreamIsCapturing)
		return false;
	if (g_lithos.real_cuStreamIsCapturing(stream, &status) != CUDA_SUCCESS)
		return false;
	return status != CU_STREAM_CAPTURE_STATUS_NONE;
}

/*** Dispatcher Thread ***/

static void* dispatcher_main(void* arg) {
	(void)arg;
	while (1) {
		bool has_partition = false;
		uint64_t disable_mask = 0;
		CUevent done_event = NULL;
		bool tracked_with_event = false;

		pthread_mutex_lock(&g_lithos.mu);
		reap_completed_launches_locked(NULL, false, false);
		while (!g_lithos.shutting_down && !g_lithos.q_head)
			pthread_cond_wait(&g_lithos.cv, &g_lithos.mu);
		if (g_lithos.shutting_down && !g_lithos.q_head) {
			pthread_mutex_unlock(&g_lithos.mu);
			break;
		}
		struct launch_item* item = g_lithos.q_head;
		g_lithos.q_head = item->next;
		if (!g_lithos.q_head)
			g_lithos.q_tail = NULL;
		struct stream_state* st = get_or_create_stream_state_locked(item->stream);
		if (st) {
			if (st->pending)
				st->pending--;
			st->inflight++;
		}
		pthread_mutex_unlock(&g_lithos.mu);

		// Fetch launch-specific allocation just before dispatch and apply to next launch.
		if (st) {
			if (g_lithos.daemon_enabled) {
				(void)daemon_alloc_launch(item, &has_partition, &disable_mask);
			} else if (st->has_partition) {
				has_partition = true;
				disable_mask = st->disable_mask;
			}
		}
		if (has_partition)
			libsmctrl_set_next_mask(disable_mask);

		// Deferred launches are replayed via packed argument form.
		void* launch_extra[] = {
			CU_LAUNCH_PARAM_BUFFER_POINTER,
			item->arg_buffer,
			CU_LAUNCH_PARAM_BUFFER_SIZE,
			&item->arg_buffer_size,
			CU_LAUNCH_PARAM_END,
		};
		CUresult res = direct_launch(item->f,
		                           item->gx,
		                           item->gy,
		                           item->gz,
		                           item->bx,
		                           item->by,
		                           item->bz,
		                           item->shared_mem,
		                           item->stream,
		                           NULL,
		                           launch_extra);
		if (res == CUDA_SUCCESS && g_lithos.daemon_enabled && g_lithos.real_cuEventCreate &&
		    g_lithos.real_cuEventRecord && g_lithos.real_cuEventQuery && g_lithos.real_cuEventDestroy) {
			if (g_lithos.real_cuEventCreate(&done_event, CU_EVENT_DISABLE_TIMING) == CUDA_SUCCESS) {
				if (g_lithos.real_cuEventRecord(done_event, item->stream) == CUDA_SUCCESS)
					tracked_with_event = true;
				else
					(void)g_lithos.real_cuEventDestroy(done_event);
			}
		}
		if (g_lithos.daemon_enabled && res != CUDA_SUCCESS)
			daemon_free_launch(item->launch_id);

		pthread_mutex_lock(&g_lithos.mu);
		st = find_stream_state_locked(item->stream);
		if (st && st->inflight)
			st->inflight--;
		if (res == CUDA_SUCCESS && g_lithos.daemon_enabled)
			add_active_launch_locked(item->launch_id,
			                       item->stream,
			                       done_event,
			                       tracked_with_event,
			                       has_partition,
			                       disable_mask);
		if (res != CUDA_SUCCESS && g_lithos.dispatch_error == CUDA_SUCCESS)
			g_lithos.dispatch_error = res;
		pthread_cond_broadcast(&g_lithos.cv);
		pthread_mutex_unlock(&g_lithos.mu);

		free(item->arg_buffer);
		free(item);
	}
	return NULL;
}

/*** Runtime Initialization and Shutdown ***/

static void init_real_cuda_once(void) {
	g_lithos.initialized = true;
	g_lithos.enabled = false;
	g_lithos.real_ready = false;

	g_lithos.real_cuda = dlopen("libcuda.so", RTLD_LAZY | RTLD_LOCAL);
	if (!g_lithos.real_cuda) {
		fprintf(stderr, "libsmctrl-lithos: unable to open real libcuda.so: %s\n", dlerror());
		return;
	}

	g_lithos.real_cuStreamCreate = resolve_real_symbol("cuStreamCreate");
	g_lithos.real_cuStreamCreateWithPriority = resolve_real_symbol("cuStreamCreateWithPriority");
	g_lithos.real_cuStreamDestroy = resolve_real_symbol("cuStreamDestroy");
	g_lithos.real_cuLaunchKernel = resolve_real_symbol("cuLaunchKernel");
	g_lithos.real_cuFuncGetParamInfo = dlsym(g_lithos.real_cuda, "cuFuncGetParamInfo");
	g_lithos.real_cuEventCreate = dlsym(g_lithos.real_cuda, "cuEventCreate");
	g_lithos.real_cuEventRecord = dlsym(g_lithos.real_cuda, "cuEventRecord");
	g_lithos.real_cuEventQuery = dlsym(g_lithos.real_cuda, "cuEventQuery");
	g_lithos.real_cuEventDestroy = dlsym(g_lithos.real_cuda, "cuEventDestroy");
	g_lithos.real_cuStreamSynchronize = resolve_real_symbol("cuStreamSynchronize");
	g_lithos.real_cuCtxSynchronize = resolve_real_symbol("cuCtxSynchronize");
	g_lithos.real_cuStreamQuery = resolve_real_symbol("cuStreamQuery");
	g_lithos.real_cuStreamIsCapturing = dlsym(g_lithos.real_cuda, "cuStreamIsCapturing");
	if (!g_lithos.real_cuStreamCreate ||
	    !g_lithos.real_cuStreamCreateWithPriority ||
	    !g_lithos.real_cuStreamDestroy ||
	    !g_lithos.real_cuLaunchKernel ||
	    !g_lithos.real_cuStreamSynchronize ||
	    !g_lithos.real_cuCtxSynchronize ||
	    !g_lithos.real_cuStreamQuery)
		return;
	g_lithos.real_ready = true;

	// Interposition path is opt-in by environment variable.
	const char* en = getenv("LIBSMCTRL_LITHOS_ENABLE");
	if (!en || strcmp(en, "1") != 0)
		return;
	g_lithos.enabled = true;
	configure_scheduler_from_env();

	if (pthread_create(&g_lithos.dispatcher, NULL, dispatcher_main, NULL) != 0) {
		fprintf(stderr, "libsmctrl-lithos: unable to start dispatcher thread (%s).\n", strerror(errno));
		g_lithos.enabled = false;
		return;
	}

	fprintf(stderr, "libsmctrl-lithos: launch queue runtime enabled.\n");
}

void lithos_wrapper_init(void) {
	pthread_once(&g_lithos.once, init_real_cuda_once);
}

void lithos_wrapper_shutdown(void) {
	if (!g_lithos.initialized || !g_lithos.enabled)
		return;
	pthread_mutex_lock(&g_lithos.mu);
	g_lithos.shutting_down = true;
	pthread_cond_broadcast(&g_lithos.cv);
	pthread_mutex_unlock(&g_lithos.mu);
	pthread_join(g_lithos.dispatcher, NULL);
	pthread_mutex_lock(&g_lithos.mu);
	reap_completed_launches_locked(NULL, false, true);
	pthread_mutex_unlock(&g_lithos.mu);
	g_lithos.enabled = false;
}

__attribute__((destructor)) static void lithos_shutdown_destructor(void) {
	lithos_wrapper_shutdown();
}

/*** Synchronization Helpers ***/

static CUresult wait_stream_queue_drained(CUstream stream) {
	pthread_mutex_lock(&g_lithos.mu);
	struct stream_state* st = get_or_create_stream_state_locked(stream);
	while (st && (st->pending || st->inflight))
		pthread_cond_wait(&g_lithos.cv, &g_lithos.mu);
	reap_completed_launches_locked(stream, true, false);
	CUresult err = g_lithos.dispatch_error;
	if (err != CUDA_SUCCESS)
		g_lithos.dispatch_error = CUDA_SUCCESS;
	pthread_mutex_unlock(&g_lithos.mu);
	return err;
}

static CUresult wait_all_queues_drained(void) {
	pthread_mutex_lock(&g_lithos.mu);
	while (any_pending_or_inflight_locked())
		pthread_cond_wait(&g_lithos.cv, &g_lithos.mu);
	reap_completed_launches_locked(NULL, false, false);
	CUresult err = g_lithos.dispatch_error;
	if (err != CUDA_SUCCESS)
		g_lithos.dispatch_error = CUDA_SUCCESS;
	pthread_mutex_unlock(&g_lithos.mu);
	return err;
}

/*** Interposed CUDA Driver API Functions ***/

CUresult cuStreamCreate(CUstream* phStream, unsigned int flags) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuStreamCreate)
		return CUDA_ERROR_NOT_INITIALIZED;
	CUresult res = g_lithos.real_cuStreamCreate(phStream, flags);
	if (res != CUDA_SUCCESS)
		return res;
	if (!g_lithos.enabled)
		return res;
	pthread_mutex_lock(&g_lithos.mu);
	if (!get_or_create_stream_state_locked(*phStream))
		res = CUDA_ERROR_OUT_OF_MEMORY;
	pthread_mutex_unlock(&g_lithos.mu);
	return res;
}

CUresult cuStreamCreateWithPriority(CUstream* phStream, unsigned int flags, int priority) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuStreamCreateWithPriority)
		return CUDA_ERROR_NOT_INITIALIZED;
	CUresult res = g_lithos.real_cuStreamCreateWithPriority(phStream, flags, priority);
	if (res != CUDA_SUCCESS)
		return res;
	if (!g_lithos.enabled)
		return res;
	pthread_mutex_lock(&g_lithos.mu);
	if (!get_or_create_stream_state_locked(*phStream))
		res = CUDA_ERROR_OUT_OF_MEMORY;
	pthread_mutex_unlock(&g_lithos.mu);
	return res;
}

CUresult cuStreamDestroy(CUstream hStream) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuStreamDestroy)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!g_lithos.enabled)
		return g_lithos.real_cuStreamDestroy(hStream);
	CUresult err = wait_stream_queue_drained(hStream);
	if (err != CUDA_SUCCESS)
		return err;
	CUresult res = g_lithos.real_cuStreamDestroy(hStream);
	pthread_mutex_lock(&g_lithos.mu);
	reap_completed_launches_locked(hStream, true, true);
	struct stream_state** pp = &g_lithos.streams;
	while (*pp) {
		if ((*pp)->stream == hStream) {
			struct stream_state* dead = *pp;
			*pp = dead->next;
			free(dead);
			break;
		}
		pp = &(*pp)->next;
	}
	pthread_mutex_unlock(&g_lithos.mu);
	return res;
}

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gx,
                        unsigned int gy,
                        unsigned int gz,
                        unsigned int bx,
                        unsigned int by,
                        unsigned int bz,
                        unsigned int sharedMemBytes,
                        CUstream hStream,
                        void** kernelParams,
                        void** extra) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuLaunchKernel)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!g_lithos.enabled)
		return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
	if (g_lithos.capture_passthrough_mode)
		return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);

	// CUDA Graph capture requires launch ordering/ownership on the capturing
	// stream thread. Once observed, force direct launches for process lifetime to
	// avoid mixing deferred and capture-sensitive execution paths.

	// TODO: Handle CUDA Graph Capture without bypassing the scheduler.
	if (stream_is_capturing(hStream)) {
		g_lithos.capture_passthrough_mode = true;
		if (!g_lithos.capture_bypass_warned) {
			fprintf(stderr,
			        "libsmctrl-lithos: CUDA Graph capture detected; switching to process-wide direct launch passthrough.\n");
			g_lithos.capture_bypass_warned = true;
		}
		return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
	}

	void* arg_buf = NULL;
	size_t arg_buf_sz = 0;
	bool arg_buf_owned = false;

	// queue only encodes packed launch arguments; kernelParams are converted.
	if (kernelParams && !extra) {
		if (!pack_kernel_params(f, kernelParams, &arg_buf, &arg_buf_sz))
			return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
		arg_buf_owned = true;
	} else if (!kernelParams && extra) {
		if (!parse_packed_args(extra, &arg_buf, &arg_buf_sz))
			return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
	} else {
		return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
	}

	struct launch_item* item = calloc(1, sizeof(*item));
	if (!item)
		return CUDA_ERROR_OUT_OF_MEMORY;
	if (arg_buf_owned) {
		item->arg_buffer = arg_buf;
	} else {
		item->arg_buffer = malloc(arg_buf_sz);
		if (!item->arg_buffer) {
			free(item);
			return CUDA_ERROR_OUT_OF_MEMORY;
		}
		memcpy(item->arg_buffer, arg_buf, arg_buf_sz);
	}
	item->arg_buffer_size = arg_buf_sz;
	item->f = f;
	item->gx = gx;
	item->gy = gy;
	item->gz = gz;
	item->bx = bx;
	item->by = by;
	item->bz = bz;
	item->shared_mem = sharedMemBytes;
	item->stream = hStream;

	pthread_mutex_lock(&g_lithos.mu);
	struct stream_state* st = get_or_create_stream_state_locked(hStream);
	if (!st) {
		pthread_mutex_unlock(&g_lithos.mu);
		free(item->arg_buffer);
		free(item);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	item->stream_id = st->stream_id;
	item->launch_id = g_lithos.next_launch_id++;
	st->pending++;
	if (g_lithos.q_tail)
		g_lithos.q_tail->next = item;
	else
		g_lithos.q_head = item;
	g_lithos.q_tail = item;
	pthread_cond_signal(&g_lithos.cv);
	pthread_mutex_unlock(&g_lithos.mu);
	return CUDA_SUCCESS;
}

CUresult cuStreamSynchronize(CUstream hStream) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuStreamSynchronize)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!g_lithos.enabled)
		return g_lithos.real_cuStreamSynchronize(hStream);
	CUresult err = wait_stream_queue_drained(hStream);
	if (err != CUDA_SUCCESS)
		return err;
	err = g_lithos.real_cuStreamSynchronize(hStream);
	if (err != CUDA_SUCCESS)
		return err;
	pthread_mutex_lock(&g_lithos.mu);
	reap_completed_launches_locked(hStream, true, true);
	pthread_mutex_unlock(&g_lithos.mu);
	return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize(void) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuCtxSynchronize)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!g_lithos.enabled)
		return g_lithos.real_cuCtxSynchronize();
	CUresult err = wait_all_queues_drained();
	if (err != CUDA_SUCCESS)
		return err;
	err = g_lithos.real_cuCtxSynchronize();
	if (err != CUDA_SUCCESS)
		return err;
	pthread_mutex_lock(&g_lithos.mu);
	reap_completed_launches_locked(NULL, false, true);
	pthread_mutex_unlock(&g_lithos.mu);
	return CUDA_SUCCESS;
}

CUresult cuStreamQuery(CUstream hStream) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuStreamQuery)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!g_lithos.enabled)
		return g_lithos.real_cuStreamQuery(hStream);
	pthread_mutex_lock(&g_lithos.mu);
	struct stream_state* st = get_or_create_stream_state_locked(hStream);
	reap_completed_launches_locked(hStream, true, false);
	bool pending = st && (st->pending || st->inflight || has_active_launches_for_stream_locked(hStream));
	pthread_mutex_unlock(&g_lithos.mu);
	if (pending)
		return CUDA_ERROR_NOT_READY;
	return g_lithos.real_cuStreamQuery(hStream);
}

// #endif
