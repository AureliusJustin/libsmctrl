/*
 * Copyright 2026 Justin and contributors
 *
 * LithOS runtime prototype layered on top of libsmctrl's libcuda.so.1 wrapper.
 *
 * Current implementation scope:
 * - Phase 1: Driver API interposition for launch/stream/sync entrypoints.
 * - Phase 2: Deferred launch queues for packed launch arguments.
 * - Phase 3: Baseline static per-stream TPC quota assignment.
 *
 * This intentionally keeps policy simple and deterministic so behavior can be
 * validated before adding dynamic stealing and prediction logic.
 */
#include "lithos_runtime.h"

#ifdef LIBSMCTRL_WRAPPER

#include "libsmctrl.h"

#include <cuda.h>

#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
	CUfunction f;
	unsigned int gx, gy, gz;
	unsigned int bx, by, bz;
	unsigned int shared_mem;
	CUstream stream;
	void* arg_buffer;
	size_t arg_buffer_size;
	struct launch_item* next;
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
	CUresult (*real_cuStreamSynchronize)(CUstream);
	CUresult (*real_cuCtxSynchronize)(void);
	CUresult (*real_cuStreamQuery)(CUstream);

	pthread_t dispatcher;
	pthread_mutex_t mu;
	pthread_cond_t cv;
	struct launch_item* q_head;
	struct launch_item* q_tail;
	struct stream_state* streams;
	CUresult dispatch_error;

	bool scheduler_enabled;
	uint32_t num_tpcs;
	uint32_t default_quota;
	uint32_t next_tpc_cursor;
	uint32_t next_stream_id;
	bool oversubscribe_warned;
	uint32_t quota_list[128];
	uint32_t quota_count;
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
	.real_cuStreamSynchronize = NULL,
	.real_cuCtxSynchronize = NULL,
	.real_cuStreamQuery = NULL,
	.dispatcher = 0,
	.mu = PTHREAD_MUTEX_INITIALIZER,
	.cv = PTHREAD_COND_INITIALIZER,
	.q_head = NULL,
	.q_tail = NULL,
	.streams = NULL,
	.dispatch_error = CUDA_SUCCESS,
	.scheduler_enabled = false,
	.num_tpcs = 0,
	.default_quota = 0,
	.next_tpc_cursor = 0,
	.next_stream_id = 0,
	.oversubscribe_warned = false,
	.quota_list = {0},
	.quota_count = 0,
};

/*** Phase 3 Baseline Scheduler Helpers ***/

static uint64_t valid_tpc_bits_mask(void) {
	if (g_lithos.num_tpcs >= 64)
		return ~0ull;
	if (g_lithos.num_tpcs == 0)
		return 0;
	return (1ull << g_lithos.num_tpcs) - 1ull;
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
	const char* en = getenv("LIBSMCTRL_LITHOS_SCHED_ENABLE");
	const char* q_default = getenv("LIBSMCTRL_LITHOS_TPC_QUOTA_DEFAULT");
	const char* q_list = getenv("LIBSMCTRL_LITHOS_TPC_QUOTAS");
	uint32_t tpcs = 0;
	int rc;

	// Scheduler policy is opt-in so Phase 1/2 behavior remains default.
	g_lithos.scheduler_enabled = false;
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

	if (q_default && *q_default)
		g_lithos.default_quota = (uint32_t)strtoul(q_default, NULL, 10);
	else
		g_lithos.default_quota = 0;

	if (!parse_quota_list(q_list, g_lithos.quota_list, &g_lithos.quota_count)) {
		fprintf(stderr, "libsmctrl-lithos: scheduler disabled (invalid LIBSMCTRL_LITHOS_TPC_QUOTAS).\n");
		g_lithos.num_tpcs = 0;
		return;
	}

	g_lithos.scheduler_enabled = true;
	fprintf(stderr,
	        "libsmctrl-lithos: Phase 3 baseline scheduler enabled (%u TPCs, default quota=%u, explicit quotas=%u).\n",
	        g_lithos.num_tpcs, g_lithos.default_quota, g_lithos.quota_count);
}

static void assign_stream_partition_locked(struct stream_state* st) {
	uint32_t quota;
	uint64_t enable_mask;
	uint64_t valid_mask;
	if (!st || !g_lithos.scheduler_enabled)
		return;

	if (st->stream_id < g_lithos.quota_count)
		quota = g_lithos.quota_list[st->stream_id];
	else
		quota = g_lithos.default_quota;

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
	st->stream_id = g_lithos.next_stream_id++;
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

/*** Dispatcher Thread ***/

static void* dispatcher_main(void* arg) {
	(void)arg;
	while (1) {
		pthread_mutex_lock(&g_lithos.mu);
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

		// Apply the stream's assigned partition to this thread's next launch.
		if (st && st->has_partition)
			libsmctrl_set_next_mask(st->disable_mask);

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

		pthread_mutex_lock(&g_lithos.mu);
		st = get_or_create_stream_state_locked(item->stream);
		if (st && st->inflight)
			st->inflight--;
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
	g_lithos.real_cuStreamSynchronize = resolve_real_symbol("cuStreamSynchronize");
	g_lithos.real_cuCtxSynchronize = resolve_real_symbol("cuCtxSynchronize");
	g_lithos.real_cuStreamQuery = resolve_real_symbol("cuStreamQuery");
	if (!g_lithos.real_cuStreamCreate ||
	    !g_lithos.real_cuStreamCreateWithPriority ||
	    !g_lithos.real_cuStreamDestroy ||
	    !g_lithos.real_cuLaunchKernel ||
	    !g_lithos.real_cuStreamSynchronize ||
	    !g_lithos.real_cuCtxSynchronize ||
	    !g_lithos.real_cuStreamQuery)
		return;
	g_lithos.real_ready = true;

	// Phase 1/2/3 interposition path is opt-in by environment variable.
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

	fprintf(stderr, "libsmctrl-lithos: Phase 2 launch queue runtime enabled.\n");
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

	// Deferred queueing currently supports packed launch arguments only.
	// kernelParams launches are passed through directly.
	if (kernelParams) {
		return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
	}

	void* arg_buf = NULL;
	size_t arg_buf_sz = 0;
	if (!parse_packed_args(extra, &arg_buf, &arg_buf_sz)) {
		return direct_launch(f, gx, gy, gz, bx, by, bz, sharedMemBytes, hStream, kernelParams, extra);
	}

	struct launch_item* item = calloc(1, sizeof(*item));
	if (!item)
		return CUDA_ERROR_OUT_OF_MEMORY;
	item->arg_buffer = malloc(arg_buf_sz);
	if (!item->arg_buffer) {
		free(item);
		return CUDA_ERROR_OUT_OF_MEMORY;
	}
	memcpy(item->arg_buffer, arg_buf, arg_buf_sz);
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
	return g_lithos.real_cuStreamSynchronize(hStream);
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
	return g_lithos.real_cuCtxSynchronize();
}

CUresult cuStreamQuery(CUstream hStream) {
	lithos_wrapper_init();
	if (!g_lithos.real_ready || !g_lithos.real_cuStreamQuery)
		return CUDA_ERROR_NOT_INITIALIZED;
	if (!g_lithos.enabled)
		return g_lithos.real_cuStreamQuery(hStream);
	pthread_mutex_lock(&g_lithos.mu);
	struct stream_state* st = get_or_create_stream_state_locked(hStream);
	bool pending = st && (st->pending || st->inflight);
	pthread_mutex_unlock(&g_lithos.mu);
	if (pending)
		return CUDA_ERROR_NOT_READY;
	return g_lithos.real_cuStreamQuery(hStream);
}

#endif
