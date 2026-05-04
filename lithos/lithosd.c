/*
 * Minimal global LithOS scheduler daemon.
 *
 * This service centralizes TPC partition assignment across wrapper clients.
 * Current policy is static first-fit assignment with optional release.
 */
#define _GNU_SOURCE

#include <errno.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "lithos_ipc.h"

enum { MAX_ALLOCS = 4096 };

struct alloc_entry {
	bool in_use;
	uint64_t pid;
	uint32_t launch_id;
	uint32_t stream_id;
	uint32_t quota;
	uint32_t start_tpc;
	uint64_t disable_mask;
};

static struct {
	int server_fd;
	char sock_path[108];
	uint32_t num_tpcs;
	uint64_t used_tpcs;
	struct alloc_entry allocs[MAX_ALLOCS];
	volatile sig_atomic_t stop;
} g_daemon;

struct cli_args {
	bool status_mode;
	char sock_path[108];
	uint32_t num_tpcs;
};

static bool lithosd_is_mps_running(void) {
	const char* mps_pipe_dir = getenv("CUDA_MPS_PIPE_DIRECTORY");
	struct sockaddr_un mps_ctrl_addr;
	int mps_ctrl;
	const int yes = 1;

	if (!mps_pipe_dir)
		mps_pipe_dir = "/tmp/nvidia-mps";
	memset(&mps_ctrl_addr, 0, sizeof(mps_ctrl_addr));
	mps_ctrl_addr.sun_family = AF_UNIX;
	snprintf(mps_ctrl_addr.sun_path, sizeof(mps_ctrl_addr.sun_path), "%s/control", mps_pipe_dir);

	if ((mps_ctrl = socket(AF_UNIX, SOCK_SEQPACKET, 0)) == -1)
		return false;
	if (setsockopt(mps_ctrl, SOL_SOCKET, SO_PASSCRED, &yes, sizeof(yes)) == -1) {
		close(mps_ctrl);
		return false;
	}
	if (connect(mps_ctrl, (struct sockaddr*)&mps_ctrl_addr, sizeof(mps_ctrl_addr)) == -1) {
		close(mps_ctrl);
		return false;
	}
	close(mps_ctrl);
	return true;
}

static void lithosd_ensure_mps_running(void) {
	const char* mps_autostart = getenv("LIBSMCTRL_LITHOS_MPS_AUTOSTART");
	const char* mps_visible = getenv("LIBSMCTRL_LITHOS_MPS_VISIBLE_DEVICES");
	int ret;

	if (mps_autostart && strcmp(mps_autostart, "0") == 0)
		return;

	if (mps_visible && *mps_visible && !getenv("CUDA_VISIBLE_DEVICES")) {
		if (setenv("CUDA_VISIBLE_DEVICES", mps_visible, 0) == -1) {
			fprintf(stderr, "lithosd: warning: unable to set CUDA_VISIBLE_DEVICES (%s)\n", strerror(errno));
		}
	}

	if (!getenv("CUDA_MPS_PIPE_DIRECTORY")) {
		if (setenv("CUDA_MPS_PIPE_DIRECTORY", "/tmp/nvidia-mps", 0) == -1) {
			fprintf(stderr, "lithosd: warning: unable to set CUDA_MPS_PIPE_DIRECTORY (%s)\n", strerror(errno));
		}
	}

	if (lithosd_is_mps_running())
		return;

	fprintf(stderr, "lithosd: MPS control daemon not detected; attempting automatic start.\n");
	ret = system("nvidia-cuda-mps-control -d");

	if (ret == 0x7f00) {
		const char* old_path = getenv("PATH");
		char* new_path = NULL;
		if (old_path)
			(void)asprintf(&new_path, "/usr/local/cuda/compat/:%s", old_path);
		else
			(void)asprintf(&new_path, "/usr/local/cuda/compat/");
		if (new_path) {
			if (setenv("PATH", new_path, 1) == -1)
				fprintf(stderr, "lithosd: warning: unable to update PATH for MPS fallback (%s)\n", strerror(errno));
			free(new_path);
		}
		ret = system("nvidia-cuda-mps-control -d");
	}

	if (ret == -1) {
		fprintf(stderr, "lithosd: warning: failed to run shell while starting MPS (%s)\n", strerror(errno));
		return;
	}
	if (WIFEXITED(ret) && WEXITSTATUS(ret) == 0) {
		if (lithosd_is_mps_running()) {
			fprintf(stderr, "lithosd: MPS control daemon started automatically.\n");
			return;
		}
		fprintf(stderr, "lithosd: warning: MPS start command returned success, but control socket is not reachable yet.\n");
		return;
	}
	fprintf(stderr, "lithosd: warning: automatic MPS start failed (status=%d); concurrent execution may time-slice.\n", ret);
}

static void on_signal(int sig) {
	(void)sig;
	g_daemon.stop = 1;
	if (g_daemon.server_fd != -1) {
		close(g_daemon.server_fd);
		g_daemon.server_fd = -1;
	}
}

static uint64_t valid_tpc_bits_mask(void) {
	if (g_daemon.num_tpcs >= 64)
		return ~0ull;
	if (g_daemon.num_tpcs == 0)
		return 0;
	return (1ull << g_daemon.num_tpcs) - 1ull;
}

static struct alloc_entry* find_alloc(uint64_t pid, uint32_t launch_id) {
	for (int i = 0; i < MAX_ALLOCS; i++) {
		if (g_daemon.allocs[i].in_use &&
		    g_daemon.allocs[i].pid == pid &&
		    g_daemon.allocs[i].launch_id == launch_id)
			return &g_daemon.allocs[i];
	}
	return NULL;
}

static struct alloc_entry* alloc_slot(void) {
	for (int i = 0; i < MAX_ALLOCS; i++) {
		if (!g_daemon.allocs[i].in_use)
			return &g_daemon.allocs[i];
	}
	return NULL;
}

static uint32_t active_alloc_count(void) {
	uint32_t n = 0;
	for (int i = 0; i < MAX_ALLOCS; i++) {
		if (g_daemon.allocs[i].in_use)
			n++;
	}
	return n;
}

static uint32_t count_bits64(uint64_t x) {
	uint32_t n = 0;
	while (x) {
		n += (uint32_t)(x & 1ull);
		x >>= 1;
	}
	return n;
}

static bool try_allocate_contiguous(uint32_t quota, uint32_t* start_out, uint64_t* disable_mask_out) {
	uint64_t valid_mask = valid_tpc_bits_mask();
	if (quota == 0 || quota > g_daemon.num_tpcs)
		return false;
	for (uint32_t start = 0; start + quota <= g_daemon.num_tpcs; start++) {
		uint64_t req_mask;
		if (quota >= 64)
			req_mask = ~0ull;
		else
			req_mask = ((1ull << quota) - 1ull) << start;
		req_mask &= valid_mask;
		if ((g_daemon.used_tpcs & req_mask) == 0) {
			g_daemon.used_tpcs |= req_mask;
			*start_out = start;
			*disable_mask_out = (~req_mask) & valid_mask;
			return true;
		}
	}
	return false;
}

static void handle_alloc(const struct lithosd_req* req, struct lithosd_resp* resp) {
	uint32_t quota = req->quota;
	struct alloc_entry* e;
	uint32_t start_tpc = 0;
	uint64_t disable_mask = 0;

	resp->status = 0;
	resp->has_partition = 0;
	resp->disable_mask = 0;

	if (quota == 0)
		return;

	e = find_alloc(req->pid, req->launch_id);
	if (e) {
		resp->has_partition = 1;
		resp->disable_mask = e->disable_mask;
		return;
	}

	// If there is no contiguous space left, caller falls back to unpartitioned.
	if (!try_allocate_contiguous(quota, &start_tpc, &disable_mask))
		return;

	e = alloc_slot();
	if (!e) {
		uint64_t release_mask;
		if (quota >= 64)
			release_mask = ~0ull;
		else
			release_mask = ((1ull << quota) - 1ull) << start_tpc;
		release_mask &= valid_tpc_bits_mask();
		g_daemon.used_tpcs &= ~release_mask;
		resp->status = -ENOSPC;
		return;
	}

	e->in_use = true;
	e->pid = req->pid;
	e->launch_id = req->launch_id;
	e->stream_id = req->stream_id;
	e->quota = quota;
	e->start_tpc = start_tpc;
	e->disable_mask = disable_mask;

	resp->has_partition = 1;
	resp->disable_mask = e->disable_mask;
}

static void handle_free(const struct lithosd_req* req, struct lithosd_resp* resp) {
	struct alloc_entry* e = find_alloc(req->pid, req->launch_id);
	resp->status = 0;
	resp->has_partition = 0;
	resp->disable_mask = 0;
	if (e) {
		uint64_t release_mask;
		if (e->quota >= 64)
			release_mask = ~0ull;
		else
			release_mask = ((1ull << e->quota) - 1ull) << e->start_tpc;
		release_mask &= valid_tpc_bits_mask();
		g_daemon.used_tpcs &= ~release_mask;
		e->in_use = false;
	}
	// Report post-operation allocation count for lifecycle assertions in tests.
	resp->active_allocs = active_alloc_count();
}

static void handle_stats(struct lithosd_resp* resp) {
	resp->status = 0;
	resp->has_partition = 0;
	resp->disable_mask = 0;
	resp->active_allocs = active_alloc_count();
	resp->num_tpcs = g_daemon.num_tpcs;
	resp->used_tpcs = g_daemon.used_tpcs & valid_tpc_bits_mask();
}

static void handle_status(const struct lithosd_req* req, struct lithosd_resp* resp) {
	uint32_t idx = req->reserved;

	resp->status = 0;
	resp->has_partition = 0;
	resp->disable_mask = 0;
	resp->active_allocs = active_alloc_count();
	resp->num_tpcs = g_daemon.num_tpcs;
	resp->used_tpcs = g_daemon.used_tpcs & valid_tpc_bits_mask();
	resp->alloc_in_use = 0;

	if (idx >= MAX_ALLOCS) {
		resp->status = -EINVAL;
		return;
	}
	if (!g_daemon.allocs[idx].in_use)
		return;

	resp->alloc_in_use = 1;
	resp->alloc_pid = g_daemon.allocs[idx].pid;
	resp->alloc_launch_id = g_daemon.allocs[idx].launch_id;
	resp->alloc_stream_id = g_daemon.allocs[idx].stream_id;
	resp->alloc_quota = g_daemon.allocs[idx].quota;
	resp->alloc_start_tpc = g_daemon.allocs[idx].start_tpc;
	resp->alloc_disable_mask = g_daemon.allocs[idx].disable_mask;
}

static bool rpc_to_daemon(const char* sock_path, const struct lithosd_req* req, struct lithosd_resp* resp) {
	int fd;
	struct sockaddr_un addr;
	ssize_t n;

	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd == -1)
		return false;

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
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

static int run_status_command(const char* sock_path) {
	struct lithosd_req req;
	struct lithosd_resp resp;
	uint32_t used_count;

	memset(&req, 0, sizeof(req));
	memset(&resp, 0, sizeof(resp));
	req.op = LITHOSD_OP_GET_STATS;
	if (!rpc_to_daemon(sock_path, &req, &resp)) {
		fprintf(stderr, "lithosd: failed to query daemon at %s\n", sock_path);
		return 2;
	}
	if (resp.status != 0) {
		fprintf(stderr, "lithosd: daemon stats request failed (status=%d)\n", resp.status);
		return 2;
	}

	used_count = count_bits64(resp.used_tpcs);
	printf("lithosd status: socket=%s num_tpcs=%u used_tpcs=%u active_allocs=%u\n",
	       sock_path,
	       resp.num_tpcs,
	       used_count,
	       resp.active_allocs);
	printf("slot   pid      launch   stream   quota   start_tpc   disable_mask\n");

	for (uint32_t i = 0; i < MAX_ALLOCS; i++) {
		memset(&req, 0, sizeof(req));
		memset(&resp, 0, sizeof(resp));
		req.op = LITHOSD_OP_GET_STATUS;
		req.reserved = i;
		if (!rpc_to_daemon(sock_path, &req, &resp)) {
			fprintf(stderr, "lithosd: status query failed at slot %u\n", i);
			return 2;
		}
		if (resp.status != 0) {
			fprintf(stderr, "lithosd: status query error at slot %u (status=%d)\n", i, resp.status);
			return 2;
		}
		if (!resp.alloc_in_use)
			continue;
		printf("%-6u %-8llu %-8u %-8u %-7u %-11u 0x%016llx\n",
		       i,
		       (unsigned long long)resp.alloc_pid,
		       resp.alloc_launch_id,
		       resp.alloc_stream_id,
		       resp.alloc_quota,
		       resp.alloc_start_tpc,
		       (unsigned long long)resp.alloc_disable_mask);
	}

	return 0;
}

static int parse_args(int argc, char** argv, struct cli_args* out) {
	if (!out)
		return -1;
	memset(out, 0, sizeof(*out));

	if (argc == 3 && strcmp(argv[1], "--status") == 0) {
		if (strlen(argv[2]) >= sizeof(out->sock_path)) {
			fprintf(stderr, "Socket path too long: %s\n", argv[2]);
			return -1;
		}
		out->status_mode = true;
		strncpy(out->sock_path, argv[2], sizeof(out->sock_path) - 1);
		return 0;
	}

	if (argc != 3) {
		fprintf(stderr, "Usage: %s <socket_path> <num_tpcs>\n", argv[0]);
		fprintf(stderr, "       %s --status <socket_path>\n", argv[0]);
		return -1;
	}
	if (strlen(argv[1]) >= sizeof(out->sock_path)) {
		fprintf(stderr, "Socket path too long: %s\n", argv[1]);
		return -1;
	}
	strncpy(out->sock_path, argv[1], sizeof(out->sock_path) - 1);
	out->num_tpcs = (uint32_t)strtoul(argv[2], NULL, 10);
	if (out->num_tpcs == 0 || out->num_tpcs > 64) {
		fprintf(stderr, "num_tpcs must be in [1, 64]\n");
		return -1;
	}
	return 0;
}

int main(int argc, char** argv) {
	struct sockaddr_un addr;
	struct cli_args args;
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	memset(&g_daemon, 0, sizeof(g_daemon));
	g_daemon.server_fd = -1;

	if (parse_args(argc, argv, &args) != 0)
		return 2;
	if (args.status_mode)
		return run_status_command(args.sock_path);
	lithosd_ensure_mps_running();

	strncpy(g_daemon.sock_path, args.sock_path, sizeof(g_daemon.sock_path) - 1);
	g_daemon.num_tpcs = args.num_tpcs;

	if ((g_daemon.server_fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
		perror("socket");
		return 2;
	}

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, g_daemon.sock_path, sizeof(addr.sun_path) - 1);
	unlink(g_daemon.sock_path);
	if (bind(g_daemon.server_fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
		perror("bind");
		close(g_daemon.server_fd);
		return 2;
	}
	if (listen(g_daemon.server_fd, 64) == -1) {
		perror("listen");
		close(g_daemon.server_fd);
		unlink(g_daemon.sock_path);
		return 2;
	}

	fprintf(stderr, "lithosd: listening on %s with %u TPCs\n", g_daemon.sock_path, g_daemon.num_tpcs);

	while (!g_daemon.stop) {
		int cfd;
		struct lithosd_req req;
		struct lithosd_resp resp;
		ssize_t n;

		cfd = accept(g_daemon.server_fd, NULL, NULL);
		if (cfd == -1) {
			if (errno == EINTR)
				continue;
			if (g_daemon.stop)
				break;
			perror("accept");
			break;
		}

		n = read(cfd, &req, sizeof(req));
		if (n == (ssize_t)sizeof(req)) {
			memset(&resp, 0, sizeof(resp));
			switch (req.op) {
				case LITHOSD_OP_ALLOC_LAUNCH:
					handle_alloc(&req, &resp);
					break;
				case LITHOSD_OP_FREE_LAUNCH:
					handle_free(&req, &resp);
					break;
				case LITHOSD_OP_GET_STATS:
					handle_stats(&resp);
					break;
				case LITHOSD_OP_GET_STATUS:
					handle_status(&req, &resp);
					break;
				default:
					resp.status = -EINVAL;
					break;
			}
			(void)write(cfd, &resp, sizeof(resp));
		}
		close(cfd);
	}

	if (g_daemon.server_fd != -1)
		close(g_daemon.server_fd);
	unlink(g_daemon.sock_path);
	return 0;
}
