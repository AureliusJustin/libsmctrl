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
#include <unistd.h>

#include "lithos_ipc.h"

enum { MAX_ALLOCS = 4096 };

struct alloc_entry {
	bool in_use;
	uint64_t pid;
	uint32_t stream_id;
	uint32_t quota;
	uint32_t start_tpc;
	uint64_t disable_mask;
};

static struct {
	int server_fd;
	char sock_path[108];
	uint32_t num_tpcs;
	uint32_t next_tpc_cursor;
	struct alloc_entry allocs[MAX_ALLOCS];
	volatile sig_atomic_t stop;
} g_daemon;

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

static struct alloc_entry* find_alloc(uint64_t pid, uint32_t stream_id) {
	for (int i = 0; i < MAX_ALLOCS; i++) {
		if (g_daemon.allocs[i].in_use &&
		    g_daemon.allocs[i].pid == pid &&
		    g_daemon.allocs[i].stream_id == stream_id)
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

static void handle_alloc(const struct lithosd_req* req, struct lithosd_resp* resp) {
	uint32_t quota = req->quota;
	uint64_t enable_mask;
	uint64_t valid_mask;
	struct alloc_entry* e;

	resp->status = 0;
	resp->has_partition = 0;
	resp->disable_mask = 0;

	if (quota == 0)
		return;

	e = find_alloc(req->pid, req->stream_id);
	if (e) {
		resp->has_partition = 1;
		resp->disable_mask = e->disable_mask;
		return;
	}

	// If there is no contiguous space left, caller falls back to unpartitioned.
	if (quota > g_daemon.num_tpcs || g_daemon.next_tpc_cursor + quota > g_daemon.num_tpcs)
		return;

	e = alloc_slot();
	if (!e) {
		resp->status = -ENOSPC;
		return;
	}

	if (quota >= 64)
		enable_mask = ~0ull;
	else
		enable_mask = ((1ull << quota) - 1ull) << g_daemon.next_tpc_cursor;
	valid_mask = valid_tpc_bits_mask();
	enable_mask &= valid_mask;

	e->in_use = true;
	e->pid = req->pid;
	e->stream_id = req->stream_id;
	e->quota = quota;
	e->start_tpc = g_daemon.next_tpc_cursor;
	e->disable_mask = (~enable_mask) & valid_mask;
	g_daemon.next_tpc_cursor += quota;

	resp->has_partition = 1;
	resp->disable_mask = e->disable_mask;
}

static void handle_free(const struct lithosd_req* req, struct lithosd_resp* resp) {
	struct alloc_entry* e = find_alloc(req->pid, req->stream_id);
	resp->status = 0;
	resp->has_partition = 0;
	resp->disable_mask = 0;
	if (e)
		e->in_use = false;
	// Report post-operation allocation count for lifecycle assertions in tests.
	resp->active_allocs = active_alloc_count();
}

static void handle_stats(struct lithosd_resp* resp) {
	resp->status = 0;
	resp->has_partition = 0;
	resp->disable_mask = 0;
	resp->active_allocs = active_alloc_count();
}

static int parse_args(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "Usage: %s <socket_path> <num_tpcs>\n", argv[0]);
		return -1;
	}
	if (strlen(argv[1]) >= sizeof(g_daemon.sock_path)) {
		fprintf(stderr, "Socket path too long: %s\n", argv[1]);
		return -1;
	}
	strncpy(g_daemon.sock_path, argv[1], sizeof(g_daemon.sock_path) - 1);
	g_daemon.num_tpcs = (uint32_t)strtoul(argv[2], NULL, 10);
	if (g_daemon.num_tpcs == 0 || g_daemon.num_tpcs > 64) {
		fprintf(stderr, "num_tpcs must be in [1, 64]\n");
		return -1;
	}
	return 0;
}

int main(int argc, char** argv) {
	struct sockaddr_un addr;
	signal(SIGINT, on_signal);
	signal(SIGTERM, on_signal);
	memset(&g_daemon, 0, sizeof(g_daemon));
	g_daemon.server_fd = -1;

	if (parse_args(argc, argv) != 0)
		return 2;

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
				case LITHOSD_OP_ALLOC_STREAM:
					handle_alloc(&req, &resp);
					break;
				case LITHOSD_OP_FREE_STREAM:
					handle_free(&req, &resp);
					break;
				case LITHOSD_OP_GET_STATS:
					handle_stats(&resp);
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
