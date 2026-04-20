/*
 * Terminal-style LithOS global scheduler validation.
 *
 * Launches lithosd and two independent arbitrary apps from the terminal path
 * (exec), with interposition env vars set before process startup. Uses a
 * long-running kernel to force overlap so disjointness across concurrent apps
 * can be validated in launch-level scheduling mode.
 */
#define _GNU_SOURCE

#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include "libsmctrl.h"
#include "lithos_ipc.h"

struct app_result {
	int pid;
	int uniq;
	int sms_per_tpc;
	unsigned min_smid;
	unsigned max_smid;
};

struct alloc_snapshot {
	bool valid;
	uint64_t pid;
	uint32_t quota;
	uint32_t start_tpc;
	uint64_t disable_mask;
};

enum { LITHOS_STATUS_SCAN_LIMIT = 4096 };

static int wait_for_socket(const char* sock_path) {
	struct stat st;
	for (int i = 0; i < 300; i++) {
		if (stat(sock_path, &st) == 0)
			return 0;
		usleep(10000);
	}
	return -1;
}

static int parse_result_line(const char* line, struct app_result* out) {
	int n = sscanf(line,
	               "RESULT pid=%d uniq=%d sms_per_tpc=%d min=%u max=%u",
	               &out->pid,
	               &out->uniq,
	               &out->sms_per_tpc,
	               &out->min_smid,
	               &out->max_smid);
	return n == 5 ? 0 : -1;
}

static int query_active_allocs(const char* sock_path, uint32_t* out_active) {
	int fd;
	ssize_t n;
	struct sockaddr_un addr;
	struct lithosd_req req = {0};
	struct lithosd_resp resp = {0};

	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd == -1)
		return -1;

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
	if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
		close(fd);
		return -1;
	}

	req.op = LITHOSD_OP_GET_STATS;
	n = write(fd, &req, sizeof(req));
	if (n != (ssize_t)sizeof(req)) {
		close(fd);
		return -1;
	}
	n = read(fd, &resp, sizeof(resp));
	close(fd);
	if (n != (ssize_t)sizeof(resp) || resp.status != 0)
		return -1;
	*out_active = resp.active_allocs;
	return 0;
}

static int query_status_row(const char* sock_path, uint32_t idx, struct lithosd_resp* out_resp) {
	int fd;
	ssize_t n;
	struct sockaddr_un addr;
	struct lithosd_req req = {0};
	struct lithosd_resp resp = {0};

	fd = socket(AF_UNIX, SOCK_STREAM, 0);
	if (fd == -1)
		return -1;

	memset(&addr, 0, sizeof(addr));
	addr.sun_family = AF_UNIX;
	strncpy(addr.sun_path, sock_path, sizeof(addr.sun_path) - 1);
	if (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) == -1) {
		close(fd);
		return -1;
	}

	req.op = LITHOSD_OP_GET_STATUS;
	req.reserved = idx;
	n = write(fd, &req, sizeof(req));
	if (n != (ssize_t)sizeof(req)) {
		close(fd);
		return -1;
	}
	n = read(fd, &resp, sizeof(resp));
	close(fd);
	if (n != (ssize_t)sizeof(resp) || resp.status != 0)
		return -1;
	*out_resp = resp;
	return 0;
}

static int snapshot_two_allocs_for_pids(const char* sock_path,
	                                    pid_t pid_a,
	                                    pid_t pid_b,
	                                    struct alloc_snapshot* out_a,
	                                    struct alloc_snapshot* out_b) {
	struct lithosd_resp resp;

	memset(out_a, 0, sizeof(*out_a));
	memset(out_b, 0, sizeof(*out_b));

	for (uint32_t i = 0; i < LITHOS_STATUS_SCAN_LIMIT; i++) {
		if (query_status_row(sock_path, i, &resp) != 0)
			return -1;
		if (!resp.alloc_in_use)
			continue;
		if ((pid_t)resp.alloc_pid == pid_a) {
			out_a->valid = true;
			out_a->pid = resp.alloc_pid;
			out_a->quota = resp.alloc_quota;
			out_a->start_tpc = resp.alloc_start_tpc;
			out_a->disable_mask = resp.alloc_disable_mask;
		}
		if ((pid_t)resp.alloc_pid == pid_b) {
			out_b->valid = true;
			out_b->pid = resp.alloc_pid;
			out_b->quota = resp.alloc_quota;
			out_b->start_tpc = resp.alloc_start_tpc;
			out_b->disable_mask = resp.alloc_disable_mask;
		}
		if (out_a->valid && out_b->valid)
			return 0;
	}

	return 0;
}

static int spawn_app(const char* sock_path, int* out_fd, pid_t* out_pid) {
	int pfd[2];
	pid_t pid;

	if (pipe(pfd) != 0) {
		perror("pipe");
		return -1;
	}

	pid = fork();
	if (pid < 0) {
		perror("fork");
		close(pfd[0]);
		close(pfd[1]);
		return -1;
	}
	if (pid == 0) {
		setenv("LIBSMCTRL_LITHOS_ENABLE", "1", 1);
		setenv("LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE", "1", 1);
		setenv("LIBSMCTRL_LITHOS_TPC_QUOTA_DEFAULT", "1", 1);
		setenv("LIBSMCTRL_LITHOSD_SOCK", sock_path, 1);
		setenv("LIBSMCTRL_LITHOS_TEST_LONG_KERNEL_CYCLES", "1200000000", 1);
		unsetenv("LIBSMCTRL_LITHOS_SCHED_ENABLE");
		setenv("LD_LIBRARY_PATH", ".", 1);

		close(pfd[0]);
		dup2(pfd[1], STDOUT_FILENO);
		close(pfd[1]);
		execl("./lithos_test_arbitrary_app", "lithos_test_arbitrary_app", NULL);
		perror("execl lithos_test_arbitrary_app");
		_exit(127);
	}

	close(pfd[1]);
	*out_fd = pfd[0];
	*out_pid = pid;
	return 0;
}

static int read_app_result(int fd, struct app_result* out) {
	char buf[256] = {0};
	ssize_t nread = read(fd, buf, sizeof(buf) - 1);
	close(fd);
	if (nread <= 0) {
		fprintf(stderr, "arbitrary app produced no output\n");
		return -1;
	}
	buf[nread] = '\0';
	if (parse_result_line(buf, out) != 0) {
		fprintf(stderr, "unexpected app output: %s\n", buf);
		return -1;
	}
	return 0;
}

int main(void) {
	char sock_path[108];
	char tpc_count_str[16];
	pid_t daemon_pid;
	pid_t app_a_pid;
	pid_t app_b_pid;
	int app_a_fd;
	int app_b_fd;
	int status;
	uint32_t num_tpcs = 0;
	struct app_result a = {0};
	struct app_result b = {0};
	struct alloc_snapshot alloc_a = {0};
	struct alloc_snapshot alloc_b = {0};
	bool disjoint;
	uint32_t active_allocs = 0;

	if (libsmctrl_get_tpc_info_cuda(&num_tpcs, 0) != 0 || num_tpcs == 0 || num_tpcs > 64)
		num_tpcs = 64;
	snprintf(tpc_count_str, sizeof(tpc_count_str), "%u", num_tpcs);

	snprintf(sock_path, sizeof(sock_path), "/tmp/lithosd_terminal_%d.sock", getpid());
	unlink(sock_path);

	daemon_pid = fork();
	if (daemon_pid < 0) {
		perror("fork");
		return 2;
	}
	if (daemon_pid == 0) {
		execl("./lithosd", "lithosd", sock_path, tpc_count_str, NULL);
		perror("execl lithosd");
		_exit(127);
	}

	if (wait_for_socket(sock_path) != 0) {
		fprintf(stderr, "Timed out waiting for %s\n", sock_path);
		kill(daemon_pid, SIGTERM);
		waitpid(daemon_pid, &status, 0);
		return 2;
	}

	if (spawn_app(sock_path, &app_a_fd, &app_a_pid) != 0 ||
	    spawn_app(sock_path, &app_b_fd, &app_b_pid) != 0) {
		kill(daemon_pid, SIGTERM);
		waitpid(daemon_pid, &status, 0);
		unlink(sock_path);
		return 3;
	}

	// Best-effort snapshot for diagnostics; allocations may already be reclaimed.
	(void)snapshot_two_allocs_for_pids(sock_path, app_a_pid, app_b_pid, &alloc_a, &alloc_b);

	if (read_app_result(app_a_fd, &a) != 0 || read_app_result(app_b_fd, &b) != 0) {
		kill(app_a_pid, SIGTERM);
		kill(app_b_pid, SIGTERM);
		kill(daemon_pid, SIGTERM);
		waitpid(app_a_pid, NULL, 0);
		waitpid(app_b_pid, NULL, 0);
		waitpid(daemon_pid, NULL, 0);
		unlink(sock_path);
		return 3;
	}

	waitpid(app_a_pid, &status, 0);
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		fprintf(stderr, "arbitrary app A failed (status=%d)\n", status);
		kill(app_b_pid, SIGTERM);
		kill(daemon_pid, SIGTERM);
		waitpid(app_b_pid, NULL, 0);
		waitpid(daemon_pid, NULL, 0);
		unlink(sock_path);
		return 3;
	}
	waitpid(app_b_pid, &status, 0);
	if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
		fprintf(stderr, "arbitrary app B failed (status=%d)\n", status);
		kill(daemon_pid, SIGTERM);
		waitpid(daemon_pid, NULL, 0);
		unlink(sock_path);
		return 3;
	}

	if (query_active_allocs(sock_path, &active_allocs) != 0) {
		fprintf(stderr, "Failed to query daemon allocation stats\n");
		kill(daemon_pid, SIGTERM);
		waitpid(daemon_pid, NULL, 0);
		unlink(sock_path);
		return 3;
	}
	if (active_allocs != 0) {
		fprintf(stderr, "Daemon still tracks %u active launch allocations after app teardown\n", active_allocs);
		kill(daemon_pid, SIGTERM);
		waitpid(daemon_pid, NULL, 0);
		unlink(sock_path);
		return 3;
	}

	kill(daemon_pid, SIGTERM);
	waitpid(daemon_pid, &status, 0);
	unlink(sock_path);

	if (a.uniq > a.sms_per_tpc || b.uniq > b.sms_per_tpc) {
		fprintf(stderr,
		        "Confinement failed: appA uniq=%d (<=%d), appB uniq=%d (<=%d)\n",
		        a.uniq,
		        a.sms_per_tpc,
		        b.uniq,
		        b.sms_per_tpc);
		return 4;
	}

	disjoint = (a.max_smid < b.min_smid) || (b.max_smid < a.min_smid);
	if (!disjoint) {
		if (alloc_a.valid && alloc_b.valid) {
			fprintf(stderr,
			        "Disjointness failed: appA=[%u,%u], appB=[%u,%u], allocA(start=%u,quota=%u,mask=0x%llx), allocB(start=%u,quota=%u,mask=0x%llx)\n",
			        a.min_smid,
			        a.max_smid,
			        b.min_smid,
			        b.max_smid,
			        alloc_a.start_tpc,
			        alloc_a.quota,
			        (unsigned long long)alloc_a.disable_mask,
			        alloc_b.start_tpc,
			        alloc_b.quota,
			        (unsigned long long)alloc_b.disable_mask);
		} else {
			fprintf(stderr,
			        "Disjointness failed: appA=[%u,%u], appB=[%u,%u] (allocation masks unavailable)\n",
			        a.min_smid,
			        a.max_smid,
			        b.min_smid,
			        b.max_smid);
		}
		return 5;
	}

	printf("lithos_test_terminal_arbitrary: pass (A=[%u,%u], B=[%u,%u])\n",
	       a.min_smid,
	       a.max_smid,
	       b.min_smid,
	       b.max_smid);
	return 0;
}
