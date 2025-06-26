// Copyright 2025 Joshua Bakita
// Show or change the GPU core affinity for a CUDA process
// taskset-like utility for NVIDIA GPUs
#define _GNU_SOURCE // For program_invocation_name
#include <argp.h>
#include <dirent.h>
#include <errno.h>
#include <error.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cuda.h> // To help with getting GPC info

#include "libsmctrl.h"

#define LINK_NAME "/memfd:libsmctrl"

// TODO: Write automated tests:
// - Change region of non-existent PID
// - Change region of permission denied PID
// - Change region of non-GPU PID
// - Change GPC list
// - Change TPC list
// - Change TPC mask
// - Start TPC mask
// - Start TPC list
// - Start GPC list
// - Start with subargument containing -
// - Query GPC list
// - Query TPC list
// - Query TPC mask
// - Set GPC list w/ non-existant GPC
// - Set TPC list w/ non-existant TPC

// Private symbols from libsmctrl
extern bool libsmctrl_is_mps_running();
extern uint128_t strtou128(const char *nptr, char **endptr, int base);

const char *argp_program_bug_address = "<jbakita@cs.unc.edu>";
const char *argp_program_version = "nvtaskset 2025.06";
const char desc[] = "Show or change the GPU core affinity for a CUDA process\v"
                    "Warning: When using GPC lists, this tool currently "
                    "derives TPC to GPC mappings from the first NVIDIA GPU in "
                    "the system (by PCI bus ID) device. To use the mappings "
                    "for a different device, use the `libsmctrl_test_get_info` "
                    "tool to get the bitmask of TPCs associated with each GPC, "
                    "OR them, and then set that bitmask via this tool. Better "
                    "multi-GPU support is intended for a future release.\n\n"
                    "Inspired by the Linux taskset utility.";
const char args_doc[] = "[mask | list] [pid | cmd [args...]]";

const struct argp_option opts[] = {
	{"gpc-list", 'g', NULL, 0, "Specify partition as a list of GPCs"},
	{"tpc-list", 't', NULL, 0, "Specify partition as a list of TPCs"},
	{"pid",      'p', NULL, 0, "Operate on an existing PID"},
	{0}
};

// Create a CUDA context and query the associated GPC to TPC mappings
// Based off logic in libsmctrl_test_gpc_info
void libsmctrl_get_gpc_info_ext_easy(uint32_t* num_gpcs, uint128_t** masks, int gpu_id) {
	int res;
	CUcontext ctx;
	char *old_order = NULL;
	int old_stderr, dev_null_fd;

	// Attempt to read the configuration, assuming the GPU is on, and fall
	// back to creating a context if this fails.
	// (Creating a CUDA context is very expensive and best avoided)
	// (Redirect stderr while doing this to mute libsmctrl error messages)
	if ((dev_null_fd = open("/dev/null", O_WRONLY)) == -1)
		error(1, errno, "Unable to open /dev/null");
	if (old_stderr = dup(STDERR_FILENO) == -1)
		error(1, errno, "Unable to duplicate stderr file descriptor");
	if (dup2(dev_null_fd, STDERR_FILENO) == -1)
		error(1, errno, "Unable to overwrite stderr file descriptor");
	res = libsmctrl_get_gpc_info_ext(num_gpcs, masks, gpu_id);
	if (dup2(old_stderr, STDERR_FILENO) == -1)
		error(1, errno, "Unable to restore stderr file descriptor");
	// End if we were successful, otherwise fallback
	if (res == 0)
		return;

	// Tell CUDA to use PCI device id ordering (to match nvdebug)
	putenv((char*)"CUDA_DEVICE_ORDER=PCI_BUS_ID");
	// Allow CUDA to see all devices (to better match nvdebug)
	if (getenv("CUDA_VISIBLE_DEVICES")) {
		if (!(old_order = strdup(getenv("CUDA_VISIBLE_DEVICES"))))
			error(1, errno, "Unable to allocate environment string");
		unsetenv("CUDA_VISIBLE_DEVICES");
	}
	// A CUDA context is required before reading the topology information
	if ((res = cuInit(0))) {
		const char* name;
		cuGetErrorName(res, &name);
		error(1, 0, "Unable to create a initialize CUDA, error %s", name);
	}
	if ((res = cuCtxCreate(&ctx, 0, gpu_id))) {
		const char* name;
		cuGetErrorName(res, &name);
		error(1, 0, "Unable to create a CUDA context, error %s", name);
	}
	// Pull topology information from libsmctrl
	if ((res = libsmctrl_get_gpc_info_ext(num_gpcs, masks, gpu_id)) != 0) {
		error(0, res, "libsmctrl_get_gpc_info() failed");
		if (res == ENOENT)
			fprintf(stderr, "%s: Is the nvdebug kernel module loaded?\n", program_invocation_name);
		if (res == EIO)
			fprintf(stderr, "%s: Is the GPU powered on, i.e., is there an active context?\n", program_invocation_name);
		exit(1);
	}
	// Delete the CUDA context
	if (res = cuCtxDestroy(ctx)) {
		const char* name;
		cuGetErrorName(res, &name);
		error(1, 0, "Unable to destroy CUDA context, error %s", name);
	}
	// Restore the environment (in case we exec() later)
	unsetenv("CUDA_DEVICE_ORDER");
	if (old_order) {
		setenv("CUDA_VISIBLE_DEVICES", old_order, 1);
		free(old_order);
	}
}

int parse_list(bool use_gpcs, char* list, uint128_t *mask_out) {
	// We support the same ranges as taskset, e.g., X,Y,Z and X,Y-Z
	uint32_t num_xpcs = 0; // Either TPC or GPC count, i.e., "X"PC
	uint128_t* masks = NULL;
	// TODO: Allow specifying GPU ID, rather than assuming 0!
	if (use_gpcs)
		libsmctrl_get_gpc_info_ext_easy(&num_xpcs, &masks, 0);
	else
		libsmctrl_get_tpc_info_cuda(&num_xpcs, 0);
	uint128_t mask = 0;
	int range_start_xpc = -1;
	char* start = list;
	int len = strlen(list);
	// Convert comma-seperated GPC/TPC list into a mask
	for (int i = 0; i < len + 1; i++) {
		if (list[i] == ',' || list[i] == '\0') {
			list[i] = '\0';
			int xpc = atoi(start);
			if (xpc > num_xpcs - 1)
				error(1, EINVAL, "%s is not a valid %s ID", start, use_gpcs ? "GPC" : "TPC");
			// Handle ranges
			if (range_start_xpc != -1) {
				if (range_start_xpc >= xpc)
					error(1, EINVAL, "Malformed %s range", use_gpcs ? "GPC" : "TPC");
				while (range_start_xpc <= xpc) {
					if (use_gpcs)
						mask |= masks[range_start_xpc];
					else
						mask |= (uint128_t)1 << range_start_xpc;
					range_start_xpc++;
				}
				range_start_xpc = -1;
			} else {
				if (use_gpcs)
					mask |= masks[xpc];
				else
					mask |= (uint128_t)1 << xpc;
			}
			start = list + i + 1;
		}
		// Range start
		if (list[i] == '-') {
			list[i] = '\0';
			range_start_xpc = atoi(start);
			start = list + i + 1;
		}
	}
	*mask_out = mask;
	return 0;
}

// Always returns a valid string
char* compose_list(uint128_t mask) {
	// List will always be shorter than every TPC, comma-seperated
	// 128 TPCs, with 10 1-char, 90 2-char, 28 3-char, 127 commas, and 1 null
	static char list[10 + 90*2 + 28*3 + 128];
	char* tail = list;
	int last_enabled = -2;
	bool in_range;
	for (int i = 0; i < 128; i++) {
		bool enabled = (mask >> i) & 1;
		if (in_range) {
			if (enabled) {
				last_enabled = i;
			} else {
				tail += sprintf(tail, "%d,", last_enabled);
				in_range = false;
			}
			continue;
		}
		if (enabled) {
			if (last_enabled == i - 1) {
				in_range = true;
				tail += sprintf(tail, "-");
			} else {
				tail += sprintf(tail, "%d", i);
			}
			last_enabled = i;
		} else {
			if (last_enabled == i - 1) {
				tail += sprintf(tail, ",");
			}
		}
	}
	// Strip trailing comma
	if (*(tail - 1) == ',')
		*(tail - 1) = '\0';
	return list;
}

// Always returns a valid string
// (Terminates the program on error)
char* compose_gpc_list(uint128_t mask) {
	uint32_t num_gpcs = 0;
	uint128_t* masks = NULL;
	libsmctrl_get_gpc_info_ext_easy(&num_gpcs, &masks, 0);
	uint128_t gpc_mask = 0;
	// Try to find correspondence between a list of TPCs and GPCs
	for (int gpc = 0; gpc < num_gpcs; gpc++) {
		if ((masks[gpc] & mask) == masks[gpc]) {
			gpc_mask |= 1 << gpc;
			mask &= ~masks[gpc];
		}
	}
	if (mask)
		error(1, EINVAL, "Unable to interpret affinity as GPC list; try -t instead of -g");
	return compose_list(gpc_mask);
}


uint128_t* get_mask_hndl(pid_t target_pid) {
	char fd_path[277];
	int fd;
	uint128_t *mask_hndl;
	DIR *dp;
	struct dirent *entry;
	// Search for the file descriptor which represents the libsmctrl control
	// region.
	snprintf(fd_path, 277, "/proc/%d/fd/", target_pid);
	if (!(dp = opendir(fd_path))) {
		if (errno == ENOENT)
			error(1, 0, "Unable to find PID %d.", target_pid);
		else
			error(1, errno, "Unable to access PID %d", target_pid);
	}
	while (entry = readdir(dp)) {
		char link[sizeof(LINK_NAME)];
		snprintf(fd_path, 277, "/proc/%d/fd/%s", target_pid, entry->d_name);
		readlink(fd_path, link, sizeof(LINK_NAME));
		if (strncmp(LINK_NAME, link, sizeof(LINK_NAME) - 1) == 0)
			break;
	}
	closedir(dp);
	if (!entry)
		error(1, 0, "Unable to find libsmctrl-wrapper control region for PID %d.", target_pid);
	// Access the shared memory region for libsmctrl control.
	if ((fd = open(fd_path, O_RDWR)) == -1)
		error(1, errno, "Unable to open libsmctrl-wrapper control file %s", fd_path);
	mask_hndl = mmap(NULL, 16, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (mask_hndl == MAP_FAILED)
		error(1, errno, "Unable to memory-map libsmctrl-wrapper control file %s", fd_path);
	close(fd);
	return mask_hndl;
}

static error_t arg_parser(int key, char* arg, struct argp_state *state){
	static bool is_cmd = true;
	static bool is_query = false;
	static bool is_list = false;
	static bool use_gpcs = false;
	static uint128_t mask = 0;
	static pid_t target_pid = 0;
	static char **sub_argv = NULL;
	char *end;
	// Handle what to do in case of each option
	switch (key) {
		case 'g':
			if (is_list)
				argp_error(state, "Only one of -g and -t may be specified.\n");
			use_gpcs = true;
			is_list = true;
			break;
		case 't':
			if (is_list)
				argp_error(state, "Only one of -g and -t may be specified.\n");
			is_list = true;
			break;
		case 'p':
			is_cmd = false;
			break;
		case ARGP_KEY_ARG:
			// Options:
			// 1. -p and one argument -> Query mask for PID
			// 2. -p and two arguments -> Set mask for PID
			// 3. No -p and at least one argument -> Set mask and launch command
			// (otherwise: invalid)
			if (state->arg_num == 0 && !is_cmd && state->argc - state->next == 0)
				is_query = true;
			// Handle invalid and valid query cases
			if (is_query) {
				if (state->arg_num == 0) {
					target_pid = strtoul(arg, &end, 10);
					if (*end != '\0')
						argp_error(state, "Invalid character \"%c\" in PID argument.\n", *end);
					break;
				} else
					return ARGP_ERR_UNKNOWN;
			}
			// Handle non-query cases
			if (state->arg_num == 0 && state->argc - state->next != 0) {
				if (is_list) {
					parse_list(use_gpcs, arg, &mask);
				} else {
					// strtoul stores a pointer to the first invalid character in `end`
					mask = strtou128(arg, &end, 16);
					if (*end != '\0')
						argp_error(state, "Invalid character \"%c\" in mask argument.\n", *end);
				}
			} else if (state->arg_num == 1 && !is_cmd) {
				target_pid = strtoul(arg, &end, 10);
				if (*end != '\0')
					argp_error(state, "Invalid character \"%c\" in PID argument.\n", *end);
			} else
				return ARGP_ERR_UNKNOWN;
			break;
		case ARGP_KEY_ARGS:
			if (!is_cmd)
				return ARGP_ERR_UNKNOWN;
			sub_argv = state->argv + state->next;
			break;
		case ARGP_KEY_END:
			if (is_query && state->arg_num < 1)
				argp_usage(state);
			else if (!is_query && state->arg_num < 2)
				argp_usage(state);
			break;
		case ARGP_KEY_FINI:
			if (is_query) {
				// query PID
				uint128_t* mask_hndl = get_mask_hndl(target_pid);
				uint128_t enable_mask = ~*mask_hndl;
				if (use_gpcs & is_list)
					printf("PID %d's current GPC affinity list: %s\n", target_pid, compose_gpc_list(enable_mask));
				else if (use_gpcs & !is_list)
					argp_error(state, "Unsupported to print query as a GPC mask.\n");
				else if (is_list)
					printf("PID %d's current TPC affinity list: %s\n", target_pid, compose_list(enable_mask));
				else
					printf("PID %d's current TPC affinity mask: 0x%.0lx%016lx\n", target_pid, (uint64_t)(enable_mask >> 64), (uint64_t)enable_mask);
			} else if (is_cmd) {
				// start MPS (as needed)
				if (!libsmctrl_is_mps_running()) {
					fprintf(stderr, "nvtaskset: MPS control deamon does not appear to be running. Automatically starting...\n");
					int ret = system("nvidia-cuda-mps-control -d");
					if (ret == -1)
						error(1, errno, "Unable to run subshell to start MPS");
					if (ret == 1) {
						fprintf(stderr, "nvtaskset: Error starting MPS control deamon. Terminating...\n");
						return 1;
					}
					fprintf(stderr, "nvtaskset: Done. Use \"echo quit | nvidia-cuda-mps-control\" to terminate it later as desired.\n");
				}
				// launch subprocess
				// Convert to string, prefix with ~, and set env var
				char mask_str[32+3+1]; // 32 hexits, "~0x", and '\0'
				snprintf(mask_str, 36, "~0x%.0lx%016lx", (uint64_t)(mask >> 64), (uint64_t)mask);
				setenv("LIBSMCTRL_MASK", mask_str, 1);
				// Start task
				execvp(sub_argv[0], sub_argv);
				error(1, errno, "Unable to launch task '%s'", sub_argv[0]);
			} else {
				if (!libsmctrl_is_mps_running())
					printf("Warning: NVIDIA MPS is not running. CUDA programs will not co-run! Run nvidia-cuda-mps-control -d before launching any CUDA-using programs that should co-run.\n");
				// change mask on PID
				uint128_t* mask_hndl = get_mask_hndl(target_pid);
				if (!is_list) {
					printf("PID %d's current TPC affinity mask: 0x%.0lx%016lx\n", target_pid, ~(uint64_t)(*mask_hndl >> 64), ~(uint64_t)*mask_hndl);
					printf("PID %d's new TPC affinity mask: 0x%.0lx%016lx\n", target_pid, (uint64_t)(mask >> 64), (uint64_t)mask);
				} else {
					printf("PID %d's current TPC affinity list: %s\n", target_pid, compose_list(~*mask_hndl));
					printf("PID %d's new TPC affinity list: %s\n", target_pid, compose_list(mask));
				}
				// Write the requested mask into the shared memory region
				*mask_hndl = ~mask;
			}
			break;
		default:
			return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

struct argp argp = {opts, arg_parser, args_doc, desc};

int main(int argc, char **argv) {
	argp_parse(&argp, argc, argv, ARGP_IN_ORDER, 0, NULL);
	return 0;
}
