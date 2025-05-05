// Copyright 2025 Joshua Bakita
// taskset-like utility for the GPU
#define _GNU_SOURCE // For program_invocation_name
#include <argp.h>
#include <errno.h>
#include <error.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <unistd.h>

#include <cuda.h> // To help with getting GPC info

#include "libsmctrl.h"

const char* maintainer = "<jbakita@cs.unc.edu>";
const char* version = "nvtaskset 2025.03";
const char* desc = "taskset-like utility for NVIDIA GPUs.";

const struct argp_option opts[] = {
	{0}
};

unsigned __int128 strtou128(const char *nptr, char **endptr, int base) {
	unsigned __int128 result = 0;
	if (base != 16)
		error(1, EINVAL, "Internal error");
	// Skip a "0x" prefix. Safe due to early evaluation
	if (*nptr == '0' && (*(nptr + 1) == 'x' || *(nptr + 1) == 'X'))
		nptr += 2;
	// Until hitting an invalid character
	while (1) {
		if (*nptr >= 'a' && *nptr <= 'f')
			result = result << 4 | (*nptr - 'a' + 10);
		else if (*nptr >= 'A' && *nptr <= 'F')
			result = result << 4 | (*nptr - 'A' + 10);
		else if (*nptr >= '0' && *nptr <= '9')
			result = result << 4 | (*nptr - '0');
		else
			break;
		nptr++;
	}
	if (endptr)
		*endptr = (char*)nptr;
	return result;
}

void libsmctrl_get_gpc_info_ext_easy(uint32_t* num_gpcs, uint128_t** masks, int gpu_id) {
	int res;
	CUcontext ctx;
	// XXX: Copied from libsmctrl_test_gpc_info
        // Tell CUDA to use PCI device id ordering (to match nvdebug)
        putenv((char*)"CUDA_DEVICE_ORDER=PCI_BUS_ID");
        // A CUDA context is required before reading the topology information
        if ((res = cuInit(0))) {
                const char* name;
                cuGetErrorName(res, &name);
                fprintf(stderr, "%s: Unable to initialize CUDA, error %s\n", program_invocation_name, name);
                exit(1);
        }
        if ((res = cuCtxCreate(&ctx, 0, 0))) {
                const char* name;
                cuGetErrorName(res, &name);
                fprintf(stderr, "%s: Unable to create a CUDA context, error %s\n", program_invocation_name, name);
                exit(1);
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
	// Not copied
	unsetenv("CUDA_DEVICE_ORDER");
}

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "Usage: %s -p <hex mask> <pid>\n", argv[0]);
		fprintf(stderr, "       %s <hex mask> <command> <argument...>\n", argv[0]);
		fprintf(stderr, "       %s --gpc-list <gpc list> <command> <argument...>\n", argv[0]);
		fprintf(stderr, " <hex mask> has a bit set for each TPC to be enabled\n");
		return 1;
	}
	// TODO: Use a proper argument parser
	if (strcmp("-p", argv[1]) == 0) { // Setting mask on running task
		char *end;
		pid_t target_pid = strtoul(argv[2], &end, 10);
		// strtoul stores a pointer to the first invalid character in `end`
		if (*end != '\0') {
			fprintf(stderr, "Invalid character \"%c\" in PID argument.\n", *end);
			return 1;
		}
		unsigned __int128 mask = strtou128(argv[3], &end, 16);
		if (*end != '\0') {
			fprintf(stderr, "Invalid character \"%c\" in mask argument.\n", *end);
			return 1;
		}
		// The shared memory lookup key is the lower 16-bits of the PID | "sm"
		key_t shm_key = target_pid << 16 | (int)'s' << 8 | (int) 'm';
		// Get a handle to the 128-bit shared memory region
		int shmid = shmget(shm_key, 16, 0);
		if (shmid == -1)
			error(1, errno, "Unable to find control region for PID %d", target_pid);
		// Open the shared memory region
		unsigned __int128 *supreme_mask = shmat(shmid, NULL, 0);
		if (supreme_mask == (void*)-1)
			error(1, errno, "Unable to open control region for PID %d", target_pid);
		// Write the requested mask into the shared memory region
		*supreme_mask = mask;
	} else { // Starting a new task with a mask
		// TODO: Check other locations for nvidia-cuda-mps-control if its not on the path
		// TODO: Use dup2() to redirect MPS startup messages
		int ret = system("echo -n | nvidia-cuda-mps-control");
		if (ret == -1)
			error(1, errno, "Unable to run subshell to check MPS status");
		if (ret != 0) { // Control deamon not yet started
			fprintf(stderr, "nvtaskset: MPS control deamon does not appear to be running. Automatically starting...\n");
			ret = system("nvidia-cuda-mps-control -d");
			if (ret == -1)
				error(1, errno, "Unable to run subshell to start MPS");
			if (ret == 1) {
				fprintf(stderr, "nvtaskset: Error starting MPS control deamon. Terminating...\n");
				return 1;
			}
			fprintf(stderr, "nvtaskset: Done. Use \"echo quit | nvidia-cuda-mps-control\" to terminate it later as desired.\n");
		}
		// Tell loader to initialize libsmctrl.so first
		// TODO: Append, rather than overwrite LD_PRELOAD
		setenv("LD_PRELOAD", "./libsmctrl.so", 1);
		// Explictly set the number of channels, otherwise CUDA will only use two
		// (see paper for why that causes problems)
		setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8", 1);
		// Check if a mask, or a list of GPCs is being provided
		if (strcmp(argv[1], "--gpc-list") == 0) {
			// TODO: Support the full syntax that taskset supports
			// We just support X,Y,Z for now
			uint32_t num_gpcs = 0;
			uint128_t* masks = NULL;
			// TODO: Allow specifying GPU ID, rather than assuming 0!
			libsmctrl_get_gpc_info_ext_easy(&num_gpcs, &masks, 0);
			uint128_t mask = 0;
			int range_start_gpc = -1;
			char* start = argv[2];
			int len = strlen(argv[2]);
			// TODO: Handle invalid input cleanly.
			// Convert comma-seperated GPC list into a mask
			for (int i = 0; i < len + 1; i++) {
				if (argv[2][i] == ',' || argv[2][i] == '\0') {
					argv[2][i] = '\0';
					int gpc = atoi(start);
					if (gpc > num_gpcs - 1) {
						fprintf(stderr, "Invalid GPC ID '%s'!\n", start);
						return 1;
					}
					// Handle ranges
					if (range_start_gpc != -1) {
						if (range_start_gpc >= gpc) {
							fprintf(stderr, "Invalid GPC range!\n");
							return 1;
						}
						while (range_start_gpc <= gpc) {
							//printf("gpc %i\n", range_start_gpc);
							mask |= masks[range_start_gpc];
							range_start_gpc++;
						}
						range_start_gpc = -1;
					} else {
						//printf("gpc %i\n", gpc);
						mask |= masks[gpc];
					}
					start = argv[2] + i + 1;
				}
				// Range start
				if (argv[2][i] == '-') {
					argv[2][i] = '\0';
					range_start_gpc = atoi(start);
					start = argv[2] + i + 1;
				}
			}
			// Convert to string, prefix with ~, and set env var
			char mask_str[32+3+1]; // 32 hexits, "~0x", and '\0'
			snprintf(mask_str, 36, "~0x%lx%016lx", (uint64_t)(mask >> 64), (uint64_t)mask);
			//printf("nvtaskset: Using mask string %s\n", mask_str);
			setenv("LIBSMCTRL_MASK", mask_str, 1);
			// Start task
			execvp(argv[3], argv+3);
			error(1, errno, "Unable to launch task '%s'", argv[3]);
		} else {
			// Tell libsmctrl what mask to use
			char* mask = malloc(strlen(argv[1]) + 2);
			mask[0] = '~'; // Make an enable mask
			strcpy(mask+1, argv[1]);
			setenv("LIBSMCTRL_MASK", mask, 1);
			free(mask); // setenv() made a copy
			// Start task
			execvp(argv[2], argv+2);
			error(1, errno, "Unable to launch task '%s'", argv[2]);
		}
	}
	fprintf(stderr, "Invalid arguments\n");
	return 1;
}
