// Copyright 2024-2025 Joshua Bakita
// Obtain and print the correspondence between TPCs and GPCs for a given GPU.
//
// Known issues:
// - If CUDA cannot see the same number of GPUs as the nvdebug kernel module,
//   the passed GPU ID may not properly correspond to to an ID an CUDA. This
//   will cause us to fail to initialize a context on the right device, and
//   may cause the test to terminate due to no initialized context. This should
//   only happen if some of the attached GPUs are too old or new for CUDA.
#define _GNU_SOURCE
#include <cuda.h>

#include <error.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "libsmctrl.h"

int main(int argc, char** argv) {
	uint32_t num_gpcs = 0, num_tpcs = 0;
	uint128_t* masks = NULL;
	int res, print_width, gpu_id;
	CUcontext ctx;
	// Optionally support specifying the GPU ID to query via an argument
	// Important: This GPU ID must match the ID used by the nvdebug module. See
	//            the documentation on libsmctrl_get_gpc_info() for details.
	if (argc > 2 || (argc == 2 && (!strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")))) {
		fprintf(stderr, "Usage: %s <nvdebug GPU ID>\n", argv[0]);
		return 1;
	}
	if (argc > 1)
		gpu_id = atoi(argv[1]);
	else
		gpu_id = 0;
	// Tell CUDA to use PCI device id ordering (to match nvdebug)
	putenv((char*)"CUDA_DEVICE_ORDER=PCI_BUS_ID");
	// Allow CUDA to see all devices (to better match nvdebug)
	unsetenv("CUDA_VISIBLE_DEVICES");
	// A CUDA context is required before reading the topology information
	if ((res = cuInit(0))) {
		const char* name;
		cuGetErrorName(res, &name);
		fprintf(stderr, "%s: Unable to initialize CUDA, error %s\n", program_invocation_name, name);
		return 1;
	}
	if ((res = cuCtxCreate(&ctx, 0, gpu_id))) {
		const char* name;
		cuGetErrorName(res, &name);
		fprintf(stderr, "%s: Unable to create a CUDA context, error %s\n", program_invocation_name, name);
		return 1;
	}
	// Pull topology information from libsmctrl
	if ((res = libsmctrl_get_gpc_info_ext(&num_gpcs, &masks, gpu_id)) != 0) {
		error(0, res, "libsmctrl_get_gpc_info() failed");
		if (res == ENOENT)
			fprintf(stderr, "%s: Is the nvdebug kernel module loaded?\n", program_invocation_name);
		if (res == EIO)
			fprintf(stderr, "%s: Is the GPU powered on, i.e., is there an active context?\n", program_invocation_name);
		return 1;
	}
	printf("%s: GPU%d has %d enabled GPCs.\n", program_invocation_name, gpu_id, num_gpcs);
	// Determine how wide the print should be (for pretty-printing)
	print_width = 0;
	for (int i = 0; i < num_gpcs; i++) {
		int shift = 0;
		while (masks[i] >> shift)
			shift++;
		if (shift > print_width)
			print_width = shift;
	}
	// Convert the width to a number of octets, rather than number of bits
	// (Result of integer divison, +1 if it does not evenly divide)
	print_width = print_width/4 + !!(print_width % 4);
	for (int i = 0; i < num_gpcs; i++) {
		// No built-in for 128-bit integers, so split it into two 64-bit ones
		int num_tpcs_local =  __builtin_popcountl(masks[i]) + __builtin_popcountl(masks[i] >> 64);
		num_tpcs += num_tpcs_local;
		if (print_width > 16)
			printf("%s: Mask of %d TPCs associated with GPC %d: 0x%0*lx%016lx\n",
			       program_invocation_name, num_tpcs_local, i, print_width - 16,
			       (uint64_t)(masks[i] >> 64), (uint64_t)masks[i]);
		else
			printf("%s: Mask of %d TPCs associated with GPC %d: 0x%0*lx\n",
			       program_invocation_name, num_tpcs_local, i, print_width,
			       (uint64_t)masks[i]);
	}
	printf("%s: Total of %u enabled TPCs.\n", program_invocation_name, num_tpcs);
	return 0;
}
