/**
 * Copyright 2022-2025 Joshua Bakita
 * Library to control SM masks on CUDA launches. Co-opts preexisting debug
 * logic in the CUDA driver library, and thus requires a build with -lcuda.
 *
 * This file implements partitioning via three different mechanisms:
 * - Modifying the QMD/TMD immediately prior to upload
 * - Changing a field in CUDA's stream struct that CUDA applies to the QMD/TMD
 * This table shows the mechanism used with each CUDA version:
 *   +-----------+---------------+---------------+--------------+
 *   |  Version  |  Global Mask  |  Stream Mask  |  Next Mask   |
 *   +-----------+---------------+---------------+--------------+
 *   | 8.0-12.8  | TMD/QMD Hook  | stream struct | TMD/QMD Hook |
 *   | 6.5-7.5   | TMD/QMD Hook  | N/A           | TMD/QMD Hook |
 *   +-----------+---------------+---------------+--------------+
 * "N/A" indicates that a mask type is unsupported on that CUDA version.
 * Please contact the authors if support is needed for a particular feature on
 * an older CUDA version. Support for those is unimplemented, not impossible.
 *
 * An old implementation of this file affected the global mask on CUDA 10.2 by
 * changing a field in CUDA's global struct that CUDA applies to the QMD/TMD.
 * That implementation was extraordinarily complicated, and was replaced in
 * 2024 with a more-backward-compatible way of hooking the TMD/QMD.
 * View the old implementation via Git: `git show aa63a02e:libsmctrl.c`.
 */
#define _GNU_SOURCE // To enable use of memfd_create()
#include <cuda.h>

#include <errno.h>
#include <error.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <unistd.h>

#include "libsmctrl.h"

// In functions that do not return an error code, we favor terminating with an
// error rather than merely printing a warning and continuing.
#define abort(ret, errno, ...) error_at_line(ret, errno, __FILE__, __LINE__, \
                                             __VA_ARGS__)

/*** QMD/TMD-based SM Mask Control via Debug Callback. ***/

// Tested working on x86_64 CUDA 6.5, 9.1, and various 10+ versions
// (No testing attempted on pre-CUDA-6.5 versions)
// Values for the following three lines can be extracted by tracing CUPTI as
// it interects with libcuda.so to set callbacks.
static const CUuuid callback_funcs_id = {{0x2c, (char)0x8e, 0x0a, (char)0xd8, 0x07, 0x10, (char)0xab, 0x4e, (char)0x90, (char)0xdd, 0x54, 0x71, (char)0x9f, (char)0xe5, (char)0xf7, 0x4b}};
// These callback descriptors appear to intercept the TMD/QMD late enough that
// CUDA has already applied the per-stream mask from its internal data
// structures, allowing us to override it with the next mask.
#define QMD_DOMAIN 0xb
#define QMD_PRE_UPLOAD 0x1
/**
 * These globals must be non-static (i.e., have global linkage) to ensure that
 * if multiple copies of the library are loaded (e.g., dynamically linked to
 * both this program and a dependency), secondary copies do not attempt to
 * repeat initialization or make changes to unused copies of mask values.
 */
// Supreme mask (cannot be overridden)
uint128_t *g_supreme_sm_mask = NULL;
// Global mask (applies across all threads)
uint64_t g_sm_mask = 0;
// Next mask (applies per-thread)
__thread uint64_t g_next_sm_mask = 0;
// Flag value to indicate if setup has been completed
bool sm_control_setup_called = false;

#ifdef LIBSMCTRL_STATIC
// Special handling for if built as a static library, and the libcuda.so.1
// libsmctrl wrapper is in use (see comment on setup() constructor for detail).
static void (*shared_set_global_mask)(uint64_t) = NULL;
static void (*shared_set_next_mask)(uint64_t) = NULL;
#endif

// v1 has been removed---it intercepted the TMD/QMD too early, making it
// impossible to override the CUDA-injected stream mask with the next mask.
static void control_callback_v2(void *ukwn, int domain, int cbid, const void *in_params) {
	// ***Only tested on platforms with 64-bit pointers.***
	// The first 8-byte element in `in_params` appears to be its size. `in_params`
	// must have at least five 8-byte elements for index four to be valid.
	if (*(uint32_t*)in_params < 5 * sizeof(void*))
		abort(1, 0, "Unsupported CUDA version for callback-based SM masking. Aborting...");
	// The fourth 8-byte element in `in_params` is a pointer to the TMD. Note
	// that this fourth pointer must exist---it only exists when the first
	// 8-byte element of `in_params` is at least 0x28 (checked above).
	void* tmd = *((void**)in_params + 4);
	if (!tmd)
		abort(1, 0, "TMD allocation appears NULL; likely forward-compatibilty issue.\n");

	uint32_t *lower_ptr, *upper_ptr, *ext_lower_ptr, *ext_upper_ptr;

	// The location of the TMD version field seems consistent across versions
	uint8_t tmd_ver = *(uint8_t*)(tmd + 72);

	if (tmd_ver >= 0x40) {
		// TMD V04_00 is used starting with Hopper to support masking >64 TPCs
		lower_ptr = tmd + 304;
		upper_ptr = tmd + 308;
		ext_lower_ptr = tmd + 312;
		ext_upper_ptr = tmd + 316;
		// XXX: Disable upper 64 TPCs until we have ...next_mask_ext and
		//      ...global_mask_ext
		*ext_lower_ptr = -1;
		*ext_upper_ptr = -1;
		// An enable bit is also required
		*(uint32_t*)tmd |= 0x80000000;
	} else if (tmd_ver >= 0x16) {
		// TMD V01_06 is used starting with Kepler V2, and is the first to
		// support TPC masking
		lower_ptr = tmd + 84;
		upper_ptr = tmd + 88;
	} else {
		// TMD V00_06 is documented to not support SM masking
		abort(1, 0, "TMD version %04o is too old! This GPU does not support SM masking.\n", tmd_ver);
	}

	// Setting the next mask overrides both per-stream and global masks
	if (g_next_sm_mask) {
		*lower_ptr = (uint32_t)g_next_sm_mask;
		*upper_ptr = (uint32_t)(g_next_sm_mask >> 32);
		g_next_sm_mask = 0;
	} else if (!*lower_ptr && !*upper_ptr){
		// Only apply the global mask if a per-stream mask hasn't been set
		*lower_ptr = (uint32_t)g_sm_mask;
		*upper_ptr = (uint32_t)(g_sm_mask >> 32);
	}

	// No one may override the supreme SM mask; any SMs disabled in it (set
	// bits) must always remain disabled.
	if (g_supreme_sm_mask) {
		*lower_ptr |= (uint32_t)*g_supreme_sm_mask;
		*upper_ptr |= (uint32_t)(*g_supreme_sm_mask >> 32);
		if (tmd_ver >= 0x40) {
			*ext_lower_ptr |= (uint32_t)(*g_supreme_sm_mask >> 64);
			*ext_upper_ptr |= (uint32_t)(*g_supreme_sm_mask >> 96);
		}
	}

	//fprintf(stderr, "Final SM Mask (lower): %x\n", *lower_ptr);
	//fprintf(stderr, "Final SM Mask (upper): %x\n", *upper_ptr);
}

static void setup_sm_control_callback() {
	int (*subscribe)(uint32_t* hndl, void(*callback)(void*, int, int, const void*), void* ukwn);
	int (*enable)(uint32_t enable, uint32_t hndl, int domain, int cbid);
	uintptr_t* tbl_base;
	uint32_t my_hndl;
	// Avoid race conditions (setup should only run once)
	if (__atomic_test_and_set(&sm_control_setup_called, __ATOMIC_SEQ_CST))
		return;

#if CUDA_VERSION <= 6050
	// Verify supported CUDA version
	// It's impossible for us to run with a version of CUDA older than we were
	// built by, so this check is excluded if built with CUDA > 6.5.
	int ver = 0;
	cuDriverGetVersion(&ver);
	if (ver < 6050)
		abort(1, ENOSYS, "Global or next masking requires at least CUDA 6.5; "
		                 "this application is using CUDA %d.%d",
		                 ver / 1000, (ver % 100));
#endif

	// Set up callback
	cuGetExportTable((const void**)&tbl_base, &callback_funcs_id);
	uintptr_t subscribe_func_addr = *(tbl_base + 3);
	uintptr_t enable_func_addr = *(tbl_base + 6);
	subscribe = (typeof(subscribe))subscribe_func_addr;
	enable = (typeof(enable))enable_func_addr;
	int res = 0;
	res = subscribe(&my_hndl, control_callback_v2, NULL);
	if (res)
		abort(1, 0, "Error subscribing to launch callback. CUDA returned error code %d.", res);
	res = enable(1, my_hndl, QMD_DOMAIN, QMD_PRE_UPLOAD);
	if (res)
		abort(1, 0, "Error enabling launch callback. CUDA returned error code %d.", res);
}

// Set default mask for all launches
void libsmctrl_set_global_mask(uint64_t mask) {
#ifdef LIBSMCTRL_STATIC
	// Special handling for if built as a static library, and the libcuda.so.1
	// libsmctrl wrapper is in use (see comment on setup() constructor for
	// detail).
	if (shared_set_global_mask)
		return (*shared_set_global_mask)(mask);
#endif
	setup_sm_control_callback();
	g_sm_mask = mask;
}

// Set mask for next launch from this thread
void libsmctrl_set_next_mask(uint64_t mask) {
#ifdef LIBSMCTRL_STATIC
	// Special handling for if built as a static library, and the libcuda.so.1
	// libsmctrl wrapper is in use (see comment on setup() constructor for
	// detail).
	if (shared_set_next_mask)
		return (*shared_set_next_mask)(mask);
#endif
	setup_sm_control_callback();
	g_next_sm_mask = mask;
}


/*** Per-Stream SM Mask (unlikely to be forward-compatible) ***/

// Offsets for the stream struct on x86_64
// No offset appears to work with CUDA 6.5 (tried 0x0--0x1b4 w/ 4-byte step)
// 6.5 tested on 340.118
#define CU_8_0_MASK_OFF 0xec
#define CU_9_0_MASK_OFF 0x130
// CUDA 9.0 and 9.1 use the same offset
// 9.1 tested on 390.157
#define CU_9_2_MASK_OFF 0x140
#define CU_10_0_MASK_OFF 0x244
// CUDA 10.0, 10.1 and 10.2 use the same offset
// 10.1 tested on 418.113
// 10.2 tested on 440.100, 440.82, 440.64, and 440.36
#define CU_11_0_MASK_OFF 0x274
#define CU_11_1_MASK_OFF 0x2c4
#define CU_11_2_MASK_OFF 0x37c
// CUDA 11.2, 11.3, 11.4, and 11.5 use the same offset
// 11.4 tested on 470.223.02
#define CU_11_6_MASK_OFF 0x38c
#define CU_11_7_MASK_OFF 0x3c4
#define CU_11_8_MASK_OFF 0x47c
// 11.8 tested on 520.56.06
#define CU_12_0_MASK_OFF 0x4cc
// CUDA 12.0 and 12.1 use the same offset
// 12.0 tested on 525.147.05
#define CU_12_2_MASK_OFF 0x4e4
// 12.2 tested on 535.129.03
#define CU_12_3_MASK_OFF 0x49c
// 12.3 tested on 545.29.06
#define CU_12_4_MASK_OFF 0x4ac
// 12.4 tested on 550.54.14 and 550.54.15
#define CU_12_5_MASK_OFF 0x4ec
// CUDA 12.5 and 12.6 use the same offset
// 12.5 tested on 555.58.02
// 12.6 tested on 560.35.03
#define CU_12_7_MASK_OFF 0x4fc
// CUDA 12.7 and 12.8 use the same offset
// 12.7 tested on 565.77
// 12.8 tested on 570.124.06

// Offsets for the stream struct on Jetson aarch64
#define CU_9_0_MASK_OFF_JETSON 0x128
// 9.0 tested on Jetpack 3.x (TX2, Nov 2023)
#define CU_10_2_MASK_OFF_JETSON 0x24c
// 10.2 tested on Jetpack 4.x (AGX Xaver and TX2, Nov 2023)
#define CU_11_4_MASK_OFF_JETSON 0x394
// 11.4 tested on Jetpack 5.x (AGX Orin, Nov 2023)
// TODO: 11.8, 12.0, 12.1, and 12.2 on Jetpack 5.x via compatibility packages
#define CU_12_2_MASK_OFF_JETSON 0x50c
// 12.2 tested on Jetpack 6.x (AGX Orin, Dec 2024)
#define CU_12_4_MASK_OFF_JETSON 0x4c4
// 12.4 tested on Jetpack 6.x with cuda-compat-12-4 (AGX Orin, Dec 2024)
#define CU_12_5_MASK_OFF_JETSON 0x50c
// 12.5 tested on Jetpack 6.x with cuda-compat-12-5 (AGX Orin, Dec 2024)
#define CU_12_6_MASK_OFF_JETSON 0x514
// 12.6 tested on Jetpack 6.x with cuda-compat-12-6 (AGX Orin, Dec 2024)

// Used up through CUDA 11.8 in the stream struct
struct stream_sm_mask {
	uint32_t upper;
	uint32_t lower;
};

// Used starting with CUDA 12.0 in the stream struct
struct stream_sm_mask_v2 {
	uint32_t enabled;
	uint32_t mask[4];
};

// Check if this system has a Parker SoC (TX2/PX2 chip)
// (CUDA 9.0 behaves slightly different on this platform.)
// @return 1 if detected, 0 if not, -cuda_err on error
#if __aarch64__
static int detect_parker_soc() {
	int cap_major, cap_minor, err, dev_count;
	if (err = cuDeviceGetCount(&dev_count))
		return -err;
	// As CUDA devices are numbered by order of compute power, check every
	// device, in case a powerful discrete GPU is attached (such as on the
	// DRIVE PX2). We detect the Parker SoC via its unique CUDA compute
	// capability: 6.2.
	for (int i = 0; i < dev_count; i++) {
		if (err = cuDeviceGetAttribute(&cap_minor,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
		                               i))
			return -err;
		if (err = cuDeviceGetAttribute(&cap_major,
		                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
		                               i))
			return -err;
		if (cap_major == 6 && cap_minor == 2)
			return 1;
	}
	return 0;
}
#endif // __aarch64__

// Should work for CUDA 8.0 through 12.8
// A cudaStream_t is a CUstream*. We use void* to avoid a cuda.h dependency in
// our header
void libsmctrl_set_stream_mask(void* stream, uint64_t mask) {
	// When the old API is used on GPUs with over 64 TPCs, disable all TPCs >64
	uint128_t full_mask = -1;
	full_mask <<= 64;
	full_mask |= mask;
	libsmctrl_set_stream_mask_ext(stream, full_mask);
}

void libsmctrl_set_stream_mask_ext(void* stream, uint128_t mask) {
	char* stream_struct_base = *(char**)stream;
	struct stream_sm_mask* hw_mask = NULL;
	struct stream_sm_mask_v2* hw_mask_v2 = NULL;
	int ver;
	cuDriverGetVersion(&ver);
	switch (ver) {
#if __x86_64__
	case 8000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_8_0_MASK_OFF);
	case 9000:
	case 9010: {
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF);
		break;
	}
	case 9020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_2_MASK_OFF);
		break;
	case 10000:
	case 10010:
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_0_MASK_OFF);
		break;
	case 11000:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_0_MASK_OFF);
		break;
	case 11010:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_1_MASK_OFF);
		break;
	case 11020:
	case 11030:
	case 11040:
	case 11050:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_2_MASK_OFF);
		break;
	case 11060:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_6_MASK_OFF);
		break;
	case 11070:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_7_MASK_OFF);
		break;
	case 11080:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_8_MASK_OFF);
		break;
	case 12000:
	case 12010:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_0_MASK_OFF);
		break;
	case 12020:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_2_MASK_OFF);
		break;
	case 12030:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_3_MASK_OFF);
		break;
	case 12040:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_4_MASK_OFF);
		break;
	case 12050:
	case 12060:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_5_MASK_OFF);
		break;
	case 12070:
	case 12080:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_7_MASK_OFF);
		break;
#elif __aarch64__
	case 9000: {
		// Jetson TX2 offset is slightly different on CUDA 9.0.
		// Only compile the check into ARM64 builds.
		// TODO: Always verify Jetson-board-only on aarch64.
		int is_parker;
		const char* err_str;
		if ((is_parker = detect_parker_soc()) < 0) {
			cuGetErrorName(-is_parker, &err_str);
			abort(1, 0, "While performing platform-specific "
			            "compatibility checks for stream masking, "
			            "CUDA call failed with error '%s'.", err_str);
		}

		if (!is_parker)
			abort(1, 0, "Not supported on non-Jetson aarch64.");
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_9_0_MASK_OFF_JETSON);
		break;
	}
	case 10020:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_10_2_MASK_OFF_JETSON);
		break;
	case 11040:
		hw_mask = (struct stream_sm_mask*)(stream_struct_base + CU_11_4_MASK_OFF_JETSON);
		break;
	case 12020:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_2_MASK_OFF_JETSON);
		break;
	case 12040:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_4_MASK_OFF_JETSON);
		break;
	case 12050:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_5_MASK_OFF_JETSON);
		break;
	case 12060:
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_6_MASK_OFF_JETSON);
		break;
#endif
	}

	// For experimenting to determine the right mask offset, set the MASK_OFF
	// environment variable (positive and negative numbers are supported)
	char* mask_off_str = getenv("MASK_OFF");
	if (mask_off_str) {
		int off = atoi(mask_off_str);
		fprintf(stderr, "libsmctrl: Attempting offset %d on CUDA 12.2 base %#x "
				"(total off: %#x)\n", off, CU_12_2_MASK_OFF, CU_12_2_MASK_OFF + off);
		if (CU_12_2_MASK_OFF + off < 0)
			abort(1, 0, "Total offset cannot be less than 0! Aborting...");
		// +4 bytes to convert a mask found with this for use with hw_mask
		hw_mask_v2 = (void*)(stream_struct_base + CU_12_2_MASK_OFF + off);
	}

	// Mask layout changed with CUDA 12.0 to support large Hopper/Ada GPUs
	if (hw_mask) {
		hw_mask->upper = mask >> 32;
		hw_mask->lower = mask;
	} else if (hw_mask_v2) {
		hw_mask_v2->enabled = 1;
		hw_mask_v2->mask[0] = mask;
		hw_mask_v2->mask[1] = mask >> 32;
		hw_mask_v2->mask[2] = mask >> 64;
		hw_mask_v2->mask[3] = mask >> 96;
	} else {
		abort(1, 0, "Stream masking unsupported on this CUDA version (%d), and"
		            " no fallback MASK_OFF set!", ver);
	}
}


/*** TPC and GPU Informational Functions ***/

// Read an integer from a file in `/proc`
static int read_int_procfile(char* filename, uint64_t* out) {
	char f_data[18] = {0};
	size_t ret;
	int fd = open(filename, O_RDONLY);
	if (fd == -1)
		return errno;
	ret = read(fd, f_data, 18);
	if (ret == -1)
		return errno;
	close(fd);
	*out = strtoll(f_data, NULL, 16);
	return 0;
}

// We support up to 128 TPCs, up to 12 GPCs per GPU, and up to 16 GPUs.
#define MAX_GPCS 12
static uint64_t tpc_mask_per_gpc_per_dev[16][MAX_GPCS];
static uint128_t tpc_mask_per_gpc_per_dev_ext[16][MAX_GPCS];
// Output mask is vtpc-indexed (virtual TPC)
// Note that this function has to undo _both_ floorsweeping and ID remapping
int libsmctrl_get_gpc_info(uint32_t* num_enabled_gpcs, uint64_t** tpcs_for_gpc, int dev) {
	int err, i;
	uint128_t *tpcs_for_gpc_ext;
	if ((err = libsmctrl_get_gpc_info_ext(num_enabled_gpcs, &tpcs_for_gpc_ext, dev)))
		return err;
	for (i = 0; i < *num_enabled_gpcs; i++) {
		if ((tpcs_for_gpc_ext[i] & -1ull) != tpcs_for_gpc_ext[i])
			return ERANGE;
		tpc_mask_per_gpc_per_dev[dev][i] = (uint64_t)tpcs_for_gpc_ext[i];
	}
	*tpcs_for_gpc = tpc_mask_per_gpc_per_dev[dev];
	return 0;
}

int libsmctrl_get_gpc_info_ext(uint32_t* num_enabled_gpcs, uint128_t** tpcs_for_gpc, int dev) {
	uint32_t i, j, tpc_id, gpc_id, num_enabled_tpcs, num_configured_tpcs;
	uint64_t gpc_mask, num_tpc_per_gpc, max_gpcs, gpc_tpc_mask, gpc_tpc_config, total_read = 0;
	uint128_t tpc_bit;
	int err;
	char filename[100];
	*num_enabled_gpcs = 0;
	// Maximum number of GPCs supported for this chip
	snprintf(filename, 100, "/proc/gpu%d/num_gpcs", dev);
	if (err = read_int_procfile(filename, &max_gpcs)) {
		fprintf(stderr, "libsmctrl: nvdebug module must be loaded into kernel before "
				"using libsmctrl_get_*_info() functions\n");
		return err;
	}
	// TODO: handle arbitrary-size GPUs
	if (dev > 16 || max_gpcs > 12) {
		fprintf(stderr, "libsmctrl: GPU possibly too large for preallocated map!\n");
		return ERANGE;
	}
	// Set bit = disabled GPC
	snprintf(filename, 100, "/proc/gpu%d/gpc_mask", dev);
	if (err = read_int_procfile(filename, &gpc_mask))
		return err;
	// Determine the number of enabled TPCs
	snprintf(filename, 100, "/proc/gpu%d/num_tpc_per_gpc", dev);
	if (err = read_int_procfile(filename, &num_tpc_per_gpc))
		return err;
	// For each enabled GPC
	num_enabled_tpcs = 0;
	for (i = 0; i < max_gpcs; i++) {
		// Skip this GPC if disabled
		if ((1 << i) & gpc_mask)
			continue;
		(*num_enabled_gpcs)++;
		// Get the bitstring of TPCs disabled for this physical GPC
		// Set bit = disabled TPC
		snprintf(filename, 100, "/proc/gpu%d/gpc%d_tpc_mask", dev, i);
		if (err = read_int_procfile(filename, &gpc_tpc_mask))
			return err;
		// Bits greater than the max number of TPCs should be ignored, so only
		// keep the `num_tpc_per_gpc`-count number of lower bits.
		gpc_tpc_mask &= -1u >> (64 - num_tpc_per_gpc);
		// Number of enabled TPCs = max - number disabled
		num_enabled_tpcs += num_tpc_per_gpc - __builtin_popcountl(gpc_tpc_mask);
	}
	// Clear any previous mask
	for (i = 0; i < MAX_GPCS; i++)
		tpc_mask_per_gpc_per_dev_ext[dev][i] = 0;
	// For each enabled TPC
	for (tpc_id = 0; tpc_id < num_enabled_tpcs;) {
		// Pull mapping for the next set of 4 TPCs
		snprintf(filename, 100, "/proc/gpu%d/CWD_GPC_TPC_ID%d", dev, tpc_id / 4);
		if (err = read_int_procfile(filename, &gpc_tpc_config))
			return err;
		total_read += gpc_tpc_config;
		for (j = 0; j < 4 && tpc_id < num_enabled_tpcs; j++, tpc_id++) {
			// Set the bit for the current TPC
			tpc_bit = 1;
			tpc_bit <<= tpc_id;
			// Determine which GPC the current TPC is associated with
			// (upper 4 bits of each byte)
			gpc_id = (gpc_tpc_config >> (j*8 + 4) & 0xfu);
			// Save mapping
			tpc_mask_per_gpc_per_dev_ext[dev][gpc_id] |= tpc_bit;
		}
	}
	// Verify each TPC is configured
	tpc_bit = 0;
	for (i = 0; i < MAX_GPCS; i++)
		tpc_bit |= tpc_mask_per_gpc_per_dev_ext[dev][i];
	num_configured_tpcs = __builtin_popcountl(tpc_bit) + __builtin_popcountl(tpc_bit >> 64);
	if (num_configured_tpcs != num_enabled_tpcs) {
		fprintf(stderr, "libsmctrl: Found configuration for only %d TPCs when %d were expected.\n", num_configured_tpcs, num_enabled_tpcs);
		return EIO;
	}
	// Verify that the configuration was not always zero (indicates a powered-
	// -off GPU).
	if (total_read == 0) {
		fprintf(stderr, "libsmctrl: Is GPU on? Configuration registers are all zero.\n");
		return EIO;
	}

	*tpcs_for_gpc = tpc_mask_per_gpc_per_dev_ext[dev];
	return 0;
}

int libsmctrl_get_tpc_info(uint32_t* num_tpcs, int dev) {
	uint32_t num_gpcs;
	uint128_t* tpcs_per_gpc;
	int res;
	if (res = libsmctrl_get_gpc_info_ext(&num_gpcs, &tpcs_per_gpc, dev))
		return res;
	*num_tpcs = 0;
	for (int gpc = 0; gpc < num_gpcs; gpc++) {
		*num_tpcs += __builtin_popcountl(tpcs_per_gpc[gpc]);
		*num_tpcs += __builtin_popcountl(tpcs_per_gpc[gpc] >> 64);
	}
	return 0;
}

// @param dev Device index as understood by CUDA **can differ from nvdebug idx**
// This implementation is fragile, and could be incorrect for odd GPUs
int libsmctrl_get_tpc_info_cuda(uint32_t* num_tpcs, int cuda_dev) {
	int num_sms, major, minor, res = 0;
	const char* err_str;
	if (res = cuInit(0))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuda_dev))
		goto abort_cuda;
	if (res = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuda_dev))
		goto abort_cuda;
	// SM masking only works on sm_35+
	if (major < 3 || (major == 3 && minor < 5))
		return ENOTSUP;
	// Everything newer than Pascal (as of Hopper) has 2 SMs per TPC, as well
	// as the P100, which is uniquely sm_60
	int sms_per_tpc;
	if (major > 6 || (major == 6 && minor == 0))
		sms_per_tpc = 2;
	else
		sms_per_tpc = 1;
	// It looks like there may be some upcoming weirdness (TPCs with only one SM?)
	// with Hopper
	if (major >= 9)
		fprintf(stderr, "libsmctrl: WARNING, TPC masking is untested on Hopper,"
				" and will likely yield incorrect results! Proceed with caution.\n");
	*num_tpcs = num_sms/sms_per_tpc;
	return 0;
abort_cuda:
	cuGetErrorName(res, &err_str);
	fprintf(stderr, "libsmctrl: CUDA call failed due to %s. Failing with EIO...\n", err_str);
	return EIO;
}


/*** Private functions for nvtaskset and building as a libcuda.so.1 wrapper ***/

// Check if NVIDIA MPS is running, following the process that `strace` shows
// `nvidia-cuda-mps-control` to use. MPS is a prerequisite to co-running
// multiple GPU-using tasks without timeslicing.
bool libsmctrl_is_mps_running() {
	char *mps_pipe_dir;
	int mps_ctrl;
	struct sockaddr_un mps_ctrl_addr;
	mps_ctrl_addr.sun_family = AF_UNIX;
	const int yes = 1;

	if (!(mps_pipe_dir = getenv("CUDA_MPS_PIPE_DIRECTORY")))
		mps_pipe_dir = "/tmp/nvidia-mps";
	// Pipe names are limited to 108 characters long
	snprintf(mps_ctrl_addr.sun_path, 108, "%s/control", mps_pipe_dir);
	// This mirrors the process `nvidia-cuda-mps-control` uses to detect MPS
	if ((mps_ctrl = socket(AF_UNIX, SOCK_SEQPACKET, 0)) == -1)
		return false;
	if (setsockopt(mps_ctrl, SOL_SOCKET, SO_PASSCRED, &yes, sizeof(yes)) == -1)
		return false;
	if (connect(mps_ctrl, &mps_ctrl_addr, sizeof(struct sockaddr_un)) == -1)
		return false;
	close(mps_ctrl);
	return true;
}

// A variant of strtoul with support for 128-bit integers
uint128_t strtou128(const char *nptr, char **endptr, int base) {
	unsigned __int128 result = 0;
	if (base != 16)
		error(1, EINVAL, "strtou128 only supports base 16");
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

#ifdef LIBSMCTRL_WRAPPER
// The CUDA runtime library uses dlopen() to load CUDA functions from
// libcuda.so.1. Since we replace that with our wrapper library, we need to
// also redirect any attempted opens of that shared object to the actual
// shared library, which is linked to by libcuda.so.
void *dlopen(const char *filename, int flags) {
	if (filename && strcmp(filename, "libcuda.so") == 0) {
		fprintf(stderr, "redirecting dlopen of %s to libcuda.so\n", filename);
		// A GNU-only dlopen variant
		return dlmopen(LM_ID_BASE, "libcuda.so", flags);
	} else
		return dlmopen(LM_ID_BASE, filename, flags);
}

// Allow setting a default mask via an environment variable
// Also enables libsmctrl to be used on unmodified programs via setting:
//   LD_LIBRARY_PATH=libsmctrl LIBSMCTRL_MASK=<your mask> ./my_program
// Where "<your mask>" is replaced with a disable mask, optionally prefixed
// with a ~ to invert it (make it an enable mask).
__attribute__((constructor)) static void setup(void) {
	char *end, *mask_str;
	// If dynamic changes are disabled (due to an error) this variable is
	// permanently used to store the supreme mask, rather than the shared
	// memory segment.
	static uint128_t mask;
	bool invert = false;

	mask_str = getenv("LIBSMCTRL_MASK");

	// Assume no mask if unspecified
	if (!mask_str)
		mask_str = "0";

	if (*mask_str == '~') {
		invert = true;
		mask_str++;
	}

	mask = strtou128(mask_str, &end, 16);
	// Verify we were able to parse the whole string
	if (*end != '\0')
		abort(1, EINVAL, "Unable to apply default mask");

	if (invert)
		mask = ~mask;

	// Explictly set the number of channels (if unset), otherwise CUDA will only
	// use two with MPS (see paper for why that causes problems)
	if (setenv("CUDA_DEVICE_MAX_CONNECTIONS", "8", 0) == -1)
		abort(1, EINVAL, "Unable to configure environment");

	// Warn if a mask was specified but MPS isn't running
	if (mask && !libsmctrl_is_mps_running())
		fprintf(stderr, "libsmctrl-libcuda-wrapper: Warning: TPC mask set via LIBSMCTRL_MASK, but NVIDIA MPS is not running. CUDA programs will not co-run!\n");

	// Initialize CUDA and the interception callback
	setup_sm_control_callback();

	// Create shared memory region for the supreme mask such that nvtaskset
	// can read and modify it
	int fd = memfd_create("libsmctrl", MFD_CLOEXEC);
	if (fd == -1) {
		abort(0, errno, "Unable to create shared memory for dynamic partition changes. Dynamic changes disabled");
		g_supreme_sm_mask = &mask;
		return;
	}
	if (ftruncate(fd, 16) == -1) {
		abort(0, errno, "Unable to resize shared memory for dynamic partition changes. Dynamic changes disabled");
		g_supreme_sm_mask = &mask;
		return;
	}
	if ((g_supreme_sm_mask = mmap(NULL, 16, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED) {
		abort(0, errno, "Unable to map shared memory for dynamic partition changes. Dynamic changes disabled");
		g_supreme_sm_mask = &mask;
		return;
	}

	// Set the super-global mask which cannot be overwritten by any libsmctrl
	// API function.
	*g_supreme_sm_mask = mask;
}
#elif defined(LIBSMCTRL_STATIC)
// If this library is statically built into a program, and the libcuda.so.1
// wrapper is enabled, we force the staticlly linked version of the library
// to defer to the function implementations in the wrapper.
//
// Longer explanation:
// If the library has been dynamically linked into a program and the wrapper
// is in use, the loader will point both to the same set of symbols (since both
// will do a dynamic lookup at load-time, the global state at the top of this
// file uses global linkage, and will thus be in the dynamic symbol table, and
// each lookup will find the same copy.)
// Symbols from a staticlly linked library are not included in the dynamic
// symbol table, and thus can exist in duplicate of those in any shared
// library. This is a problem, since only one callback function, using one set
// of global variables can be registered with CUDA. We work around this by
// having our statically linked library use the functions from the wrapper or
// any shared library, if one such instance is loaded.
__attribute__((constructor)) static void setup(void) {
	// dlsym can only view the dynamic symbol tables, and so these lookups will
	// fail if neither the wrapper (libcuda.so.1) nor libsmctrl.so are loaded.
	// (That indicates that we should the static library implementations.)
	// These are a NOP on failure since they return NULL when not found.
	shared_set_next_mask = dlsym(RTLD_DEFAULT, "libsmctrl_set_next_mask");
	shared_set_global_mask = dlsym(RTLD_DEFAULT, "libsmctrl_set_global_mask");
}
#endif
