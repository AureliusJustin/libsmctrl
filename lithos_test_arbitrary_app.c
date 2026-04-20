/*
 * Terminal-friendly arbitrary CUDA Driver API app for LithOS testing.
 *
 * This app launches one real CUDA kernel via packed launch arguments so the
 * interposed deferred-queue path is exercised. It prints one parseable result
 * line for external orchestration tools.
 */
#include "lithos_test_common.h"
#include "libsmctrl.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(void) {
	CUcontext ctx;
	CUmodule mod;
	CUfunction fn;
	CUstream stream;
	CUdeviceptr d_smids;
	uint32_t h_smids[128] = {0};
	CUdeviceptr arg_ptr;
	size_t arg_size = sizeof(arg_ptr);
	int num_sms = 0;
	uint32_t num_tpcs = 0;
	int sms_per_tpc = 1;
	int uniq;
	uint32_t min_smid;
	uint32_t max_smid;

	lithos_test_setup(&ctx, &mod, &fn, &stream);
	CHECK_CU(cuModuleGetFunction(&fn, mod, "write_smid"));
	CHECK_CU(cuMemAlloc(&d_smids, sizeof(h_smids)));
	CHECK_CU(cuMemsetD8(d_smids, 0, sizeof(h_smids)));

	CHECK_CU(cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0));
	if (libsmctrl_get_tpc_info_cuda(&num_tpcs, 0) == 0 && num_tpcs > 0)
		sms_per_tpc = num_sms / (int)num_tpcs;
	if (sms_per_tpc < 1)
		sms_per_tpc = 1;

	arg_ptr = d_smids;
	void* extra[] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER, &arg_ptr,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
		CU_LAUNCH_PARAM_END
	};

	CHECK_CU(cuLaunchKernel(fn,
	                        128, 1, 1,
	                        64, 1, 1,
	                        0,
	                        stream,
	                        NULL,
	                        extra));
	CHECK_CU(cuStreamSynchronize(stream));
	CHECK_CU(cuMemcpyDtoH(h_smids, d_smids, sizeof(h_smids)));

	uniq = count_unique_u32(h_smids, 128);
	min_smid = h_smids[0];
	max_smid = h_smids[0];
	for (int i = 1; i < 128; i++) {
		if (h_smids[i] < min_smid)
			min_smid = h_smids[i];
		if (h_smids[i] > max_smid)
			max_smid = h_smids[i];
	}

	printf("RESULT pid=%d uniq=%d sms_per_tpc=%d min=%u max=%u\n",
	       (int)getpid(),
	       uniq,
	       sms_per_tpc,
	       min_smid,
	       max_smid);
	fflush(stdout);

	CHECK_CU(cuMemFree(d_smids));
	lithos_test_teardown(ctx, mod, stream);
	return 0;
}
