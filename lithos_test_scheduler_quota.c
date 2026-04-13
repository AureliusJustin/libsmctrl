// Test Phase 3 baseline scheduler by enforcing a 1-TPC quota and verifying
// confined SM usage for a queued packed-args launch.
#include "lithos_test_common.h"
#include "libsmctrl.h"

int main(void) {
	CUcontext ctx;
	CUmodule mod;
	CUfunction fn;
	CUstream stream;
	CUdeviceptr d_smids;
	uint32_t h_smids[256] = {0};
	CUdeviceptr arg_ptr;
	size_t arg_size = sizeof(arg_ptr);
	int num_sms = 0;
	uint32_t num_tpcs = 0;
	int sms_per_tpc;
	int uniq;

	lithos_test_setup(&ctx, &mod, &fn, &stream);
	CHECK_CU(cuModuleGetFunction(&fn, mod, "write_smid"));
	CHECK_CU(cuMemAlloc(&d_smids, sizeof(h_smids)));
	CHECK_CU(cuMemsetD8(d_smids, 0, sizeof(h_smids)));

	CHECK_CU(cuDeviceGetAttribute(&num_sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0));
	if (libsmctrl_get_tpc_info_cuda(&num_tpcs, 0) != 0 || num_tpcs == 0) {
		fprintf(stderr, "Unable to query TPC count for scheduler test\n");
		return 2;
	}
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
	                        256, 1, 1,
	                        64, 1, 1,
	                        0,
	                        stream,
	                        NULL,
	                        extra));
	CHECK_CU(cuStreamSynchronize(stream));
	CHECK_CU(cuMemcpyDtoH(h_smids, d_smids, sizeof(h_smids)));

	uniq = count_unique_u32(h_smids, 256);
	if (uniq > sms_per_tpc) {
		fprintf(stderr,
		        "scheduler quota test failed: expected <= %d unique SMs for 1-TPC quota, got %d\n",
		        sms_per_tpc,
		        uniq);
		return 3;
	}

	CHECK_CU(cuMemFree(d_smids));
	lithos_test_teardown(ctx, mod, stream);
	printf("lithos_test_scheduler_quota: pass (unique_sms=%d, sms_per_tpc=%d)\n", uniq, sms_per_tpc);
	return 0;
}
