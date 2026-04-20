// Test deferred launch behavior using cuLaunchKernel packed-args API.
#include "lithos_test_common.h"

int main(void) {
	CUcontext ctx;
	CUmodule mod;
	CUfunction fn;
	CUstream stream;
	CUdeviceptr d_out;
	uint32_t h_out = 0;
	CUdeviceptr arg_ptr;
	size_t arg_size = sizeof(arg_ptr);

	lithos_test_setup(&ctx, &mod, &fn, &stream);
	CHECK_CU(cuMemAlloc(&d_out, sizeof(uint32_t)));
	CHECK_CU(cuMemsetD32(d_out, 0, 1));

	arg_ptr = d_out;
	void* extra[] = {
		CU_LAUNCH_PARAM_BUFFER_POINTER, &arg_ptr,
		CU_LAUNCH_PARAM_BUFFER_SIZE, &arg_size,
		CU_LAUNCH_PARAM_END
	};
	CHECK_CU(cuLaunchKernel(fn,
	                        1, 1, 1,
	                        1, 1, 1,
	                        0,
	                        stream,
	                        NULL,
	                        extra));
	CHECK_CU(cuStreamSynchronize(stream));
	CHECK_CU(cuMemcpyDtoH(&h_out, d_out, sizeof(uint32_t)));

	if (h_out != 1) {
		fprintf(stderr, "packed launch test failed: expected 1, got %u\n", h_out);
		return 2;
	}

	CHECK_CU(cuMemFree(d_out));
	lithos_test_teardown(ctx, mod, stream);
	printf("lithos_test_launch_packed: pass\n");
	return 0;
}
