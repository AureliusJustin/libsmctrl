// Test pass-through launch behavior using cuLaunchKernel kernelParams API.
#include "lithos_test_common.h"

int main(void) {
	CUcontext ctx;
	CUmodule mod;
	CUfunction fn;
	CUstream stream;
	CUdeviceptr d_out;
	uint32_t h_out = 0;

	lithos_test_setup(&ctx, &mod, &fn, &stream);
	CHECK_CU(cuMemAlloc(&d_out, sizeof(uint32_t)));
	CHECK_CU(cuMemsetD32(d_out, 0, 1));

	void* params[] = { &d_out };
	CHECK_CU(cuLaunchKernel(fn,
	                        1, 1, 1,
	                        1, 1, 1,
	                        0,
	                        stream,
	                        params,
	                        NULL));
	CHECK_CU(cuStreamSynchronize(stream));
	CHECK_CU(cuMemcpyDtoH(&h_out, d_out, sizeof(uint32_t)));

	if (h_out != 1) {
		fprintf(stderr, "kernelParams launch test failed: expected 1, got %u\n", h_out);
		return 2;
	}

	CHECK_CU(cuMemFree(d_out));
	lithos_test_teardown(ctx, mod, stream);
	printf("lithos_test_launch_kernelparams: pass\n");
	return 0;
}
