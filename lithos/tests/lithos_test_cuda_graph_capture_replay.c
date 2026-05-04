// Test CUDA Graph capture + replay behavior under LithOS interposition.
#include "lithos_test_common.h"

int main(void) {
	CUcontext ctx;
	CUmodule mod;
	CUfunction fn_dummy;
	CUfunction fn_inc;
	CUstream stream;
	CUdeviceptr d_counter;
	CUgraph graph;
	CUgraphExec graph_exec;
	uint32_t h_counter = 0;
	const int replay_count = 8;

	lithos_test_setup(&ctx, &mod, &fn_dummy, &stream);
	CHECK_CU(cuModuleGetFunction(&fn_inc, mod, "inc_counter"));

	CHECK_CU(cuMemAlloc(&d_counter, sizeof(uint32_t)));
	CHECK_CU(cuMemsetD32(d_counter, 0, 1));

	void* params[] = { &d_counter };
	CHECK_CU(cuStreamBeginCapture(stream, CU_STREAM_CAPTURE_MODE_GLOBAL));
	CHECK_CU(cuLaunchKernel(fn_inc,
	                        1, 1, 1,
	                        1, 1, 1,
	                        0,
	                        stream,
	                        params,
	                        NULL));
	CHECK_CU(cuStreamEndCapture(stream, &graph));

	CHECK_CU(cuGraphInstantiate(&graph_exec, graph, 0));
	for (int i = 0; i < replay_count; i++)
		CHECK_CU(cuGraphLaunch(graph_exec, stream));
	CHECK_CU(cuStreamSynchronize(stream));
	CHECK_CU(cuMemcpyDtoH(&h_counter, d_counter, sizeof(uint32_t)));

	if (h_counter != (uint32_t)replay_count) {
		fprintf(stderr,
		        "cuda graph capture/replay test failed: expected %d, got %u\n",
		        replay_count,
		        h_counter);
		return 2;
	}

	CHECK_CU(cuGraphExecDestroy(graph_exec));
	CHECK_CU(cuGraphDestroy(graph));
	CHECK_CU(cuMemFree(d_counter));
	lithos_test_teardown(ctx, mod, stream);
	printf("lithos_test_cuda_graph_capture_replay: pass\n");
	return 0;
}
