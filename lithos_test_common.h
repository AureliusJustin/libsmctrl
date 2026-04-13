/*
 * Shared helpers for LithOS runtime tests.
 *
 * Includes a minimal PTX module and common CUDA driver setup/teardown logic.
 */
#pragma once

#include <cuda.h>

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK_CU(expr) do { \
	CUresult _res = (expr); \
	if (_res != CUDA_SUCCESS) { \
		const char* _name = NULL; \
		const char* _desc = NULL; \
		cuGetErrorName(_res, &_name); \
		cuGetErrorString(_res, &_desc); \
		fprintf(stderr, "%s:%d: CUDA call failed: %s (%s)\n", __FILE__, __LINE__, _name ? _name : "<unknown>", _desc ? _desc : "<no description>"); \
		exit(1); \
	} \
} while (0)

static const char* lithos_test_ptx =
".version 6.4\n"
".target sm_52\n"
".address_size 64\n"
"\n"
".visible .entry write_one(\n"
"    .param .u64 out_ptr\n"
")\n"
"{\n"
"    .reg .pred %p;\n"
"    .reg .b64 %rd<2>;\n"
"    .reg .b32 %r<2>;\n"
"\n"
"    ld.param.u64 %rd1, [out_ptr];\n"
"    mov.u32 %r1, %tid.x;\n"
"    setp.ne.u32 %p, %r1, 0;\n"
"    @%p bra DONE;\n"
"    mov.u32 %r1, 1;\n"
"    st.global.u32 [%rd1], %r1;\n"
"DONE:\n"
"    ret;\n"
"}\n"
"\n"
".visible .entry write_smid(\n"
"    .param .u64 out_ptr\n"
")\n"
"{\n"
"    .reg .pred %p;\n"
"    .reg .b64 %rd<4>;\n"
"    .reg .b32 %r<4>;\n"
"\n"
"    ld.param.u64 %rd1, [out_ptr];\n"
"    mov.u32 %r1, %tid.x;\n"
"    setp.ne.u32 %p, %r1, 0;\n"
"    @%p bra SMID_DONE;\n"
"    mov.u32 %r2, %ctaid.x;\n"
"    mul.wide.u32 %rd2, %r2, 4;\n"
"    add.u64 %rd3, %rd1, %rd2;\n"
"    mov.u32 %r3, %smid;\n"
"    st.global.u32 [%rd3], %r3;\n"
"SMID_DONE:\n"
"    ret;\n"
"}\n";

static int count_unique_u32(uint32_t* arr, int len) {
	int uniq = 0;
	for (int i = 0; i < len; i++) {
		bool seen = false;
		for (int j = 0; j < i; j++) {
			if (arr[j] == arr[i]) {
				seen = true;
				break;
			}
		}
		if (!seen)
			uniq++;
	}
	return uniq;
}

static void lithos_test_setup(CUcontext* ctx, CUmodule* mod, CUfunction* fn, CUstream* stream) {
	int dev = 0;
	CHECK_CU(cuInit(0));
	CHECK_CU(cuDeviceGet(&dev, 0));
	CHECK_CU(cuCtxCreate(ctx, 0, dev));
	CHECK_CU(cuModuleLoadData(mod, lithos_test_ptx));
	CHECK_CU(cuModuleGetFunction(fn, *mod, "write_one"));
	CHECK_CU(cuStreamCreate(stream, CU_STREAM_DEFAULT));
}

static void lithos_test_teardown(CUcontext ctx, CUmodule mod, CUstream stream) {
	(void)cuStreamDestroy(stream);
	(void)cuModuleUnload(mod);
	(void)cuCtxDestroy(ctx);
}
