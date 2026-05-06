# Set this if CUDA is installed in a different location
CUDA ?= /usr/local/cuda
# Note that CXX and CC are predefined as g++ and cc (respectively) by Make
NVCC ?= $(CUDA)/bin/nvcc
# Everything has to have -lcuda, as it's needed for libsmctrl
LDFLAGS := -ldl -lcuda -I$(CUDA)/include -L$(CUDA)/lib64
ARCH = $(shell $(CC) -dumpmachine)
CFLAGS := -Wall -Wno-parentheses -I. -Ilithos -Ilithos/tests
PYTHON ?= python3

.PHONY: clean tests all install remove run_tests lithos_tests run_lithos_tests

# ----- Main Library -----
libsmctrl.so: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ -fPIC $(CFLAGS) $(LDFLAGS)

# -fPIC is needed even if built as a static library, in case we are linked into
# another shared library
libsmctrl.a: libsmctrl.c libsmctrl.h
	$(CC) $< -c -o libsmctrl.o -fPIC -DLIBSMCTRL_STATIC $(CFLAGS) $(LDFLAGS)
	ar rcs $@ libsmctrl.o

# ----- CUDA Wrapper -----
libcuda.so.1: libsmctrl.c libsmctrl.h lithos/lithos_runtime.c lithos/lithos_runtime.h
	$(CC) libsmctrl.c lithos/lithos_runtime.c -shared -o $@ -fPIC -DLIBSMCTRL_WRAPPER $(CFLAGS) $(LDFLAGS) -pthread
	@# Replace dynamic symbol dependency on libcuda.so.1 with libcuda.so
	@# Could also be done via patchelf --replace-needed libcuda.so.1 libcuda.so libcuda.so.1
	sed -i "s/libcuda.so.1\x00/libcuda.so\x00\x00\x00/g" libcuda.so.1

# ----- Utilities -----
# Use static linking with tests to avoid LD_LIBRARY_PATH issues
nvtaskset: nvtaskset.c libsmctrl.so libsmctrl.a
	$(CC) $@.c -o $@ -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

# ----- LithOS Interposer Payload -----
liblithos_preload.so: libsmctrl.c libsmctrl.h lithos/lithos_runtime.c lithos/lithos_runtime.h
	$(CC) libsmctrl.c lithos/lithos_runtime.c -shared -o $@ -fPIC -DLIBSMCTRL_WRAPPER $(CFLAGS) $(LDFLAGS) -pthread

lithosd: lithos/lithosd.c lithos/lithos_ipc.h
	$(CC) $< -o $@ -g $(CFLAGS)

libsmctrl_test_gpc_info: libsmctrl_test_gpc_info.c libsmctrl.a testbench.h
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

lithos_test_launch_kernelparams: lithos/tests/lithos_test_launch_kernelparams.c lithos/tests/lithos_test_common.h liblithos_preload.so
	$(CC) $< -o $@ -g $(CFLAGS) $(LDFLAGS)

lithos_test_launch_packed: lithos/tests/lithos_test_launch_packed.c lithos/tests/lithos_test_common.h liblithos_preload.so
	$(CC) $< -o $@ -g $(CFLAGS) $(LDFLAGS)

lithos_test_scheduler_quota: lithos/tests/lithos_test_scheduler_quota.c lithos/tests/lithos_test_common.h libsmctrl.a liblithos_preload.so
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

lithos_test_kernelparams_scheduler: lithos/tests/lithos_test_kernelparams_scheduler.c lithos/tests/lithos_test_common.h libsmctrl.a liblithos_preload.so
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

lithos_test_arbitrary_app: lithos/tests/lithos_test_arbitrary_app.c lithos/tests/lithos_test_common.h libsmctrl.a liblithos_preload.so
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

lithos_test_cuda_graph_capture_replay: lithos/tests/lithos_test_cuda_graph_capture_replay.c lithos/tests/lithos_test_common.h liblithos_preload.so
	$(CC) $< -o $@ -g $(CFLAGS) $(LDFLAGS)

run_lithos_framework_smoke: lithosd lithos/tests/lithos_test_framework_smoke.py lithos/tests/lithos_test_torch_large_mm.py lithos/tests/lithos_test_jax_large_mm.py
	@SOCK=/tmp/lithosd.sock; \
	./lithosd $$SOCK 54 >/tmp/lithosd.log 2>&1 & D=$$!; \
	trap 'kill $$D 2>/dev/null; wait $$D 2>/dev/null; rm -f $$SOCK' EXIT INT TERM; \
	LIBSMCTRL_LITHOS_ENABLE=1 LIBSMCTRL_LITHOS_GLOBAL_SCHED_ENABLE=1 LIBSMCTRL_LITHOSD_SOCK=$$SOCK LD_PRELOAD=$(PWD)/liblithos_preload.so $(PYTHON) ./lithos/tests/lithos_test_framework_smoke.py

# ----- Tests -----
libsmctrl_test_mask_shared.o: libsmctrl_test_mask_shared.cu testbench.h
	$(NVCC) -ccbin $(CXX) $< -c -g

libsmctrl_test_supreme_mask: libsmctrl_test_supreme_mask.c libsmctrl.a libsmctrl_test_mask_shared.o libcuda.so.1 nvtaskset
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_global_mask: libsmctrl_test_global_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask: libsmctrl_test_stream_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_stream_mask_override: libsmctrl_test_stream_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask: libsmctrl_test_next_mask.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

libsmctrl_test_next_mask_override: libsmctrl_test_next_mask_override.c libsmctrl.a libsmctrl_test_mask_shared.o
	$(NVCC) -ccbin $(CXX) $@.c -o $@ libsmctrl_test_mask_shared.o -g -L. -l:libsmctrl.a $(LDFLAGS)

tests: libsmctrl_test_gpc_info libsmctrl_test_supreme_mask \
       libsmctrl_test_global_mask libsmctrl_test_stream_mask \
       libsmctrl_test_stream_mask_override libsmctrl_test_next_mask \
       libsmctrl_test_next_mask_override

lithos_tests: lithos_test_launch_kernelparams lithos_test_launch_packed lithos_test_scheduler_quota lithos_test_kernelparams_scheduler lithos_test_arbitrary_app lithos_test_cuda_graph_capture_replay

all: libsmctrl.so libcuda.so.1 nvtaskset tests

clean:
	rm -f libsmctrl.so libsmctrl.o libsmctrl.a libsmctrl_test_gpc_info \
	      libsmctrl_test_mask_shared.o libsmctrl_test_supreme_mask \
	      libsmctrl_test_global_mask \
	      libsmctrl_test_stream_mask libsmctrl_test_stream_mask_override \
	      libsmctrl_test_next_mask libsmctrl_test_next_mask_override \
	      lithos_test_launch_kernelparams lithos_test_launch_packed \
	      lithos_test_scheduler_quota lithos_test_kernelparams_scheduler \
	      lithos_test_arbitrary_app \
	      lithosd \
	      nvtaskset libcuda.so.1 liblithos_preload.so

# On L4T (Linux4Tegra), the paths are different, and there may be multiple copies of libcuda.so.1
install: libcuda.so.1
	@set -e -x; \
	for DIR in /usr/lib/$(ARCH) /usr/local/cuda-*.*/compat /usr/lib/$(ARCH)/nvidia; do \
		if [ ! -d $$DIR ]; then continue; fi; \
		# Check that CUDA is installed in this location \
		if [ ! -f $$DIR/libcuda.so.*.* ]; then continue; fi; \
		# Change libcuda.so link to bypass libcuda.so.1 \
		sudo ln -sf $$DIR/libcuda.so.*.* $$DIR/libcuda.so; \
		# Remove libcuda.so.1 symlink \
		sudo rm $$DIR/libcuda.so.1; \
		# Install wrapper as libcuda.so.1 \
		sudo cp libcuda.so.1 $$DIR/libcuda.so.1; \
	done \
	# Special handling for L4T \
	if [ -d /usr/lib/$(ARCH)/nvidia ]; then sudo ln -sf nvidia/libcuda.so.1 /usr/lib/$(ARCH)/libcuda.so.1; fi

remove:
	@set -e -x; \
	for DIR in /usr/lib/$(ARCH) /usr/local/cuda-*.*/compat /usr/lib/$(ARCH)/nvidia; do \
		if [ ! -d $$DIR ]; then continue; fi; \
		# Check that CUDA is installed in this location \
		if [ ! -f $$DIR/libcuda.so.*.* ]; then continue; fi; \
		# Test that our library in installed here \
		if [ -L $$DIR/libcuda.so.1 ]; then continue; fi; \
		# Overwrite install with original symlinks \
		sudo ln -sf libcuda.so.1 $$DIR/libcuda.so; \
		sudo ln -sf $$DIR/libcuda.so.*.* $$DIR/libcuda.so.1; \
	done

run_tests: tests
	./libsmctrl_test_global_mask
	./libsmctrl_test_next_mask
	./libsmctrl_test_stream_mask
	./libsmctrl_test_next_mask_override
	./libsmctrl_test_stream_mask_override
	@# Must set LD_LIBRARY_PATH in case make install has not been run
	LD_LIBRARY_PATH=. ./libsmctrl_test_supreme_mask
	./libsmctrl_test_gpc_info
	@ echo "All tests passed!"

run_lithos_tests: lithos_tests
	@# Pass-through behavior
	LD_PRELOAD=$(PWD)/liblithos_preload.so ./lithos_test_launch_kernelparams
	@# Deferred queue behavior for packed launch args
	LIBSMCTRL_LITHOS_ENABLE=1 LD_PRELOAD=$(PWD)/liblithos_preload.so ./lithos_test_launch_packed
	@# Baseline scheduler: 1 TPC quota on first stream
	LIBSMCTRL_LITHOS_ENABLE=1 LIBSMCTRL_LITHOS_SCHED_ENABLE=1 LIBSMCTRL_LITHOS_TPC_QUOTAS=1 LD_PRELOAD=$(PWD)/liblithos_preload.so ./lithos_test_scheduler_quota
	@# KernelParams launches should also be deferred/scheduled now
	LIBSMCTRL_LITHOS_ENABLE=1 LIBSMCTRL_LITHOS_SCHED_ENABLE=1 LIBSMCTRL_LITHOS_TPC_QUOTAS=1 LD_PRELOAD=$(PWD)/liblithos_preload.so ./lithos_test_kernelparams_scheduler
	@# CUDA Graph capture + replay should run cleanly under interposition
	LIBSMCTRL_LITHOS_ENABLE=1 LIBSMCTRL_LITHOS_SCHED_ENABLE=1 LIBSMCTRL_LITHOS_TPC_QUOTAS=1 LD_PRELOAD=$(PWD)/liblithos_preload.so ./lithos_test_cuda_graph_capture_replay
	@# Framework smoke under interposition + daemon (Torch + JAX)
	$(MAKE) PYTHON=$(PYTHON) run_lithos_framework_smoke
	@ echo "All LithOS tests passed!"
