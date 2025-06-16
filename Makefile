# Set this if CUDA is installed in a different location
CUDA ?= /usr/local/cuda
# Note that CXX and CC are predefined as g++ and cc (respectively) by Make
NVCC ?= $(CUDA)/bin/nvcc
# Everything has to have -lcuda, as it's needed for libsmctrl
LDFLAGS := -ldl -lcuda -I$(CUDA)/include -L$(CUDA)/lib64
ARCH = $(shell $(CC) -dumpmachine)
CFLAGS := -Wall -Wno-parentheses

.PHONY: clean tests all install remove run_tests

# ----- Main Library -----
libsmctrl.so: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ -fPIC $(CFLAGS) $(LDFLAGS)

# -fPIC is needed even if built as a static library, in case we are linked into
# another shared library
libsmctrl.a: libsmctrl.c libsmctrl.h
	$(CC) $< -c -o libsmctrl.o -fPIC -DLIBSMCTRL_STATIC $(CFLAGS) $(LDFLAGS)
	ar rcs $@ libsmctrl.o

# ----- CUDA Wrapper -----
libcuda.so.1: libsmctrl.c libsmctrl.h
	$(CC) $< -shared -o $@ -fPIC -DLIBSMCTRL_WRAPPER $(CFLAGS) $(LDFLAGS)
	patchelf libcuda.so.1 --add-needed libcuda.so

# ----- Utilities -----
# Use static linking with tests to avoid LD_LIBRARY_PATH issues
nvtaskset: nvtaskset.c libsmctrl.so libsmctrl.a
	$(CC) $@.c -o $@ -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

libsmctrl_test_gpc_info: libsmctrl_test_gpc_info.c libsmctrl.a testbench.h
	$(CC) $< -o $@ -g -L. -l:libsmctrl.a $(CFLAGS) $(LDFLAGS)

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

all: libsmctrl.so libcuda.so.1 nvtaskset tests

clean:
	rm -f libsmctrl.so libsmctrl.o libsmctrl.a libsmctrl_test_gpc_info \
	      libsmctrl_test_mask_shared.o libsmctrl_test_supreme_mask \
	      libsmctrl_test_global_mask \
	      libsmctrl_test_stream_mask libsmctrl_test_stream_mask_override \
	      libsmctrl_test_next_mask libsmctrl_test_next_mask_override \
	      nvtaskset libcuda.so.1

install: libcuda.so.1
	@# Check that CUDA is installed first
	test -f /lib/$(ARCH)/libcuda.so.*.*
	@# Change libcuda.so link to bypass libcuda.so.1
	sudo ln -sf /lib/$(ARCH)/libcuda.so.*.* /lib/$(ARCH)/libcuda.so
	@# Remove libcuda.so.1 symlink
	sudo rm /lib/$(ARCH)/libcuda.so.1
	@# Install wrapper as libcuda.so.1
	sudo cp libcuda.so.1 /lib/$(ARCH)/libcuda.so.1

remove:
	@# Test that our library in installed first
	test ! -L /lib/$(ARCH)/libcuda.so.1
	@# Overwrite install with original symlinks
	sudo ln -sf libcuda.so.1 /lib/$(ARCH)/libcuda.so
	sudo ln -sf /lib/$(ARCH)/libcuda.so.*.* /lib/$(ARCH)/libcuda.so.1

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
