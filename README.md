# libsmctrl: Quick & Easy Hardware Compute Partitioning on NVIDIA GPUs

This library was developed as part of the following paper:

_J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs", Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium, pp. 54-66, May 2023._

Please cite this paper in any work which leverages our library. Here's the BibTeX entry:
```
@inproceedings{bakita2023hardware,
  title={Hardware Compute Partitioning on {NVIDIA} {GPUs}},
  author={Bakita, Joshua and Anderson, James H},
  booktitle={Proceedings of the 29th IEEE Real-Time and Embedded Technology and Applications Symposium},
  year={2023},
  month={May},
  pages={54--66},
  doi={10.1109/RTAS58335.2023.00012},
  _series={RTAS}
}
```

The ability for `libsmctrl` to work on unmodified tasks was developed as part of a follow-up paper:

_J. Bakita and J. H. Anderson, "Hardware Compute Partitioning on NVIDIA GPUs for Composable Systems", Proceedings of the 37th Euromicro Conference on Real-Time Systems, pp. 18:1-18:24, July 2025._

Please cite this paper in any work which uses this for partitioning unmodified tasks. Here's the BibTeX entry:
```
@inproceedings{bakita2025hardware,
  title={Hardware Compute Partitioning on {NVIDIA} {GPUs} for Composable Systems},
  author={Bakita, Joshua and Anderson, James H},
  booktitle={Proceedings of the 37th Euromicro Conference on Real-Time Systems},
  year={2025},
  month={July},
  pages={18:1--18:24},
  doi={10.1109/ECRTS.2025.18},
  _series={ECRTS}
}
```

Please see [the first paper](https://www.cs.unc.edu/~jbakita/rtas23.pdf), [the second paper](https://www.cs.unc.edu/~jbakita/ecrts25.pdf) and `libsmctrl.h` for details and examples of how to use this library.
We strongly encourage consulting those resources first; the below comments serve merely as an appendum.

## Run-time Dependencies
`libcuda.so`, which is automatically installed by the NVIDIA GPU driver.

(Technically `libdl` is also required, but this should never need to be manually installed. This is a dependency of CUDA, and is also part of the GNU C Standard Library starting with version 2.34.)

## Building
To build, ensure that you have `gcc` installed and access to the CUDA SDK including `nvcc`. Then run:
```
make libsmctrl.a
```

If you see errors about CUDA headers or libraries not being found, your CUDA installation may be in a non-standard location.
Correct this error by explictly specifying the location of the CUDA install `make`, e.g.:
```
make CUDA=/playpen/jbakita/CUDA/cuda-archive/cuda-10.2/ libsmctrl.a
```

For binary backwards-compatibility to old versions of the NVIDIA GPU driver, we recommend building with an old version of the CUDA SDK.
For example, by building against CUDA 10.2, the binary will be compatible with any version of the NVIDIA GPU driver newer than 440.36 (Nov 2019), but by building against CUDA 8.0, the binary will be compatible with any version of the NVIDIA GPU driver newer that 375.26 (Dec 2016).

Older versions of `nvcc` may require you to use an older version of `g++`.
This can be explictly specified via the `CXX` variable, e.g.:
```
make CUDA=/playpen/jbakita/CUDA/cuda-archive/cuda-8.0/ CXX=g++-5 libsmctrl.a
```

`libsmctrl` supports being built as a shared library.
This will require you to distribute `libsmctrl.so` with your compiled program.
If you do not know what a shared library is, or why you would need to specify the path to `libsmctrl.so` in `LD_LIBRARY_PATH`, do not do this.
To build as a shared library, replace `libsmctrl.a` with `libsmctrl.so` in the above commands.

## Linking in Your Application
If you have cloned and built `libsmctrl` in the folder `/playpen/libsmctrl` (replace this with the location you use):

1. Add `-I/playpen/libsmctrl` to your compiler command (this allows `#include <libsmctrl.h>` in your C/C++ files).
2. Add `-lsmctrl` to your linker command (this allows the linker to resolve the `libsmctrl` functions you use to the implementations in `libsmctrl.a` or `libsmctrl.so`).
3. Add `-L/playpen/libsmctrl` to your linker command (this allows the linker to find `libsmctrl.a` or `libsmctrl.so`).
4. (If not already included) add `-lcuda` to your linker command (this links against the CUDA driver library).

Note that if you have compiled both `libsmctrl.a` (the static library) and `libsmctrl.so` (the shared library), most compilers will prefer the shared library.
To statically link against `libsmctrl.a`, delete `libsmctrl.so`.

For example, if you have a CUDA program written in `benchmark.cu` and have built `libsmctrl`, you can compile and link against `libsmctrl` via the following command:
```
nvcc benchmark.cu -o benchmark -I/playpen/libsmctl -lsmctrl -lcuda -L/playpen/libsmctrl
```
The resultant `benchmark` binary should be portable to any system with an equivalent or newer version of the NVIDIA GPU driver installed.

## Use Without Application Modification
As an alternative to modifying your application, `libsmctrl` can be installed system-wide, and partitions for each application can be set via the `nvtaskset` tool.
The `nvtaskset` tool works very similarly to the Linux CPU-affinity-setting tool `taskset`.

To install `libsmctrl` system-wide, such that all CUDA-using applications automatically load it, run:
```
make libcuda.so.1 install
```
Or, if you do not want to modify any system-wide state, and only want `libsmctrl` loaded as part of anything run from this console:
```
make libcuda.so.1
export LD_LIBRARY_PATH=$(pwd)
```
(This works because CUDA is always dynamically loaded from `libcuda.so.1`, and `lbsmctrl` creates a "fake" `libcuda.so.1` in this directory that wraps CUDA.
 Setting `LD_LIBRARY_PATH` ensures that the wrapped version is the first one loaded.
 The only difference with running `make install` is that it copies our "fake" `libcuda.so.1` to a location where the loader will automatically find it.)

And then to start an application within a specific TPC partition, e.g., the first 10 TPCs:
```
./nvtaskset -t 0-9 my_program my_args
```
Note that this will automatically start NVIDIA MPS, which is a prerequisite to co-run tasks on NVIDIA GPUs without timeslicing.

And to change the TPCs available for a process ID 1234 to to the first 10 TPCs:
```
./nvtaskset -tp 0-9 1234
```

Or, to change a process of ID 1234 to only run on GPC 3:
```
./nvtaskset -gp 3 1234
```

To remove the system-wide installation of `libsmctrl`, run:
```
make remove
```

## Run Tests

To run them all:
```
make run_tests
```

If you prefer to run them individually, to test partitioning:
```
make tests
./libsmctrl_test_global_mask
./libsmctrl_test_stream_mask
./libsmctrl_test_next_mask
```

To test that high-granularity masks override low-granularity ones:
```
make tests
./libsmctrl_test_stream_mask_override
./libsmctrl_test_next_mask_override
```

To test that `nvtaskset` can dynamically change the mask of a running program:
```
make libsmctrl_test_supreme_mask
./libsmctrl_test_supreme_mask
```

To test that TPC to GPC mappings can be obtained (if `nvdebug` has been installed):
```
make libsmctrl_test_gpc_info
./libsmctrl_test_gpc_info
```

The `CUDA_VISIBLE_DEVICES` environment variable can be set to run any of the partitioning tests on a different GPU.

## Supported GPUs

#### Known Working

- NVIDIA GPUs from compute capability 3.5 through 8.9, including embedded "Jetson" GPUs
- CUDA 6.5 through 12.8
- `x86_64` and Jetson `aarch64` platforms

#### Known Issues

- `global_mask` and `next_mask` cannot disable TPCs with IDs above 128
    - Only relevant on GPUs with over 128 TPCs, such as the RTX 6000 Ada
- Untested on non-Jetson `aarch64` platforms
- Untested on CUDA 11.8, 12.0, and 12.1 on Jetson `aarch64`
- Mask bit indexes do not directly correlate to software-visible TPC/SM IDs in V4 TMD/QMDs (Hopper+; compute capability 9.0). The mask bit indexes instead appear to correspond to on-chip-units, including disabled ones; i.e. the set of pre-SM-ID-remapping and pre-floorsweeping TPCs
- Tests fail when NVIDIA MPS is enabled on Volta-generation and newer GPUs, as non-physical SM IDs are returned by `%%smid` register in this configuration (see US Patent 11,307,903 by NVIDIA)

## Important Limitations

1. Only supports partitioning _within_ a single GPU context.
   At time of writing, it is challenging to impossible to share a GPU context across multiple CPU address spaces.
   The implication is that your applications must first be combined together into a single CPU process.
2. No aspect of this system prevents implicit synchronization on the GPU.
   See prior work, particularly that of Amert et al. (perhaps the CUPiD^RT paper), for ways to avoid this.

## Porting Stream Masking to Newer CUDA Versions

Build the tests with `make tests`. And then run the following:
```
for (( i=0; $?!=0; i+=8 )); do MASK_OFF=$i ./libsmctrl_test_stream_mask; done
```

How this works:

1. If `MASK_OFF` is set, `libsmctrl` applies this as a byte offset to a base address for the location
   of the SM mask fields in CUDA's stream data structure.
  - That base address is the one for CUDA 12.2 at time of writing.
2. The stream masking test is run.
3. If the test succeeded (returned zero) the loop aborts, otherwise it increments the offset to attempt and repeats.

Once this loop aborts, take the found offset and add it into the switch statement for the appropriate CUDA version and CPU architecture.

If the loop hangs (e.g. at offset 40), terminate and restart the loop with `i` initialized past the offset that hung (e.g. at offset 48).
