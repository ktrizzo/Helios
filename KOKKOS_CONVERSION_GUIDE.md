# Helios CUDA to Kokkos Conversion Guide

## Project Overview
This guide tracks the ongoing conversion of Helios GPU-accelerated plugins from CUDA to Kokkos for GPU-agnostic portability.

## Motivation
- **GPU Vendor Independence**: Support NVIDIA (CUDA), AMD (HIP), Intel (SYCL)
- **Future-Proofing**: Not tied to single GPU vendor
- **Performance Portability**: Single source code, multiple backends
- **CPU Fallback**: OpenMP and Serial backends for systems without GPUs

## Conversion Status

### ✅ Completed Plugins
1. **EnergyBalance** - Converted 2025-11-07
   - 1 kernel converted
   - 517 lines of CUDA code
   - See: `plugins/energybalance/KOKKOS_CONVERSION.md`

### 🔄 In Progress
None currently

### 📋 Planned Conversions
1. **VoxelIntersection** (361 lines, 1 kernel) - Simplest remaining
2. **AerialLiDAR** (1,644 lines, 4 kernels) - Medium complexity
3. **LiDAR** (5,051 lines, 6 kernels) - Largest plugin
4. **Radiation** (2,041 lines, 39+ OptiX kernels) - Most complex, requires architecture decision

## Prerequisites for Building with Kokkos

### 1. Install Kokkos
```bash
git clone https://github.com/kokkos/kokkos.git
cd kokkos
mkdir build && cd build

# For CUDA backend:
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -DKokkos_ENABLE_CUDA=ON \
  -DKokkos_ARCH_AMPERE80=ON \
  -DKokkos_ENABLE_CUDA_LAMBDA=ON

# For HIP backend (AMD):
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -DKokkos_ENABLE_HIP=ON \
  -DKokkos_ARCH_VEGA90A=ON

# For OpenMP backend (CPU):
cmake .. \
  -DCMAKE_INSTALL_PREFIX=/path/to/install \
  -DKokkos_ENABLE_OPENMP=ON

make install
```

### 2. Build Helios with Kokkos
```bash
# Set Kokkos path
export Kokkos_DIR=/path/to/kokkos/install

# Build specific plugin
cd plugins/energybalance
mkdir build && cd build
cmake ..
make
```

## Conversion Patterns

### CUDA → Kokkos Mapping Reference

| CUDA Concept | Kokkos Equivalent |
|--------------|-------------------|
| `__global__ void kernel(...)` | `Kokkos::parallel_for("label", range, KOKKOS_LAMBDA ...)` |
| `__device__ function` | `KOKKOS_INLINE_FUNCTION` |
| `cudaMalloc/cudaFree` | `Kokkos::View<T*>` (RAII) |
| `cudaMemcpy` | `Kokkos::deep_copy` |
| `blockIdx.x*blockDim.x+threadIdx.x` | Loop index parameter |
| `atomicAdd` | `Kokkos::atomic_add` |
| `__syncthreads()` | `team.team_barrier()` (team policy) |
| `exp(), sqrt(), etc.` | `Kokkos::exp()`, `Kokkos::sqrt()` |

### Example Conversion

#### Before (CUDA):
```cpp
__device__ float compute(float x) {
    return exp(x) * sqrt(x);
}

__global__ void myKernel(int N, float* data) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        data[i] = compute(data[i]);
    }
}

// Host code
float* d_data;
cudaMalloc(&d_data, N*sizeof(float));
cudaMemcpy(d_data, h_data, N*sizeof(float), cudaMemcpyHostToDevice);
myKernel<<<grid, block>>>(N, d_data);
cudaMemcpy(h_data, d_data, N*sizeof(float), cudaMemcpyDeviceToHost);
cudaFree(d_data);
```

#### After (Kokkos):
```cpp
KOKKOS_INLINE_FUNCTION float compute(float x) {
    return Kokkos::exp(x) * Kokkos::sqrt(x);
}

// Host code
Kokkos::View<float*> data("data", N);
auto h_data = Kokkos::create_mirror_view(data);
// ... populate h_data ...
Kokkos::deep_copy(data, h_data);

Kokkos::parallel_for("myKernel", N, KOKKOS_LAMBDA(const int i) {
    data(i) = compute(data(i));
});

Kokkos::deep_copy(h_data, data);
// Automatic cleanup via RAII
```

## Build System Changes

### Old CMakeLists.txt (CUDA):
```cmake
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
CUDA_ADD_LIBRARY(plugin STATIC src/code.cu)
target_link_libraries(plugin ${CUDA_LIBRARIES})
```

### New CMakeLists.txt (Kokkos):
```cmake
find_package(Kokkos REQUIRED)
add_library(plugin STATIC src/code.cpp)
target_link_libraries(plugin Kokkos::kokkos)
target_compile_features(plugin PUBLIC cxx_std_17)
```

## Testing Strategy

### 1. Functional Testing
- Run existing self-tests for each converted plugin
- Compare numerical results with CUDA version
- Validate with different input datasets

### 2. Performance Testing
- Benchmark against original CUDA implementation
- Test across different backends (CUDA, HIP, OpenMP)
- Profile memory transfer overhead

### 3. Portability Testing
- Test on NVIDIA GPUs (CUDA backend)
- Test on AMD GPUs (HIP backend)
- Test on CPUs (OpenMP backend)

## Common Pitfalls

### 1. Forgetting KOKKOS_INLINE_FUNCTION
Device functions must be marked with `KOKKOS_INLINE_FUNCTION`:
```cpp
// Wrong:
float myFunction() { ... }

// Correct:
KOKKOS_INLINE_FUNCTION float myFunction() { ... }
```

### 2. Using Standard Math Functions
Use Kokkos math functions in device code:
```cpp
// Wrong:
float x = exp(y);

// Correct:
float x = Kokkos::exp(y);
```

### 3. Direct Memory Access
Don't mix raw pointers with Views:
```cpp
// Wrong:
float* ptr = data.data();
ptr[i] = ...;  // May not work on GPU

// Correct:
data(i) = ...;  // Use View accessor
```

## Support for Mixed CUDA/Kokkos Development

During transition, you may need to maintain both versions:
- Keep original `.cu` files for reference
- Use conditional compilation if needed
- Ensure both versions pass the same tests

## Resources

### Kokkos Documentation
- Main site: https://kokkos.org/
- Programming Guide: https://kokkos.github.io/kokkos-core-wiki/
- Tutorials: https://github.com/kokkos/kokkos-tutorials

### Helios-Specific
- Plugin conversion docs in each `plugins/*/KOKKOS_CONVERSION.md`
- Original CUDA code preserved as `.cu` files

## Contributors
- Conversion effort started: 2025-11-07

## License
Same as Helios project (GPL v2)
