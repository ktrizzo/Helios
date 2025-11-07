# EnergyBalance Plugin: CUDA to Kokkos Conversion

## Overview
This document describes the conversion of the EnergyBalance plugin from CUDA to Kokkos for GPU-agnostic computing.

## Conversion Date
2025-11-07

## Files Modified

### New Files
- `src/EnergyBalanceModel_kokkos.cpp` - Kokkos implementation replacing CUDA kernel

### Modified Files
- `CMakeLists.txt` - Updated build system to use Kokkos instead of CUDA

### Original Files (Preserved for Reference)
- `src/EnergyBalanceModel.cu` - Original CUDA implementation (kept for comparison)

## Key Changes

### 1. Kernel Conversion
**Original CUDA:**
```cpp
__global__ void solveEnergyBalance(...) {
    uint p = blockIdx.x*blockDim.x+threadIdx.x;
    if(p >= Nprimitives) return;
    // kernel logic
}
```

**Kokkos Equivalent:**
```cpp
Kokkos::parallel_for("solveEnergyBalance", Nprimitives, KOKKOS_LAMBDA(const uint p) {
    // kernel logic (unchanged)
});
```

### 2. Memory Management
**Original CUDA:**
```cpp
cudaMalloc((void**)&d_To, Nprimitives*sizeof(float));
cudaMemcpy(d_To, To, Nprimitives*sizeof(float), cudaMemcpyHostToDevice);
cudaFree(d_To);
```

**Kokkos Equivalent:**
```cpp
Kokkos::View<float*> d_To("To", Nprimitives);
auto h_To = Kokkos::create_mirror_view(d_To);
// ... populate h_To ...
Kokkos::deep_copy(d_To, h_To);
// Automatic cleanup via RAII
```

### 3. Device Functions
**Original CUDA:**
```cpp
__device__ float evaluateEnergyBalance(...) { ... }
```

**Kokkos Equivalent:**
```cpp
KOKKOS_INLINE_FUNCTION float evaluateEnergyBalance(...) { ... }
```

### 4. Math Functions
- `exp()` → `Kokkos::exp()`
- `fabs()` → `Kokkos::fabs()`
- `printf()` → `Kokkos::printf()`

## Build Requirements

### CMake Version
- Minimum: 3.16 (updated from 2.0)

### Dependencies
- **Kokkos** (replaces CUDA dependency)
- C++17 standard or later

### Build Configuration
The plugin now uses:
```cmake
find_package(Kokkos REQUIRED)
target_link_libraries(energybalance Kokkos::kokkos)
target_compile_features(energybalance PUBLIC cxx_std_17)
```

## Backend Support
With Kokkos, this plugin now supports multiple backends:
- **CUDA** - NVIDIA GPUs (original functionality preserved)
- **HIP** - AMD GPUs
- **SYCL** - Intel GPUs
- **OpenMP** - CPU multi-threading
- **Serial** - Single-threaded CPU execution

The backend is selected at Kokkos build time, not at compile time of this plugin.

## Performance Considerations
- Memory transfers are explicit via `Kokkos::deep_copy`
- Automatic memory management reduces memory leaks
- Performance should be comparable to CUDA when using CUDA backend
- Cross-platform performance depends on Kokkos backend optimization

## Testing
To test the conversion:
1. Build with Kokkos configured for desired backend
2. Run existing self-tests: `EnergyBalanceModel::selfTest()`
3. Compare results with CUDA version (if available)

## Known Limitations
- Kokkos must be installed and configured before building this plugin
- Backend selection is a Kokkos configuration choice, not runtime

## Future Work
- Performance benchmarking across different backends
- Potential optimization of memory access patterns for specific architectures
- Integration with other Kokkos-enabled plugins

## References
- Kokkos documentation: https://kokkos.org/
- Original CUDA implementation: `src/EnergyBalanceModel.cu`
