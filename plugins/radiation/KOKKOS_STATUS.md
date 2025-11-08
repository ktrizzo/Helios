# Radiation Plugin: Kokkos Implementation Status

## Current State: HYBRID BUILD SYSTEM (Functional Stub)

**Date**: 2025-11-08
**Status**: ✅ **Build System Complete** | ⚠️ **Kokkos Path is Stub Only**

---

## What Works

### ✅ Conditional Compilation System
The radiation plugin now supports **two build paths**:

#### 1. **OptiX Path** (NVIDIA GPUs with CUDA)
- **Trigger**: `find_package(CUDA)` succeeds
- **Preprocessor**: `HELIOS_USE_OPTIX` defined
- **Implementation**: Full-featured OptiX ray tracer (existing code)
- **Status**: ✅ **Production Ready**
- **Performance**: Excellent (hardware-accelerated)

#### 2. **Kokkos Path** (Non-NVIDIA platforms)
- **Trigger**: CUDA not found
- **Preprocessor**: `HELIOS_USE_KOKKOS` defined
- **Implementation**: ⚠️ **STUB ONLY** (returns zero flux)
- **Status**: 🚧 **Under Development**
- **Performance**: N/A (not functional)

---

## Build Behavior

### On Systems **WITH** CUDA/OptiX (NVIDIA)
```cmake
-- CUDA found - building Radiation plugin with OptiX
-- Using OptiX version 6.5
```

- ✅ Full radiation functionality
- ✅ All tests pass
- ✅ High performance ray tracing

### On Systems **WITHOUT** CUDA (Mac, AMD, Intel)
```cmake
-- CUDA not found - building Radiation plugin with Kokkos (GPU-agnostic)
-- Note: Kokkos radiation model is a simplified implementation
--       For best performance on NVIDIA GPUs, install CUDA and rebuild
```

- ✅ Plugin compiles successfully
- ✅ No runtime crashes
- ⚠️ All radiation functions return **zero flux**
- ⚠️ Tests will FAIL (expected)

---

## What the Kokkos Stub Does

### Current Implementation (`RadiationModel_kokkos.cpp`)

| Function | Behavior | Status |
|----------|----------|--------|
| `initializeKokkos()` | Prints warning message | ✅ Complete |
| `updateGeometry_kokkos()` | Counts primitives, returns | ✅ Complete |
| `runBand_kokkos(label)` | Sets all flux to 0.0 | ✅ Complete |
| `runBand_kokkos(vector)` | Calls single-band version | ✅ Complete |

### What's Missing (TODO)

**Phase 1: Basic Ray Tracing** (~1 week)
- [ ] Build primitive database in Kokkos::Views
- [ ] Implement brute-force ray-primitive intersection
- [ ] Cast rays from collimated sources
- [ ] Accumulate flux on primitives
- [ ] Support patches and triangles

**Phase 2: Acceleration Structure** (~1 week)
- [ ] Build BVH on host
- [ ] Implement BVH traversal in Kokkos
- [ ] Optimize memory layout (SoA)

**Phase 3: Full Features** (~1-2 weeks)
- [ ] Multi-bounce scattering
- [ ] Diffuse radiation sources
- [ ] Camera simulation
- [ ] Texture transparency masks
- [ ] All primitive types (voxels, disks, tiles)

**Phase 4: Optimization** (~3-5 days)
- [ ] Memory coalescing
- [ ] Warp efficiency
- [ ] Performance tuning

---

## Files Modified

### ✅ Created/Modified
1. **CMakeLists.txt** - Conditional OptiX/Kokkos build logic
2. **src/RadiationModel_kokkos.cpp** - Stub implementation (135 lines)
3. **include/RayTracing_kokkos.h** - Ray-geometry intersection utilities
4. **KOKKOS_RAYTRACING_PLAN.md** - Complete development plan
5. **KOKKOS_STATUS.md** - This file

### ⚠️ Needs Modification (Future Work)
1. **include/RadiationModel.h** - Add Kokkos function declarations
2. **src/RadiationModel.cpp** - Add `#ifdef` switches to call Kokkos versions
3. **samples/radiation_selftest/main.cpp** - Add Kokkos::initialize/finalize

---

## How to Use

### Building with OptiX (NVIDIA)
```bash
# Ensure CUDA is installed and findable
cd /path/to/Helios
mkdir build && cd build
cmake ..
make radiation
./samples/radiation_selftest/radiation_selftest  # Should pass
```

### Building with Kokkos (Non-NVIDIA)
```bash
# Install Kokkos first
cd /tmp
git clone https://github.com/kokkos/kokkos.git
cd kokkos && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/kokkos-install \
         -DKokkos_ENABLE_SERIAL=ON -DCMAKE_CXX_STANDARD=17
make install

# Build Helios
cd /path/to/Helios
mkdir build && cd build
export Kokkos_DIR=$HOME/kokkos-install/lib/cmake/Kokkos
cmake ..
make radiation  # Compiles successfully
./samples/radiation_selftest/radiation_selftest  # Returns zero flux (expected)
```

---

## Testing Status

### Expected Behavior

| Test | OptiX Path | Kokkos Path (Current) |
|------|-----------|----------------------|
| Test #1: 90° squares | ✅ PASS | ❌ FAIL (returns 0) |
| Test #2: Parallel rectangles | ✅ PASS | ❌ FAIL (returns 0) |
| Test #3: Spherical sources | ✅ PASS | ❌ FAIL (returns 0) |
| Test #4: Anisotropic scattering | ✅ PASS | ❌ FAIL (returns 0) |
| Test #5: Periodic boundaries | ✅ PASS | ❌ FAIL (returns 0) |

**Note**: Kokkos path failures are **expected** until full implementation is complete.

---

## Development Timeline

### ✅ Completed (Nov 8, 2025)
- Hybrid build system
- CMake conditional compilation
- Stub implementation
- Documentation

### 🚧 In Progress
- None currently (awaiting decision on full implementation)

### 📋 Planned (If Funded)
- **Week 1**: Phase 1 (Basic ray tracing)
- **Week 2**: Phase 2 (BVH acceleration)
- **Week 3-4**: Phase 3 (Full features)
- **Week 5**: Phase 4 (Optimization)

**Total**: ~1 month for production-quality Kokkos radiation model

---

## Design Resources

### Reference Implementations
1. **PlantSimulationLab/Helios** - Collision detection plugin with BVH
   - File: `plugins/collisiondetection/src/CollisionDetection_RayTracing.cpp`
   - Algorithms: Möller-Trumbore, BVH traversal, SoA optimization

2. **Existing OptiX Code**
   - File: `src/rayGeneration.cu`, `src/primitiveIntersection.cu`
   - Monte Carlo algorithms, scattering logic

### Algorithm References
- **Ray-Triangle**: Möller-Trumbore (1997) - Already implemented in `RayTracing_kokkos.h`
- **Ray-Patch**: Ray-plane + bounds check - Already implemented
- **BVH**: Surface Area Heuristic (SAH) for construction
- **Monte Carlo**: Importance sampling for radiation transport

---

## Contribution Guidelines

### To Implement Kokkos Radiation

1. **Start with Phase 1 MVP** (1 week effort)
   - Focus on Test #1 (simplest)
   - Brute-force ray casting (no BVH)
   - Patches and triangles only

2. **Use Existing Patterns**
   - See `plugins/energybalance/src/EnergyBalanceModel_kokkos.cpp` for Kokkos functor pattern
   - See `PlantSimulationLab/Helios` collision detection for ray tracing algorithms

3. **Test Incrementally**
   - Add unit tests for each intersection function
   - Compare against OptiX results
   - Validate Monte Carlo convergence

---

## Known Issues

### Current
- ⚠️ `RadiationModel.h` still includes OptiX headers (causes warnings on non-NVIDIA)
- ⚠️ `RadiationModel.cpp` doesn't call Kokkos versions yet (requires `#ifdef` additions)

### Future
- Performance may be 2-10x slower than OptiX initially (before optimization)
- Memory usage higher without OptiX's automatic BVH builder

---

## Contact / Questions

For questions about:
- **Build system**: Check `CMakeLists.txt` comments
- **Implementation plan**: See `KOKKOS_RAYTRACING_PLAN.md`
- **Ray tracing utilities**: See `include/RayTracing_kokkos.h`

---

## Summary

✅ **What works**: Conditional build system, compiles on all platforms
⚠️ **What's missing**: Actual ray tracing computation (~4 weeks of work)
🎯 **Immediate value**: Can build Helios on Mac/AMD/Intel for non-radiation workflows
🚀 **Future potential**: Full GPU-agnostic radiation model with Kokkos

**Recommendation**: Use OptiX path for production radiation work. Implement Kokkos path only if cross-platform radiation is a critical requirement.
