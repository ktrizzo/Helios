# Radiation Plugin: OptiX to Kokkos Ray Tracing Conversion Plan

## Overview
This document tracks the conversion of the Radiation plugin from NVIDIA OptiX to Kokkos-based ray tracing for GPU-agnostic execution.

## Status: Foundation Complete (2025-11-08)

### ✅ Completed

#### 1. Research & Analysis
- **PlantSimulationLab Helios Discovery**: Found existing ray tracing implementation in collision detection plugin
- **Code Extraction**: Downloaded and analyzed ~3600 lines of ray tracing code
- **Algorithm Identification**: Möller-Trumbore (triangles), ray-plane+bounds (patches)

#### 2. Core Ray Tracing Utilities Created
**File**: `include/RayTracing_kokkos.h`

**Functions Implemented**:
- `rayTriangleIntersect()` - Möller-Trumbore algorithm
- `rayPatchIntersect()` - Ray-quadrilateral intersection
- `calculateSurfaceNormal()` - Surface normal computation

**Key Features**:
- ✅ Kokkos-compatible (`KOKKOS_INLINE_FUNCTION`)
- ✅ Zero dynamic allocation (stack-only)
- ✅ Optimized component-wise operations
- ✅ Epsilon tolerance for numerical stability

---

## Implementation Strategy

### Phase 1: Minimal Viable Product (MVP) - 2-3 days
**Goal**: Get radiation working on non-NVIDIA platforms

**Approach**: Brute-force ray tracing (no BVH initially)

**Tasks**:
1. **Create RadiationModel_kokkos.cpp**
   - Implement simple ray casting kernel
   - Test all primitives for each ray (O(N) per ray)
   - Support patches and triangles only

2. **Update CMakeLists.txt**
   - Conditional compilation:
     ```cmake
     if(KOKKOS_ENABLE_CUDA)
       # Use OptiX (existing code)
     else()
       # Use Kokkos ray tracer
     endif()
     ```

3. **Test on Mac/CPU**
   - Verify numerical correctness
   - Accept slower performance initially

**Expected Performance**: 10-100x slower than OptiX (acceptable for MVP)

---

### Phase 2: BVH Acceleration - 1-2 weeks
**Goal**: Competitive performance with OptiX

**Approach**: Add spatial acceleration structure

**Tasks**:
1. **BVH Construction**
   - Build bounding volume hierarchy on host
   - Use SAH (Surface Area Heuristic) for quality splits
   - Store in Kokkos::View for device access

2. **BVH Traversal**
   - Stack-based traversal per ray
   - Early ray termination
   - Optimize memory layout (SoA)

3. **Benchmarking**
   - Compare against OptiX on NVIDIA
   - Optimize hotspots

**Expected Performance**: 1-5x slower than OptiX

---

### Phase 3: Full Feature Parity - Additional 1 week
**Goal**: Support all Radiation plugin features

**Tasks**:
1. **Additional Primitive Types**
   - Voxels (AABB intersection)
   - Disks
   - Tiles

2. **Advanced Features**
   - Texture transparency masking
   - Multi-ray batching
   - Camera lens simulation

3. **Optimization**
   - Memory coalescing
   - Warp/wavefront efficiency
   - Dynamic parallelism (if beneficial)

---

## Current Code Structure

### Ray Tracing Utilities (`RayTracing_kokkos.h`)
```cpp
namespace helios {
    // Triangle intersection (Möller-Trumbore)
    KOKKOS_INLINE_FUNCTION
    bool rayTriangleIntersect(origin, direction, v0, v1, v2, distance);

    // Patch intersection (ray-plane + bounds check)
    KOKKOS_INLINE_FUNCTION
    bool rayPatchIntersect(origin, direction, v0, v1, v2, v3, distance);

    // Surface normal calculation
    KOKKOS_INLINE_FUNCTION
    vec3 calculateSurfaceNormal(v0, v1, v2, ray_direction);
}
```

### Future: Radiation Kernel (Conceptual)
```cpp
Kokkos::parallel_for("castRays", Nrays, KOKKOS_LAMBDA(const int ray_id) {
    vec3 origin = ray_origins(ray_id);
    vec3 direction = ray_directions(ray_id);

    float closest_distance = FLT_MAX;
    uint hit_primitive = UINT_MAX;

    // Brute force: test all primitives
    for (uint prim_id = 0; prim_id < Nprimitives; prim_id++) {
        float distance;
        bool hit = false;

        if (primitive_type(prim_id) == TRIANGLE) {
            hit = rayTriangleIntersect(origin, direction,
                                       vertices(prim_id, 0),
                                       vertices(prim_id, 1),
                                       vertices(prim_id, 2),
                                       distance);
        } else if (primitive_type(prim_id) == PATCH) {
            hit = rayPatchIntersect(origin, direction,
                                   vertices(prim_id, 0),
                                   vertices(prim_id, 1),
                                   vertices(prim_id, 2),
                                   vertices(prim_id, 3),
                                   distance);
        }

        if (hit && distance < closest_distance) {
            closest_distance = distance;
            hit_primitive = prim_id;
        }
    }

    // Store result
    hit_distances(ray_id) = closest_distance;
    hit_primitives(ray_id) = hit_primitive;
});
```

---

## Comparison: OptiX vs Kokkos

| Feature | OptiX (Current) | Kokkos (Planned) |
|---------|----------------|------------------|
| **GPU Vendors** | NVIDIA only | NVIDIA, AMD, Intel |
| **CPU Fallback** | No | Yes (Serial, OpenMP) |
| **BVH** | Automatic | Manual implementation |
| **Performance (NVIDIA)** | Excellent | Good (90-95% of OptiX) |
| **Performance (AMD/Intel)** | N/A | Good |
| **Code Complexity** | High (PTX, OptiX API) | Medium (standard C++) |
| **Portability** | Low | High |
| **Maintainability** | Medium | High (single codebase) |

---

## Open Questions

### 1. BVH Builder Location
**Options**:
- **Host-side** (CPU): Simpler, works for static scenes
- **Device-side** (GPU): Faster for dynamic scenes, more complex

**Decision**: Start with host-side, migrate to device if needed

### 2. Stack Management in Traversal
**Challenge**: BVH traversal uses recursion/stack per ray

**Options**:
- **Fixed stack per thread**: Simple, memory overhead
- **Stackless traversal**: Complex, better memory efficiency
- **Work queue**: Good for load balancing, complex

**Decision**: Start with fixed stack (32 levels sufficient for most scenes)

### 3. Memory Layout
**Options**:
- **Array of Structures (AoS)**: Simple, poor coalescing
- **Structure of Arrays (SoA)**: Better coalescing, more complex

**Decision**: SoA for frequently accessed data (AABB bounds), AoS for cold data

---

## Testing Strategy

### Unit Tests
1. **Ray-Triangle**: Test corner cases (parallel, edge hits, degenerate)
2. **Ray-Patch**: Test various quadrilateral configurations
3. **BVH Build**: Verify hierarchy correctness
4. **BVH Traversal**: Compare against brute-force results

### Integration Tests
1. **Radiation Self-Test**: Must pass existing tests
2. **Performance Benchmarks**: Compare OptiX vs Kokkos
3. **Cross-Platform**: Test on NVIDIA, AMD, Intel, CPU

---

## Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|-----------|
| **✅ Foundation** | 1 day | 1 day |
| **Phase 1: MVP** | 2-3 days | 3-4 days |
| **Phase 2: BVH** | 1-2 weeks | ~3 weeks |
| **Phase 3: Full Feature** | 1 week | ~4 weeks |

**Total**: ~1 month for complete, production-ready Kokkos ray tracer

---

## Next Immediate Steps

1. **Create MVP radiation kernel** (RadiationModel_kokkos.cpp)
2. **Update CMakeLists.txt** for conditional compilation
3. **Test on Mac** with Serial backend
4. **Iterate based on results**

---

## References

- **Source**: PlantSimulationLab/Helios collision detection plugin
- **Algorithms**: Möller-Trumbore (1997), ray-plane intersection
- **Kokkos Docs**: https://kokkos.github.io/kokkos-core-wiki/
- **BVH Resources**: "Bounding Volume Hierarchies" (Wald, 2007)

---

## Contributors
- Conversion started: 2025-11-08
- Based on: PlantSimulationLab/Helios CollisionDetection plugin

## License
GPL v2 (same as Helios)
