/** \file "RadiationModel_kokkos.cpp" Kokkos-based radiation transport (GPU-agnostic)
 *
 *  Copyright (C) 2016-2024 Brian Bailey
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, version 2.
 */

#include "RadiationModel.h"
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include "RayTracing_kokkos.h"
#include <cmath>
#include <limits>

using namespace helios;

#ifdef HELIOS_USE_KOKKOS

namespace {
    // Constants
    constexpr float INFINITY_DISTANCE = 1e10f;
    constexpr uint PRIMITIVE_TYPE_PATCH = 0;
    constexpr uint PRIMITIVE_TYPE_TRIANGLE = 1;
    constexpr uint PRIMITIVE_TYPE_DISK = 2;
    constexpr uint PRIMITIVE_TYPE_TILE = 3;
    constexpr uint PRIMITIVE_TYPE_VOXEL = 4;
    constexpr uint PRIMITIVE_TYPE_BBOX = 5;
}

// Helper struct to store primitive data
struct PrimitiveData {
    uint type;              // Primitive type
    vec3 vertices[4];       // Vertices (patches=4, triangles=3)
    float area;             // Surface area
    uint UUID;              // Original UUID
    float rho;              // Reflectivity
    float tau;              // Transmissivity
    float emission;         // Thermal emission flux
    bool twosided;          // Two-sided flag
};

// Random number generator functor
struct RandomGenerator {
    using generator_pool = Kokkos::Random_XorShift64_Pool<>;
    generator_pool pool;

    RandomGenerator(uint64_t seed) : pool(seed) {}

    KOKKOS_INLINE_FUNCTION
    float uniform(uint64_t state) const {
        auto generator = pool.get_state(state);
        float value = generator.frand();
        pool.free_state(generator);
        return value;
    }
};

// Direct radiation ray generation functor
struct DirectRaysFunctor {
    Kokkos::View<PrimitiveData*> primitives;
    Kokkos::View<float*> flux_out;
    vec3 source_direction;
    float source_flux;
    uint Nprimitives;
    uint Nrays_per_prim;
    RandomGenerator rng;

    DirectRaysFunctor(Kokkos::View<PrimitiveData*> prims,
                      Kokkos::View<float*> flux,
                      vec3 source_dir,
                      float source_f,
                      uint Nprims,
                      uint Nrays,
                      const RandomGenerator& random)
        : primitives(prims), flux_out(flux), source_direction(source_dir),
          source_flux(source_f), Nprimitives(Nprims),
          Nrays_per_prim(Nrays), rng(random) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint ray_idx) const {
        uint prim_id = ray_idx / Nrays_per_prim;
        uint ray_local = ray_idx % Nrays_per_prim;

        if (prim_id >= Nprimitives) return;

        const PrimitiveData& prim = primitives(prim_id);

        // Generate ray origin on primitive surface using stratified sampling
        float u = (float(ray_local % uint(Kokkos::sqrt(float(Nrays_per_prim)))) + 0.5f) /
                  Kokkos::sqrt(float(Nrays_per_prim));
        float v = (float(ray_local / uint(Kokkos::sqrt(float(Nrays_per_prim)))) + 0.5f) /
                  Kokkos::sqrt(float(Nrays_per_prim));

        vec3 ray_origin;
        vec3 normal;

        if (prim.type == PRIMITIVE_TYPE_PATCH) {
            // Bilinear interpolation for patch
            ray_origin.x = (1-u)*(1-v)*prim.vertices[0].x + u*(1-v)*prim.vertices[1].x +
                          u*v*prim.vertices[2].x + (1-u)*v*prim.vertices[3].x;
            ray_origin.y = (1-u)*(1-v)*prim.vertices[0].y + u*(1-v)*prim.vertices[1].y +
                          u*v*prim.vertices[2].y + (1-u)*v*prim.vertices[3].y;
            ray_origin.z = (1-u)*(1-v)*prim.vertices[0].z + u*(1-v)*prim.vertices[1].z +
                          u*v*prim.vertices[2].z + (1-u)*v*prim.vertices[3].z;
            normal = calculateSurfaceNormal(prim.vertices[0], prim.vertices[1], prim.vertices[2], source_direction);
        } else if (prim.type == PRIMITIVE_TYPE_TRIANGLE) {
            // Barycentric sampling for triangle
            if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
            }
            float w = 1.0f - u - v;
            ray_origin.x = w*prim.vertices[0].x + u*prim.vertices[1].x + v*prim.vertices[2].x;
            ray_origin.y = w*prim.vertices[0].y + u*prim.vertices[1].y + v*prim.vertices[2].y;
            ray_origin.z = w*prim.vertices[0].z + u*prim.vertices[1].z + v*prim.vertices[2].z;
            normal = calculateSurfaceNormal(prim.vertices[0], prim.vertices[1], prim.vertices[2], source_direction);
        } else {
            return; // Unsupported primitive type
        }

        // Check if ray hits the primitive (dot product with normal)
        float cos_theta = -(normal.x * source_direction.x +
                           normal.y * source_direction.y +
                           normal.z * source_direction.z);

        if (cos_theta <= 0.0f) {
            if (!prim.twosided) return; // Backface, no contribution
            cos_theta = -cos_theta;
        }

        // Cast ray in source direction and check for occlusions
        bool occluded = false;
        float min_distance = INFINITY_DISTANCE;

        for (uint other_id = 0; other_id < Nprimitives; other_id++) {
            if (other_id == prim_id) continue;

            const PrimitiveData& other = primitives(other_id);
            float hit_distance;
            bool hit = false;

            if (other.type == PRIMITIVE_TYPE_PATCH) {
                hit = rayPatchIntersect(ray_origin, source_direction,
                                       other.vertices[0], other.vertices[1],
                                       other.vertices[2], other.vertices[3],
                                       hit_distance);
            } else if (other.type == PRIMITIVE_TYPE_TRIANGLE) {
                hit = rayTriangleIntersect(ray_origin, source_direction,
                                          other.vertices[0], other.vertices[1], other.vertices[2],
                                          hit_distance);
            }

            if (hit && hit_distance < min_distance) {
                occluded = true;
                min_distance = hit_distance;
            }
        }

        if (!occluded) {
            // Accumulate flux contribution
            float contribution = source_flux * cos_theta / float(Nrays_per_prim);
            Kokkos::atomic_add(&flux_out(prim_id), contribution);
        }
    }
};

// Diffuse radiation ray generation functor
struct DiffuseRaysFunctor {
    Kokkos::View<PrimitiveData*> primitives;
    Kokkos::View<float*> flux_out;
    float diffuse_flux;
    uint Nprimitives;
    uint Nrays_per_prim;
    RandomGenerator rng;
    uint64_t seed_offset;

    DiffuseRaysFunctor(Kokkos::View<PrimitiveData*> prims,
                       Kokkos::View<float*> flux,
                       float diff_flux,
                       uint Nprims,
                       uint Nrays,
                       const RandomGenerator& random,
                       uint64_t offset)
        : primitives(prims), flux_out(flux), diffuse_flux(diff_flux),
          Nprimitives(Nprims), Nrays_per_prim(Nrays), rng(random), seed_offset(offset) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint ray_idx) const {
        auto generator = rng.pool.get_state(ray_idx + seed_offset);

        uint prim_id = ray_idx / Nrays_per_prim;
        if (prim_id >= Nprimitives) return;

        const PrimitiveData& prim = primitives(prim_id);

        // Random point on primitive surface
        float u = generator.frand();
        float v = generator.frand();

        vec3 ray_origin;
        vec3 normal;

        if (prim.type == PRIMITIVE_TYPE_PATCH) {
            ray_origin.x = (1-u)*(1-v)*prim.vertices[0].x + u*(1-v)*prim.vertices[1].x +
                          u*v*prim.vertices[2].x + (1-u)*v*prim.vertices[3].x;
            ray_origin.y = (1-u)*(1-v)*prim.vertices[0].y + u*(1-v)*prim.vertices[1].y +
                          u*v*prim.vertices[2].y + (1-u)*v*prim.vertices[3].y;
            ray_origin.z = (1-u)*(1-v)*prim.vertices[0].z + u*(1-v)*prim.vertices[1].z +
                          u*v*prim.vertices[2].z + (1-u)*v*prim.vertices[3].z;
            vec3 dummy_dir = make_vec3(0, 0, 1);
            normal = calculateSurfaceNormal(prim.vertices[0], prim.vertices[1], prim.vertices[2], dummy_dir);
        } else if (prim.type == PRIMITIVE_TYPE_TRIANGLE) {
            if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
            }
            float w = 1.0f - u - v;
            ray_origin.x = w*prim.vertices[0].x + u*prim.vertices[1].x + v*prim.vertices[2].x;
            ray_origin.y = w*prim.vertices[0].y + u*prim.vertices[1].y + v*prim.vertices[2].y;
            ray_origin.z = w*prim.vertices[0].z + u*prim.vertices[1].z + v*prim.vertices[2].z;
            vec3 dummy_dir = make_vec3(0, 0, 1);
            normal = calculateSurfaceNormal(prim.vertices[0], prim.vertices[1], prim.vertices[2], dummy_dir);
        } else {
            rng.pool.free_state(generator);
            return;
        }

        // Sample cosine-weighted hemisphere direction
        float r1 = generator.frand();
        float r2 = generator.frand();

        float theta = Kokkos::acos(Kokkos::sqrt(r1));
        float phi = 2.0f * M_PI * r2;

        // Local coordinates relative to normal
        vec3 tangent, bitangent;
        if (Kokkos::fabs(normal.z) < 0.999f) {
            tangent = make_vec3(-normal.y, normal.x, 0.0f);
        } else {
            tangent = make_vec3(1.0f, 0.0f, 0.0f);
        }
        float tangent_mag = Kokkos::sqrt(tangent.x*tangent.x + tangent.y*tangent.y + tangent.z*tangent.z);
        tangent.x /= tangent_mag;
        tangent.y /= tangent_mag;
        tangent.z /= tangent_mag;

        bitangent.x = normal.y*tangent.z - normal.z*tangent.y;
        bitangent.y = normal.z*tangent.x - normal.x*tangent.z;
        bitangent.z = normal.x*tangent.y - normal.y*tangent.x;

        // Convert to world space
        float sin_theta = Kokkos::sin(theta);
        float cos_theta = Kokkos::cos(theta);
        float sin_phi = Kokkos::sin(phi);
        float cos_phi = Kokkos::cos(phi);

        vec3 ray_direction;
        ray_direction.x = sin_theta * cos_phi * tangent.x + sin_theta * sin_phi * bitangent.x + cos_theta * normal.x;
        ray_direction.y = sin_theta * cos_phi * tangent.y + sin_theta * sin_phi * bitangent.y + cos_theta * normal.y;
        ray_direction.z = sin_theta * cos_phi * tangent.z + sin_theta * sin_phi * bitangent.z + cos_theta * normal.z;

        // Normalize
        float mag = Kokkos::sqrt(ray_direction.x*ray_direction.x +
                                ray_direction.y*ray_direction.y +
                                ray_direction.z*ray_direction.z);
        ray_direction.x /= mag;
        ray_direction.y /= mag;
        ray_direction.z /= mag;

        // Cast ray and check for occlusions
        bool occluded = false;

        for (uint other_id = 0; other_id < Nprimitives; other_id++) {
            if (other_id == prim_id) continue;

            const PrimitiveData& other = primitives(other_id);
            float hit_distance;
            bool hit = false;

            if (other.type == PRIMITIVE_TYPE_PATCH) {
                hit = rayPatchIntersect(ray_origin, ray_direction,
                                       other.vertices[0], other.vertices[1],
                                       other.vertices[2], other.vertices[3],
                                       hit_distance);
            } else if (other.type == PRIMITIVE_TYPE_TRIANGLE) {
                hit = rayTriangleIntersect(ray_origin, ray_direction,
                                          other.vertices[0], other.vertices[1], other.vertices[2],
                                          hit_distance);
            }

            if (hit) {
                occluded = true;
                break;
            }
        }

        if (!occluded) {
            // Cosine-weighted sampling already accounts for cos(theta)
            float contribution = diffuse_flux / float(Nrays_per_prim);
            Kokkos::atomic_add(&flux_out(prim_id), contribution);
        }

        rng.pool.free_state(generator);
    }
};

// Emission functor
struct EmissionFunctor {
    Kokkos::View<PrimitiveData*> primitives;
    Kokkos::View<float*> flux_out;
    uint Nprimitives;

    EmissionFunctor(Kokkos::View<PrimitiveData*> prims,
                   Kokkos::View<float*> flux)
        : primitives(prims), flux_out(flux), Nprimitives(prims.extent(0)) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint prim_id) const {
        if (prim_id >= Nprimitives) return;
        const PrimitiveData& prim = primitives(prim_id);

        // Add emission contribution (negative because it's outgoing)
        Kokkos::atomic_add(&flux_out(prim_id), -prim.emission);
    }
};

// Scattering functor (simplified - single bounce)
struct ScatteringFunctor {
    Kokkos::View<PrimitiveData*> primitives;
    Kokkos::View<float*> flux_in;
    Kokkos::View<float*> flux_out;
    uint Nprimitives;
    uint Nrays_per_prim;
    RandomGenerator rng;
    uint64_t seed_offset;

    ScatteringFunctor(Kokkos::View<PrimitiveData*> prims,
                     Kokkos::View<float*> flux_i,
                     Kokkos::View<float*> flux_o,
                     uint Nprims,
                     uint Nrays,
                     const RandomGenerator& random,
                     uint64_t offset)
        : primitives(prims), flux_in(flux_i), flux_out(flux_o),
          Nprimitives(Nprims), Nrays_per_prim(Nrays), rng(random), seed_offset(offset) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const uint ray_idx) const {
        auto generator = rng.pool.get_state(ray_idx + seed_offset);

        uint prim_id = ray_idx / Nrays_per_prim;
        if (prim_id >= Nprimitives) return;

        const PrimitiveData& prim = primitives(prim_id);

        // Energy to scatter = absorbed * reflectivity
        float incident = flux_in(prim_id);
        if (incident <= 0.0f) {
            rng.pool.free_state(generator);
            return;
        }

        float to_scatter = incident * prim.rho;

        // Random point on primitive
        float u = generator.frand();
        float v = generator.frand();

        vec3 ray_origin;
        vec3 normal;

        if (prim.type == PRIMITIVE_TYPE_PATCH) {
            ray_origin.x = (1-u)*(1-v)*prim.vertices[0].x + u*(1-v)*prim.vertices[1].x +
                          u*v*prim.vertices[2].x + (1-u)*v*prim.vertices[3].x;
            ray_origin.y = (1-u)*(1-v)*prim.vertices[0].y + u*(1-v)*prim.vertices[1].y +
                          u*v*prim.vertices[2].y + (1-u)*v*prim.vertices[3].y;
            ray_origin.z = (1-u)*(1-v)*prim.vertices[0].z + u*(1-v)*prim.vertices[1].z +
                          u*v*prim.vertices[2].z + (1-u)*v*prim.vertices[3].z;
            vec3 dummy_dir = make_vec3(0, 0, 1);
            normal = calculateSurfaceNormal(prim.vertices[0], prim.vertices[1], prim.vertices[2], dummy_dir);
        } else if (prim.type == PRIMITIVE_TYPE_TRIANGLE) {
            if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
            }
            float w = 1.0f - u - v;
            ray_origin.x = w*prim.vertices[0].x + u*prim.vertices[1].x + v*prim.vertices[2].x;
            ray_origin.y = w*prim.vertices[0].y + u*prim.vertices[1].y + v*prim.vertices[2].y;
            ray_origin.z = w*prim.vertices[0].z + u*prim.vertices[1].z + v*prim.vertices[2].z;
            vec3 dummy_dir = make_vec3(0, 0, 1);
            normal = calculateSurfaceNormal(prim.vertices[0], prim.vertices[1], prim.vertices[2], dummy_dir);
        } else {
            rng.pool.free_state(generator);
            return;
        }

        // Sample hemisphere
        float r1 = generator.frand();
        float r2 = generator.frand();

        float theta = Kokkos::acos(Kokkos::sqrt(r1));
        float phi = 2.0f * M_PI * r2;

        vec3 tangent, bitangent;
        if (Kokkos::fabs(normal.z) < 0.999f) {
            tangent = make_vec3(-normal.y, normal.x, 0.0f);
        } else {
            tangent = make_vec3(1.0f, 0.0f, 0.0f);
        }
        float tangent_mag = Kokkos::sqrt(tangent.x*tangent.x + tangent.y*tangent.y + tangent.z*tangent.z);
        tangent.x /= tangent_mag;
        tangent.y /= tangent_mag;
        tangent.z /= tangent_mag;

        bitangent.x = normal.y*tangent.z - normal.z*tangent.y;
        bitangent.y = normal.z*tangent.x - normal.x*tangent.z;
        bitangent.z = normal.x*tangent.y - normal.y*tangent.x;

        float sin_theta = Kokkos::sin(theta);
        float cos_theta = Kokkos::cos(theta);
        float sin_phi = Kokkos::sin(phi);
        float cos_phi = Kokkos::cos(phi);

        vec3 ray_direction;
        ray_direction.x = sin_theta * cos_phi * tangent.x + sin_theta * sin_phi * bitangent.x + cos_theta * normal.x;
        ray_direction.y = sin_theta * cos_phi * tangent.y + sin_theta * sin_phi * bitangent.y + cos_theta * normal.y;
        ray_direction.z = sin_theta * cos_phi * tangent.z + sin_theta * sin_phi * bitangent.z + cos_theta * normal.z;

        float mag = Kokkos::sqrt(ray_direction.x*ray_direction.x +
                                ray_direction.y*ray_direction.y +
                                ray_direction.z*ray_direction.z);
        ray_direction.x /= mag;
        ray_direction.y /= mag;
        ray_direction.z /= mag;

        // Find hit primitive
        float min_distance = INFINITY_DISTANCE;
        int hit_prim_id = -1;

        for (uint other_id = 0; other_id < Nprimitives; other_id++) {
            if (other_id == prim_id) continue;

            const PrimitiveData& other = primitives(other_id);
            float hit_distance;
            bool hit = false;

            if (other.type == PRIMITIVE_TYPE_PATCH) {
                hit = rayPatchIntersect(ray_origin, ray_direction,
                                       other.vertices[0], other.vertices[1],
                                       other.vertices[2], other.vertices[3],
                                       hit_distance);
            } else if (other.type == PRIMITIVE_TYPE_TRIANGLE) {
                hit = rayTriangleIntersect(ray_origin, ray_direction,
                                          other.vertices[0], other.vertices[1], other.vertices[2],
                                          hit_distance);
            }

            if (hit && hit_distance < min_distance) {
                min_distance = hit_distance;
                hit_prim_id = other_id;
            }
        }

        if (hit_prim_id >= 0) {
            float contribution = to_scatter / float(Nrays_per_prim);
            Kokkos::atomic_add(&flux_out(hit_prim_id), contribution);
        }

        rng.pool.free_state(generator);
    }
};

// Kokkos-specific initialization
void RadiationModel::initializeKokkos() {
    if (message_flag) {
        std::cout << "Initializing Kokkos radiation model..." << std::endl;
    }

    isgeometryinitialized = false;
}

// Kokkos-specific geometry update
void RadiationModel::updateGeometry_kokkos() {
    if (message_flag) {
        std::cout << "Updating geometry for Kokkos radiation model..." << std::flush;
    }

    // Get all UUIDs from context
    context_UUIDs = context->getAllUUIDs();
    size_t Nprimitives = context_UUIDs.size();

    if (Nprimitives == 0) {
        std::cerr << "WARNING: No primitives in context!" << std::endl;
        return;
    }

    if (message_flag) {
        std::cout << " (" << Nprimitives << " primitives)" << std::endl;
    }

    isgeometryinitialized = true;
}

// Main Kokkos ray tracing function
void RadiationModel::runBand_kokkos(const std::string &label) {
    if (!doesBandExist(label)) {
        helios_runtime_error("ERROR (RadiationModel::runBand_kokkos): Radiation band '" + label + "' does not exist.");
    }

    if (!isgeometryinitialized) {
        helios_runtime_error("ERROR (RadiationModel::runBand_kokkos): Geometry has not been initialized. Call updateGeometry() first.");
    }

    const RadiationBand& band = radiation_bands.at(label);
    size_t Nprimitives = context_UUIDs.size();

    if (Nprimitives == 0) return;

    // Build primitive database on host
    std::vector<PrimitiveData> h_primitives(Nprimitives);

    for (size_t i = 0; i < Nprimitives; i++) {
        uint UUID = context_UUIDs[i];
        helios::PrimitiveType ptype = context->getPrimitiveType(UUID);

        h_primitives[i].UUID = UUID;
        h_primitives[i].area = context->getPrimitiveArea(UUID);

        // Get vertices
        if (ptype == helios::PRIMITIVE_TYPE_PATCH) {
            h_primitives[i].type = PRIMITIVE_TYPE_PATCH;
            std::vector<vec3> verts = context->getPatchPointer(UUID)->getVertices();
            for (int j = 0; j < 4; j++) {
                h_primitives[i].vertices[j] = verts[j];
            }
        } else if (ptype == helios::PRIMITIVE_TYPE_TRIANGLE) {
            h_primitives[i].type = PRIMITIVE_TYPE_TRIANGLE;
            std::vector<vec3> verts = context->getTrianglePointer(UUID)->getVertices();
            for (int j = 0; j < 3; j++) {
                h_primitives[i].vertices[j] = verts[j];
            }
        } else {
            // Unsupported type - skip
            h_primitives[i].type = 999;
            continue;
        }

        // Get radiative properties
        std::string rho_label = "reflectivity_" + label;
        std::string tau_label = "transmissivity_" + label;
        std::string eps_label = "emissivity_" + label;
        std::string temp_label = "temperature";

        if (context->doesPrimitiveDataExist(UUID, rho_label.c_str())) {
            context->getPrimitiveData(UUID, rho_label.c_str(), h_primitives[i].rho);
        } else {
            h_primitives[i].rho = rho_default;
        }

        if (context->doesPrimitiveDataExist(UUID, tau_label.c_str())) {
            context->getPrimitiveData(UUID, tau_label.c_str(), h_primitives[i].tau);
        } else {
            h_primitives[i].tau = tau_default;
        }

        // Calculate emission
        if (band.emissionFlag) {
            float eps, temp;
            if (context->doesPrimitiveDataExist(UUID, eps_label.c_str())) {
                context->getPrimitiveData(UUID, eps_label.c_str(), eps);
            } else {
                eps = eps_default;
            }

            if (context->doesPrimitiveDataExist(UUID, temp_label.c_str())) {
                context->getPrimitiveData(UUID, temp_label.c_str(), temp);
            } else {
                temp = temperature_default;
            }

            h_primitives[i].emission = eps * sigma * powf(temp, 4);
        } else {
            h_primitives[i].emission = 0.0f;
        }

        // Get two-sided flag
        if (context->doesPrimitiveDataExist(UUID, "twosided_flag")) {
            uint flag;
            context->getPrimitiveData(UUID, "twosided_flag", flag);
            h_primitives[i].twosided = (flag != 0);
        } else {
            h_primitives[i].twosided = true;
        }
    }

    // Copy to device
    Kokkos::View<PrimitiveData*> d_primitives("primitives", Nprimitives);
    Kokkos::View<float*> d_flux("flux", Nprimitives);
    Kokkos::deep_copy(d_primitives, Kokkos::View<PrimitiveData*,Kokkos::HostSpace>(h_primitives.data(), Nprimitives));
    Kokkos::deep_copy(d_flux, 0.0f);

    // Random number generator
    RandomGenerator rng(12345);

    // 1. Process direct radiation sources
    for (const auto& source : radiation_sources) {
        if (source.source_fluxes.find(label) == source.source_fluxes.end()) continue;
        float source_flux = source.source_fluxes.at(label);
        if (source_flux < 0) continue;

        if (source.source_type == RADIATION_SOURCE_TYPE_COLLIMATED) {
            vec3 source_dir = source.source_position;
            source_dir.normalize();

            uint Nrays_total = Nprimitives * band.directRayCount;

            DirectRaysFunctor functor(d_primitives, d_flux, source_dir, source_flux,
                                     Nprimitives, band.directRayCount, rng);
            Kokkos::parallel_for("DirectRays", Nrays_total, functor);
            Kokkos::fence();
        }
    }

    // 2. Process diffuse radiation
    if (band.diffuseFlux > 0) {
        uint Nrays_total = Nprimitives * band.diffuseRayCount;

        DiffuseRaysFunctor functor(d_primitives, d_flux, band.diffuseFlux,
                                  Nprimitives, band.diffuseRayCount, rng, Nrays_total * 2);
        Kokkos::parallel_for("DiffuseRays", Nrays_total, functor);
        Kokkos::fence();
    }

    // 3. Add emission
    if (band.emissionFlag) {
        EmissionFunctor emission_functor(d_primitives, d_flux);
        Kokkos::parallel_for("Emission", Nprimitives, emission_functor);
        Kokkos::fence();
    }

    // 4. Handle scattering
    if (band.scatteringDepth > 0) {
        Kokkos::View<float*> d_flux_scatter("flux_scatter", Nprimitives);

        for (uint iter = 0; iter < band.scatteringDepth; iter++) {
            Kokkos::deep_copy(d_flux_scatter, 0.0f);

            uint Nrays_total = Nprimitives * band.diffuseRayCount;
            ScatteringFunctor scatter_functor(d_primitives, d_flux, d_flux_scatter,
                                             Nprimitives, band.diffuseRayCount, rng,
                                             Nrays_total * (iter + 10));
            Kokkos::parallel_for("Scattering", Nrays_total, scatter_functor);
            Kokkos::fence();

            // Add scattered contribution to main flux
            Kokkos::parallel_for("AddScatter", Nprimitives, KOKKOS_LAMBDA(const uint i) {
                d_flux(i) += d_flux_scatter(i);
            });
            Kokkos::fence();
        }
    }

    // Copy results back to host and update context
    auto h_flux = Kokkos::create_mirror_view(d_flux);
    Kokkos::deep_copy(h_flux, d_flux);

    std::string flux_label = "radiation_flux_" + label;
    for (size_t i = 0; i < Nprimitives; i++) {
        context->setPrimitiveData(context_UUIDs[i], flux_label.c_str(), h_flux(i));
    }
}

// Multi-band execution
void RadiationModel::runBand_kokkos(const std::vector<std::string> &labels) {
    for (const auto &label : labels) {
        runBand_kokkos(label);
    }
}

#endif // HELIOS_USE_KOKKOS
