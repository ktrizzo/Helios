/** \file "RayTracing_kokkos.h" Kokkos-compatible ray-geometry intersection utilities
 *
 *  Copyright (C) 2016-2024 Brian Bailey
 *  Adapted from PlantSimulationLab/Helios CollisionDetection plugin
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, version 2.
 */

#ifndef RAYTRACING_KOKKOS_H
#define RAYTRACING_KOKKOS_H

#include <Kokkos_Core.hpp>
#include "global.h"

namespace helios {

// Ray-primitive intersection tolerance
constexpr float RAY_EPSILON = 1e-8f;

/**
 * \brief Ray-triangle intersection using Möller-Trumbore algorithm
 *
 * \param[in] origin Ray origin point
 * \param[in] direction Ray direction (must be normalized)
 * \param[in] v0 First triangle vertex
 * \param[in] v1 Second triangle vertex
 * \param[in] v2 Third triangle vertex
 * \param[out] distance Distance along ray to intersection point
 * \return true if ray intersects triangle, false otherwise
 */
KOKKOS_INLINE_FUNCTION
bool rayTriangleIntersect(const vec3 &origin, const vec3 &direction,
                          const vec3 &v0, const vec3 &v1, const vec3 &v2,
                          float &distance) {
    // Möller-Trumbore triangle intersection algorithm (optimized - no vec3 temporaries)

    // Compute triangle edges directly as components (avoid vec3 constructors)
    float edge1_x = v1.x - v0.x, edge1_y = v1.y - v0.y, edge1_z = v1.z - v0.z;
    float edge2_x = v2.x - v0.x, edge2_y = v2.y - v0.y, edge2_z = v2.z - v0.z;

    // Cross product: h = direction × edge2 (computed directly)
    float h_x = direction.y * edge2_z - direction.z * edge2_y;
    float h_y = direction.z * edge2_x - direction.x * edge2_z;
    float h_z = direction.x * edge2_y - direction.y * edge2_x;

    // Dot product: a = edge1 · h
    float a = edge1_x * h_x + edge1_y * h_y + edge1_z * h_z;

    if (a > -RAY_EPSILON && a < RAY_EPSILON) {
        return false; // Ray is parallel to triangle
    }

    float f = 1.0f / a;

    // Vector s = origin - v0 (computed as components)
    float s_x = origin.x - v0.x, s_y = origin.y - v0.y, s_z = origin.z - v0.z;

    // u = f * (s · h)
    float u = f * (s_x * h_x + s_y * h_y + s_z * h_z);

    if (u < -RAY_EPSILON || u > 1.0f + RAY_EPSILON) {
        return false;
    }

    // Cross product: q = s × edge1 (computed directly)
    float q_x = s_y * edge1_z - s_z * edge1_y;
    float q_y = s_z * edge1_x - s_x * edge1_z;
    float q_z = s_x * edge1_y - s_y * edge1_x;

    // v = f * (direction · q)
    float v = f * (direction.x * q_x + direction.y * q_y + direction.z * q_z);

    if (v < -RAY_EPSILON || u + v > 1.0f + RAY_EPSILON) {
        return false;
    }

    // t = f * (edge2 · q) - computed directly as dot product
    float t = f * (edge2_x * q_x + edge2_y * q_y + edge2_z * q_z);

    if (t > RAY_EPSILON) {
        distance = t;
        return true;
    }

    return false; // Line intersection but not ray intersection
}

/**
 * \brief Ray-patch (quadrilateral) intersection
 *
 * \param[in] origin Ray origin point
 * \param[in] direction Ray direction (must be normalized)
 * \param[in] v0 First patch vertex (anchor)
 * \param[in] v1 Second patch vertex
 * \param[in] v2 Third patch vertex
 * \param[in] v3 Fourth patch vertex
 * \param[out] distance Distance along ray to intersection point
 * \return true if ray intersects patch, false otherwise
 */
KOKKOS_INLINE_FUNCTION
bool rayPatchIntersect(const vec3 &origin, const vec3 &direction,
                       const vec3 &v0, const vec3 &v1, const vec3 &v2, const vec3 &v3,
                       float &distance) {
    // Patch (quadrilateral) intersection using radiation model algorithm

    // Calculate patch vectors and normal
    vec3 anchor = v0;

    // Compute normal = (v1-v0) × (v2-v0)
    float edge1_x = v1.x - v0.x, edge1_y = v1.y - v0.y, edge1_z = v1.z - v0.z;
    float edge2_x = v2.x - v0.x, edge2_y = v2.y - v0.y, edge2_z = v2.z - v0.z;

    float normal_x = edge1_y * edge2_z - edge1_z * edge2_y;
    float normal_y = edge1_z * edge2_x - edge1_x * edge2_z;
    float normal_z = edge1_x * edge2_y - edge1_y * edge2_x;

    // Normalize
    float normal_mag = Kokkos::sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z);
    if (normal_mag < RAY_EPSILON) return false;

    normal_x /= normal_mag;
    normal_y /= normal_mag;
    normal_z /= normal_mag;

    vec3 a = make_vec3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z); // First edge vector
    vec3 b = make_vec3(v3.x - v0.x, v3.y - v0.y, v3.z - v0.z); // Second edge vector

    // Ray-plane intersection
    float denom = direction.x * normal_x + direction.y * normal_y + direction.z * normal_z;

    if (Kokkos::fabs(denom) > RAY_EPSILON) { // Not parallel to plane
        float anchor_to_origin_x = anchor.x - origin.x;
        float anchor_to_origin_y = anchor.y - origin.y;
        float anchor_to_origin_z = anchor.z - origin.z;

        float t = (anchor_to_origin_x * normal_x + anchor_to_origin_y * normal_y + anchor_to_origin_z * normal_z) / denom;

        if (t > RAY_EPSILON && t < 1e8f) { // Valid intersection distance
            // Find intersection point
            vec3 p = make_vec3(origin.x + direction.x * t,
                              origin.y + direction.y * t,
                              origin.z + direction.z * t);
            vec3 d = make_vec3(p.x - anchor.x, p.y - anchor.y, p.z - anchor.z);

            // Project onto patch coordinate system
            float ddota = d.x * a.x + d.y * a.y + d.z * a.z;
            float ddotb = d.x * b.x + d.y * b.y + d.z * b.z;
            float adota = a.x * a.x + a.y * a.y + a.z * a.z;
            float bdotb = b.x * b.x + b.y * b.y + b.z * b.z;

            // Check if point is within patch bounds (with epsilon tolerance for edge cases)
            if (ddota >= -RAY_EPSILON && ddota <= adota + RAY_EPSILON &&
                ddotb >= -RAY_EPSILON && ddotb <= bdotb + RAY_EPSILON) {
                distance = t;
                return true;
            }
        }
    }

    return false;
}

/**
 * \brief Calculate surface normal at intersection point
 *
 * \param[in] v0 First vertex
 * \param[in] v1 Second vertex
 * \param[in] v2 Third vertex
 * \param[in] ray_direction Ray direction (to orient normal correctly)
 * \return Surface normal (pointing towards ray origin)
 */
KOKKOS_INLINE_FUNCTION
vec3 calculateSurfaceNormal(const vec3 &v0, const vec3 &v1, const vec3 &v2,
                           const vec3 &ray_direction) {
    // Calculate normal from first two edges
    float edge1_x = v1.x - v0.x, edge1_y = v1.y - v0.y, edge1_z = v1.z - v0.z;
    float edge2_x = v2.x - v0.x, edge2_y = v2.y - v0.y, edge2_z = v2.z - v0.z;

    float normal_x = edge1_y * edge2_z - edge1_z * edge2_y;
    float normal_y = edge1_z * edge2_x - edge1_x * edge2_z;
    float normal_z = edge1_x * edge2_y - edge1_y * edge2_x;

    float magnitude = Kokkos::sqrt(normal_x*normal_x + normal_y*normal_y + normal_z*normal_z);

    if (magnitude > RAY_EPSILON) {
        normal_x /= magnitude;
        normal_y /= magnitude;
        normal_z /= magnitude;

        // Ensure normal points opposite to ray direction
        float dot = normal_x * ray_direction.x + normal_y * ray_direction.y + normal_z * ray_direction.z;
        if (dot > 0) {
            normal_x = -normal_x;
            normal_y = -normal_y;
            normal_z = -normal_z;
        }
    } else {
        // Degenerate triangle - return reversed ray direction
        normal_x = -ray_direction.x;
        normal_y = -ray_direction.y;
        normal_z = -ray_direction.z;
    }

    return make_vec3(normal_x, normal_y, normal_z);
}

} // namespace helios

#endif // RAYTRACING_KOKKOS_H
