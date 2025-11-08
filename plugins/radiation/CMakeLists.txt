cmake_minimum_required(VERSION 3.16)
project(helios CXX)

include_directories(include)
include_directories(../../core/include)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules/")

# Core source files (always compiled)
set(RADIATION_SOURCES
    "src/RadiationModel.cpp"
    "src/selfTest.cpp"
    "src/CameraCalibration.cpp"
)

# Try to find CUDA (not required - will fallback to Kokkos if not found)
find_package(CUDA QUIET)

if(CUDA_FOUND)
    #=============================================================================
    # CUDA/OptiX PATH - Use existing high-performance OptiX ray tracer
    #=============================================================================
    message(STATUS "CUDA found - building Radiation plugin with OptiX")

    include_directories(${CUDA_INCLUDE_DIRS})

    # Determine OptiX path based on platform
    if(UNIX AND NOT APPLE)
        if( OPTIX_VERSION_LEGACY )
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-5.1.0/)
            message(STATUS "Using legacy OptiX version 5.1")
        else()
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/linux64-6.5.0/)
            message(STATUS "Using OptiX version 6.5")
        endif()
        include_directories(${OPTIX_PATH}include)
        add_subdirectory(${OPTIX_PATH} "plugins/radiation")
        set(OPTIX_LIBRARY ${OPTIX_PATH}lib64/liboptix.so)
    elseif(APPLE)
        set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/mac64-4.0.2/)
        include_directories(${OPTIX_PATH}include)
        add_subdirectory(${OPTIX_PATH})
        set(OPTIX_LIBRARY ${OPTIX_PATH}lib64/liboptix.dylib)
    elseif(WIN32)
        if( OPTIX_VERSION_LEGACY )
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-5.1.1/)
            set(OPTIX_LIB "optix.51.lib")
            set(OPTIX_DLL "optix.51.dll")
            message(STATUS "Using legacy OptiX version 5.1")
        else()
            set(OPTIX_PATH ${CMAKE_CURRENT_SOURCE_DIR}/lib/OptiX/windows64-6.5.0/)
            set(OPTIX_LIB "optix.6.5.0.lib")
            set(OPTIX_DLL "optix.6.5.0.dll")
            message(STATUS "Using OptiX version 6.5")
        endif()
        include_directories(${OPTIX_PATH}include)
        set(OPTIX_LIBRARY "${OPTIX_PATH}lib64/${OPTIX_LIB}")
    endif()

    if(NOT DEFINED OPTIX_PATH)
        message( FATAL_ERROR "ERROR: Could not determine operating system for OptiX path." )
    endif()

    # Create library with OptiX
    add_library(radiation STATIC ${RADIATION_SOURCES})
    target_link_libraries(radiation ${CUDA_LIBRARIES} ${OPTIX_LIBRARY})
    target_compile_definitions(radiation PRIVATE HELIOS_USE_OPTIX)

    # Configure CUDA compilation flags
    if( OPTIX_VERSION_LEGACY )
        set( GPU_COMPUTE "compute_35" )
    else()
        set( GPU_COMPUTE "compute_50" )
    endif()

    set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math --gpu-architecture=${GPU_COMPUTE} -Wno-deprecated-gpu-targets" )

    if( CMAKE_BUILD_TYPE STREQUAL Debug OR NOT DEFINED CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "" )
        set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -O0" )
        set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_0 ON )
    else()
        set( CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3" )
        set( OPTIX_COMPILE_OPTIMIZATION_LEVEL_3 ON )
    endif()

    # Compile CUDA kernels to PTX
    set( CUDA_SOURCES "src/primitiveIntersection.cu;src/rayGeneration.cu;src/rayHit.cu;" )
    cuda_compile_ptx(cuda_ptx_files ${CUDA_SOURCES})
    add_custom_target(ptx ALL DEPENDS ${cuda_ptx_files} ${CUDA_SOURCES} ${OPTIX_LIBRARIES} SOURCES ${CUDA_SOURCES})
    add_dependencies(radiation ptx)

    # Handle PTX filename changes in CMake 3.7+
    if( CMAKE_VERSION VERSION_GREATER 3.6 )
        add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_primitiveIntersection.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_primitiveIntersection.cu.ptx DEPENDS ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_primitiveIntersection.cu.ptx )
        add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayGeneration.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayGeneration.cu.ptx DEPENDS ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayGeneration.cu.ptx )
        add_custom_command( TARGET ptx POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayHit.cu.ptx ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_generated_rayHit.cu.ptx DEPENDS ${CMAKE_BINARY_DIR}/plugins/radiation/cuda_compile_ptx_1_generated_rayHit.cu.ptx )
    endif()

    # Windows DLL copying
    if(WIN32)
        add_custom_command( TARGET radiation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}lib64/${OPTIX_LIB}" "${CMAKE_BINARY_DIR}/." )
        add_custom_command( TARGET radiation POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy "${OPTIX_PATH}bin64/${OPTIX_DLL}" "${CMAKE_BINARY_DIR}/." )
    endif()

    set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${OPTIX_PATH}include;${CUDA_INCLUDE_DIRS};${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

else()
    #=============================================================================
    # KOKKOS PATH - GPU-agnostic fallback when CUDA not available
    #=============================================================================
    message(STATUS "CUDA not found - building Radiation plugin with Kokkos (GPU-agnostic)")

    # Find Kokkos package
    # Try to find in common locations if not specified
    if(NOT DEFINED Kokkos_DIR AND NOT DEFINED CMAKE_PREFIX_PATH)
        set(Kokkos_DIR "/root/kokkos-install/lib/cmake/Kokkos" CACHE PATH "Kokkos directory")
    endif()
    find_package(Kokkos REQUIRED HINTS ${Kokkos_DIR})

    # Add Kokkos implementation source
    list(APPEND RADIATION_SOURCES "src/RadiationModel_kokkos.cpp")

    # Create library with Kokkos
    add_library(radiation STATIC ${RADIATION_SOURCES})
    target_compile_definitions(radiation PUBLIC HELIOS_USE_KOKKOS)
    target_compile_features(radiation PUBLIC cxx_std_17)
    # Link Kokkos using plain signature (no PUBLIC/PRIVATE keywords) to match CMake_project.txt
    target_link_libraries(radiation Kokkos::kokkos)

    set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

    message(STATUS "Note: Kokkos radiation model is a simplified implementation")
    message(STATUS "      For best performance on NVIDIA GPUs, install CUDA and rebuild")
endif()

# Copy data files (needed for both OptiX and Kokkos builds)
file( COPY include/disk.png DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/)
file( COPY spectral_data DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )
file( COPY camera_light_models DESTINATION ${CMAKE_BINARY_DIR}/plugins/radiation/ )
