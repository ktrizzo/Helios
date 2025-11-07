cmake_minimum_required(VERSION 3.16)
project(helios CXX)

include_directories(include)
include_directories(../../core/include)

if(NOT DEFINED CMAKE_SUPPRESS_DEVELOPER_WARNINGS)
     set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS 1 CACHE INTERNAL "No dev warnings")
endif()

# Find Kokkos package
find_package(Kokkos REQUIRED)

# Create library with Kokkos-enabled sources
add_library(energybalance STATIC
     "src/EnergyBalanceModel.cpp"
     "src/EnergyBalanceModel_kokkos.cpp"
)

# Link Kokkos
target_link_libraries(energybalance Kokkos::kokkos)

# Set C++ standard (Kokkos requires C++17 or later)
target_compile_features(energybalance PUBLIC cxx_std_17)

set( PLUGIN_INCLUDE_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/include;${PLUGIN_INCLUDE_PATHS}" PARENT_SCOPE )

