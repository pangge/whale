project(DeltaExport)
cmake_minimum_required(VERSION 2.8)

set(DELTA_EXPORT_ROOT ${PROJECT_SOURCE_DIR})
include_directories(${DELTA_EXPORT_ROOT}/whale)

# compile options
option(BUILD_DEBUG "debug or not" ON)
option(BUILD_SHARED "build shared or static lib" ON)

# platform
option(BUILD_MOBILE "build on android platform" OFF)
option(BUILD_GPU "build on " OFF)
option(BUILD_X86 "build on x86 platform" ON)

# configure environment
include(cmake/utils.cmake)
include(cmake/build.cmake)

if(BUILD_MOBILE)
    add_definitions(-DBUILD_MOBILE)
    option(WITH_ARM "" OFF)
    if(WITH_ARM)
        add_definitions(-DWITH_ARM)
    endif()
    option(WITH_VULCAN "" OFF)
    if(WITH_VULCAN)
        add_definitions(-DWITH_VULCAN)
    endif()
endif()

if(BUILD_GPU)
    option(WITH_CUDA "" OFF)
    if(WITH_CUDA)
        add_definitions(-DWITH_CUDA)
        include(cmake/cuda.cmake)
    endif()
endif()

if(BUILD_X86)
    add_definitions(-DBUILD_X86)
    option(WITH_OMP "" ON)
    if(WITH_OMP) 
        add_definitions(-DWITH_OMP)
        find_openmp()
    endif()
endif()

add_subdirectory(${DELTA_EXPORT_ROOT}/whale)
add_subdirectory(${DELTA_EXPORT_ROOT}/tests)

