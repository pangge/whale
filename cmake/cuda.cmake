set(WHALE_LINK_LIBS "")

macro(find_cuda)
    set(CUDA_VERBOSE_BUILD ON)
    find_package(CUDA 9.0 QUIET)
    if(CUDA_FOUND)
        wl_msg(INFO STR "Found cuda in ${CUDA_INCLUDE_DIRS}")
        include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
        list(APPEND WHALE_LINK_LIBS ${CUDA_CUBLAS_LIBRARIES})
        list(APPEND WHALE_LINK_LIBS ${CUDA_curand_LIBRARY})
        list(APPEND WHALE_LINK_LIBS ${CUDA_LIBRARIES})
    endif()
endmacro()

macro(find_cudnn)
    set(CUDNN_ROOT "" CACHE PATH "CUDNN root dir.")
    find_path(CUDNN_INCLUDE_DIR cudnn.h PATHS ${CUDNN_ROOT}
                                              $ENV{CUDNN_ROOT}
                                              $ENV{CUDNN_ROOT}/include
                                              ${ANAKIN_ROOT}/third-party/cudnn/include NO_DEFAULT_PATH)
    if(BUILD_SHARED)
        find_library(CUDNN_LIBRARY NAMES libcudnn.so
                                   PATHS ${CUDNN_INCLUDE_DIR}/../lib64/ ${CUDNN_INCLUDE_DIR}/ 
                                   DOC "library path for cudnn.")
    else()
        find_library(CUDNN_LIBRARY NAMES libcudnn_static.a
                               	   PATHS ${CUDNN_INCLUDE_DIR}/../lib64/
                               	   DOC "library path for cudnn.")
    endif()
    if(CUDNN_INCLUDE_DIR AND CUDNN_LIBRARY)
        set(CUDNN_FOUND YES)
        file(READ ${CUDNN_INCLUDE_DIR}/cudnn.h CUDNN_FILE_VERSION)
        string(REGEX MATCH "define CUDNN_MAJOR * +([0-9]+)"
                CUDNN_VERSION_MAJOR "${CUDNN_FILE_VERSION}")
        string(REGEX REPLACE "define CUDNN_MAJOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MAJOR "${CUDNN_VERSION_MAJOR}")
        string(REGEX MATCH "define CUDNN_MINOR * +([0-9]+)"
                CUDNN_VERSION_MINOR "${CUDNN_FILE_VERSION}")
        string(REGEX REPLACE "define CUDNN_MINOR * +([0-9]+)" "\\1"
                CUDNN_VERSION_MINOR "${CUDNN_VERSION_MINOR}")
        string(REGEX MATCH "define CUDNN_PATCHLEVEL * +([0-9]+)"
                CUDNN_VERSION_PATCH "${CUDNN_FILE_VERSION}")
        string(REGEX REPLACE "define CUDNN_PATCHLEVEL * +([0-9]+)" "\\1"
                CUDNN_VERSION_PATCH "${CUDNN_VERSION_PATCH}")
        #message(STATUS "Found cudnn version ${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH}")
        set(Cudnn_VERSION ${CUDNN_VERSION_MAJOR}.${CUDNN_VERSION_MINOR}.${CUDNN_VERSION_PATCH})
        string(COMPARE LESS "${CUDNN_VERSION_MAJOR}" 6 CUDNN_TOO_LOW)
        if(CUDNN_TOO_LOW)
            message(FATAL_ERROR " Cudnn version should > 6 ")
        endif()
    endif()
    if(CUDNN_FOUND)
        include_directories(SYSTEM ${CUDNN_INCLUDE_DIR})
        list(APPEND WHALE_LINK_LIBS ${CUDNN_LIBRARY})
        wl_msg(INFO STR "Found cudnn: ${CUDNN_INCLUDE_DIR}")
    else()
        wl_msg(ERROR STR "Could not find cudnn library in: ${CUDNN_ROOT}")
    endif()
endmacro()


set(ARCH 61 CACHE STRING "NV GPU arch" )
# we can't use debug mode when use NVCC due to the bug of tensorflow : https://github.com/tensorflow/tensorflow/issues/22766
if(BUILD_DEBUG) 
    wl_add_compile(NVCC FLAGS -Xcompiler -fPIC -std=c++11)
    wl_add_compile(NVCC FLAGS -O0 -G -g)
    wl_add_compile(NVCC FLAG "-gencode arch=compute_${ARCH},code=sm_${ARCH}")
    wl_add_compile(NVCC FLAG -Wno-deprecated-gpu-targets)
else()
    wl_add_compile(NVCC FLAGS -Xcompiler -fPIC -std=c++11)
    wl_add_compile(NVCC FLAGS -O3 -DNDEBUG)
    wl_add_compile(NVCC FLAG "-gencode arch=compute_${ARCH},code=sm_${ARCH}")
    wl_add_compile(NVCC FLAG -Wno-deprecated-gpu-targets)  
endif()

# find cuda 
find_cuda()
# find cudnn
find_cudnn()


