project(whale)
cmake_minimum_required(VERSION 3.2)

# compile options
option(BUILD_DEBUG "debug or not" OFF)

# platform
option(BUILD_MOBILE "build on android platform" ON)
option(BUILD_GPU "build on " OFF)
option(BUILD_X86 "build on x86 platform" OFF)


# configure environment
include(cmake/utils.cmake)

if(BUILD_MOBILE)
    include(cmake/android.cmake)
endif()

if(BUILD_GPU)
    option(WITH_CUDA "" ON)
    if(WITH_CUDA)
        include(cmake/cuda.cmake)
    endif()
endif()

if(BUILD_X86)
endif()

#foreach(SRC_NAME ${cuda_source})
#    #unpack the dir "/"
#    string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
#    list(GET SEXY_LIST -1 TEST_CASE_NAME)
#    #get the file name without suffix
#    string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
#    list(GET SEXY_LIST 0 TEST_CASE_NAME)
#    cuda_add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
#    target_link_libraries(${TEST_CASE_NAME} ${CUDA_TEST_LINKER_LIBS})
#    #set_target_properties(${TEST_CASE_NAME} PROPERTIES
#    #                    RUNTIME_OUTPUT_DIRECTORY
#    #                    ${PROJECT_SOURCE_DIR}/output/unit_test)
#    find_package(Threads) 
#    target_link_libraries(${TEST_CASE_NAME} ${CMAKE_THREAD_LIBS_INIT}) # For pthreads
#endforeach()
#
#foreach(SRC_NAME ${source})
#    #unpack the dir "/"
#    string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
#    list(GET SEXY_LIST -1 TEST_CASE_NAME)
#    #get the file name without suffix
#    string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
#    list(GET SEXY_LIST 0 TEST_CASE_NAME)
#    add_executable(${TEST_CASE_NAME}  ${SRC_NAME})
#    target_link_libraries(${TEST_CASE_NAME} ${CUDA_TEST_LINKER_LIBS})
#    #set_target_properties(${TEST_CASE_NAME} PROPERTIES
#    #                    RUNTIME_OUTPUT_DIRECTORY
#    #                    ${PROJECT_SOURCE_DIR}/output/unit_test)
#    find_package(Threads)
#    target_link_libraries(${TEST_CASE_NAME} ${CMAKE_THREAD_LIBS_INIT}) # For pthreads
#endforeach()
