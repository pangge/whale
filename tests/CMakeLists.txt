
# fetch test files
wl_fetch_files_with_suffix(${WHALE_ROOT}/tests "cpp" WHALE_TESTS)


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

foreach(SRC_NAME ${WHALE_TESTS})
    #unpack the dir "/"
    string(REPLACE "/" ";" SEXY_LIST ${SRC_NAME})
    list(GET SEXY_LIST -1 TEST_CASE_NAME)
    #get the file name without suffix
    string(REPLACE "." ";" SEXY_LIST ${TEST_CASE_NAME})
    list(GET SEXY_LIST 0 TEST_CASE_NAME)
    find_package(Threads)
    cc_binary(${TEST_CASE_NAME} SRCS ${SRC_NAME} DEPS whale LINK_LIBS ${CMAKE_THREAD_LIBS_INIT})
endforeach()

