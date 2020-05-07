
set(WHALE ${WHALE_ROOT}/whale)
set(WHALE_SRC "")

wl_fetch_include_recursively(${WHALE})

# fetch source files
wl_fetch_files_with_suffix(${WHALE}/core "cpp" WHALE_SRC)
#wl_fetch_files_with_suffix(${WHALE}/operation "cpp" WHALE_SRC)
#wl_fetch_files_with_suffix(${WHALE}/sys/cli "cpp" WHALE_SRC)

# build library
if(BUILD_SHARED)
    find_package(Threads)
    cc_library(whale SHARED SRCS ${WHALE_SRC} LINK_LIBS ${WHALE_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT})
else()
    find_package(Threads)
    cc_library(whale STATIC SRCS ${WHALE_SRC} LINK_LIBS ${WHALE_LINK_LIBS} ${CMAKE_THREAD_LIBS_INIT})
endif()

