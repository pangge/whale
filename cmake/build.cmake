# ----------------------------------------------------------------------------
# section: compile flags
# ----------------------------------------------------------------------------
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
endif()

if(BUILD_DEBUG)
    wl_add_compile(GCC FLAGS -std=c++11 -Wall -ldl -fPIC)
    wl_add_compile(GCC FLAGS -O0 -g)
else()
    wl_add_compile(GCC FLAGS -std=c++11 -Wall -ldl -fPIC)
    wl_add_compile(GCC FLAGS -O3)
endif()

wl_add_compile(GCC FLAG -lncurses)
wl_add_compile(GCC FLAG -Wno-sign-compare)
wl_add_compile(GCC FLAG -Wno-narrowing)
wl_add_compile(GCC FLAG -Wno-unused-command-line-argument)
#wl_add_compile(GCC FLAG -Wno-return-local-addr)
wl_add_compile(GCC FLAG -Wno-return-type)
wl_add_compile(GCC FLAG -Wno-unused-variable)
wl_add_compile(GCC FLAG -Wno-reorder)
wl_add_compile(GCC FLAG -Wno-int-to-pointer-cast)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma -fopenmp -mavx512f -mavx512cd -mavx512er -mavx512pf")

macro(find_openmp) 
    find_package(OpenMP REQUIRED) 
    if(OPENMP_FOUND OR OpenMP_CXX_FOUND) 
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}") 
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
        wl_msg(INFO STR "Found openmp in ${OPENMP_INCLUDE_DIR}") 
        wl_msg(INFO STR " |--openmp cflags: ${OpenMP_C_FLAGS}") 
        wl_msg(INFO STR " |--openmp cxxflags: ${OpenMP_CXX_FLAGS}") 
        wl_msg(INFO STR " |--openmp link flags: ${OpenMP_EXE_LINKER_FLAGS}") 
    else() 
        wl_msg(ERROR STR "Could not found openmp !") 
    endif() 
endmacro()

# ----------------------------------------------------------------------------
# section: build shared or static library
# ----------------------------------------------------------------------------
function(cc_library TARGET_NAME)
  	set(options STATIC static SHARED shared)
  	set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS LINK_LIBS)
  	cmake_parse_arguments(LIB "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  	if(LIB_SRCS)
        if(LIB_SHARED OR LIB_shared)
            add_library(${TARGET_NAME} SHARED ${LIB_SRCS})
        elseif(LIB_STATIC OR LIB_static)
            add_library(${TARGET_NAME} STATIC ${LIB_SRCS})
        else()
            wl_msg(ERROR STR "$cc_library's options must be set one of (STATIC static SHARED shared)")
        endif()
        if(LIB_DEPS OR LIB_LINK_LIBS) 
            foreach(dep ${LIB_DEPS})
                add_dependencies(${TARGET_NAME} ${dep})
                target_link_libraries(${TARGET_NAME} ${dep})
            endforeach()
            foreach(link_lib ${LIB_LINK_LIBS})
                target_link_libraries(${TARGET_NAME} ${link_lib})
            endforeach()
        endif()
    endif(LIB_SRCS)
endfunction()

function(cc_binary EXEC_NAME)
    set(options "")
  	set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS LINK_LIBS)
    cmake_parse_arguments(BINARY "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${EXEC_NAME} ${BINARY_SRCS})
    if(BINARY_DEPS OR BINARY_LINK_LIBS)
        foreach(dep ${BINARY_DEPS})
            add_dependencies(${EXEC_NAME} ${dep})
            target_link_libraries(${EXEC_NAME} ${dep})
        endforeach()
        foreach(link_lib ${BINARY_LINK_LIBS})
            target_link_libraries(${EXEC_NAME} ${link_lib})
        endforeach()
    endif()
    get_property(os_dependency_modules GLOBAL PROPERTY OS_DEPENDENCY_MODULES)
    target_link_libraries(${EXEC_NAME} ${os_dependency_modules})
endfunction()
