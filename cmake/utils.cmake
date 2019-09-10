# ----------------------------------------------------------------------------
# section: help to search src and include files
# ----------------------------------------------------------------------------
# fetch files(.cc .cpp .cu .c or .h .hpp etc.) in dir(search_dir)
# and save to parent scope var outputs
function(wl_fetch_files_with_suffix search_dir suffix outputs)
	exec_program(ls ${search_dir}
             ARGS "*.${suffix}"
             OUTPUT_VARIABLE OUTPUT
             RETURN_VALUE VALUE)
	if(NOT VALUE)
		string(REPLACE "\n" ";" OUTPUT_LIST "${OUTPUT}")
		set(abs_dir "")
		foreach(var ${OUTPUT_LIST})
			set(abs_dir ${abs_dir} ${search_dir}/${var})
            #wl_msg(WARN STR "fetch_result: ${abs_dir}")
		endforeach()
		set(${outputs} ${${outputs}} ${abs_dir} PARENT_SCOPE)
	else()
		#message(WARNING "wl_fetch_files_recursively ${BoldRed}failed${ColourReset}:\n"
		#                "real_dir:${BoldYellow}${search_dir}${ColourReset}\n"
		#                "suffix:*.${BoldYellow}${suffix}${ColourReset} \n")
	endif()
endfunction()

# recursively fetch files
function(wl_fetch_files_with_suffix_recursively search_dir suffix outputs)
	file(GLOB_RECURSE ${outputs} ${search_dir} "*.${suffix}")
	set(${outputs} ${${outputs}} PARENT_SCOPE)
endfunction()

# recursively fetch include dir
function(wl_fetch_include_recursively root_dir)
    if (IS_DIRECTORY ${root_dir})
        wl_msg(INFO STR "include dir: ${root_dir}")
		include_directories(${root_dir})
    endif()

    file(GLOB ALL_SUB RELATIVE ${root_dir} ${root_dir}/*)
    foreach(sub ${ALL_SUB})
        if (IS_DIRECTORY ${root_dir}/${sub})
            wl_fetch_include_recursively(${root_dir}/${sub})
        endif()
    endforeach()
endfunction()

# ----------------------------------------------------------------------------
# section: function for sending message with verbose (INFO, WARNING, ERROR)
# ----------------------------------------------------------------------------
if(NOT WIN32)
    string(ASCII 27 Esc)
    set(ColourReset "${Esc}[m")
    set(ColourBold  "${Esc}[1m")
    set(Red         "${Esc}[31m")
    set(Green       "${Esc}[32m")
    set(Yellow      "${Esc}[33m")
    set(Blue        "${Esc}[34m")
    set(Magenta     "${Esc}[35m")
    set(Cyan        "${Esc}[36m")
    set(White       "${Esc}[37m")
    set(BoldRed     "${Esc}[1;31m")
    set(BoldGreen   "${Esc}[1;32m")
    set(BoldYellow  "${Esc}[1;33m")
    set(BoldBlue    "${Esc}[1;34m")
    set(BoldMagenta "${Esc}[1;35m")
    set(BoldCyan    "${Esc}[1;36m")
    set(BoldWhite   "${Esc}[1;37m")
endif()

function(wl_msg)
    set(options INFO WARN ERROR)
    set(oneValueArgs STR)
    set(multiValueArgs ITEMS)
    cmake_parse_arguments(MSG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    if(MSG_INFO)     
        message(STATUS "${BoldGreen}INFO:${ColourReset} ${MSG_STR}")
        if(MSG_ITEMS)
            foreach(item ${MSG_ITEMS})
                message(STATUS "       |__ ${BoldWhite}${item}${ColourReset}")
            endforeach()
        endif()
    elseif(MSG_WARN)
        message(WARNING "${BoldYellow}WARN:${ColourReset} ${MSG_STR}")
        if(MSG_ITEMS)
            foreach(item ${MSG_ITEMS})
                message(WARNING "       |__ ${BoldWhite}${item}${ColourReset}")
            endforeach()
        endif()
    else() # msg error
        message(FATAL_ERROR "${BoldRed}ERROR:${ColourReset} ${MSG_STR}")
    endif()
endfunction()
