
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find opencl libraries and include directories.
#
#   https://www.khronos.org/opencl/
#   https://developer.nvidia.com/cuda-downloads
#   https://software.intel.com/en-us/intel-opencl
#   http://developer.amd.com/tools-and-sdks/opencl-zone/amd-accelerated-parallel-processing-app-sdk/
#
#   Variables defined after execution:
#
#	OPENCL_FOUND
#	OPENCL_INCLUDE_DIR
#	OPENCL_INCLUDE_DIRS
#	OPENCL_LIBRARIES
#	OPENCL_LIBRARY
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
#   Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{OPENCL_ROOT}/include"
        "$ENV{AMDAPPSDKROOT}/include"
        "$ENV{INTELOCLSDKROOT}/include"
        "$ENV{NVSDKCOMPUTE_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{OPENCL_ROOT}/lib"
        "$ENV{AMDAPPSDKROOT}/lib"
        "$ENV{INTELOCLSDKROOT}/lib"
        "$ENV{NVSDKCOMPUTE_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{OPENCL_ROOT}/include"
        "$ENV{AMDAPPSDKROOT}/include"
        "$ENV{INTELOCLSDKROOT}/include"
        "$ENV{NVSDKCOMPUTE_ROOT}/include"
        "/usr/include"
        "/usr/opencl/include"
        "/usr/local/include"
        "/usr/local/opencl/include"
        "~/usr/include"
        "~/usr/opencl/include"
        "~/usr/local/include"
        "~/usr/local/opencl/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{OPENCL_ROOT}/lib"
        "$ENV{AMDAPPSDKROOT}/lib"
        "$ENV{INTELOCLSDKROOT}/lib"
        "$ENV{NVSDKCOMPUTE_ROOT}/lib"
        "/usr/lib"
        "/usr/opencl/lib"
        "/usr/local/lib"
        "/usr/local/opencl/lib"
        "~/usr/lib"
        "~/usr/opencl/lib"
        "~/usr/local/lib"
        "~/usr/local/opencl/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "unsupported operating system")

endif()

#
#   Detect include paths based on the above search paths and hints.
#

find_path(OPENCL_INCLUDE_DIR
            NAMES
            "CL/opencl.h"
            HINTS
            ${INCLUDE_SEARCH_HINTS}
            PATHS
            ${INCLUDE_SEARCH_PATHS}
            DOC "absolute path to opencl include directory")

#
#   Set library names.
#

set(LIBRARY_LIB_NAME_LIST
    "OpenCL"
)

#
#   Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "OPENCL_LIBRARY"
)

#
#   Detect the paths of the above libraries and save them in the appropriate 
#   variable.
#

list(LENGTH LIBRARY_LIB_VARIABLE_NAME_LIST LIBRARY_LIB_NAME_LIST_LENGTH)
MATH(EXPR LIBRARY_LIB_NAME_LIST_LENGTH "${LIBRARY_LIB_NAME_LIST_LENGTH}-1")
foreach(i RANGE ${LIBRARY_LIB_NAME_LIST_LENGTH})
    list(GET LIBRARY_LIB_VARIABLE_NAME_LIST ${i} LIB_NAME_VAR)
    list(GET LIBRARY_LIB_NAME_LIST ${i} LIB_NAME)
    find_library(${LIB_NAME_VAR}
                    NAMES
                    ${LIB_NAME}
                    HINTS
                    ${LIBRARY_SEARCH_HINTS}
                    PATHS
                    ${LIBRARY_SEARCH_PATHS}
                    DOC "absolute path to ${LIB_NAME} library")
    unset(LIB_NAME_VAR)
    unset(LIB_NAME)
endforeach(i)

#
#   Combine all library paths into one variable.
#

set(OPENCL_INCLUDE_DIRS
    ${OPENCL_INCLUDE_DIR}
)

set(OPENCL_LIBRARIES
    ${OPENCL_LIBRARY}
)

#
#   Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OPENCL DEFAULT_MSG
                                  OPENCL_LIBRARIES OPENCL_INCLUDE_DIRS)

#
#   Unset all the temporary variables.
#

unset(LIBRARY_LIB_VARIABLE_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST_LENGTH)
unset(INCLUDE_SEARCH_PATHS)
unset(INCLUDE_SEARCH_HINTS)
unset(LIBRARY_SEARCH_PATHS)
unset(LIBRARY_SEARCH_HINTS)
