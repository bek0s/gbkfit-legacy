
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find mpfit libraries and include directories.
#
#   http://www.physics.wisc.edu/~craigm/idl/cmpfit.html
#
#   Variables defined after execution:
#
#	MPFIT_FOUND
#	MPFIT_INCLUDE_DIR
#	MPFIT_INCLUDE_DIRS
#	MPFIT_LIBRARY
#	MPFIT_LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{MPFIT_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{MPFIT_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{MPFIT_ROOT}/include"
        "/usr/include"
        "/usr/mpfit/include"
        "/usr/local/include"
        "/usr/local/mpfit/include"
        "~/usr/include"
        "~/usr/mpfit/include"
        "~/usr/local/include"
        "~/usr/local/mpfit/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{MPFIT_ROOT}/lib"
        "/usr/lib"
        "/usr/mpfit/lib"
        "/usr/local/lib"
        "/usr/local/mpfit/lib"
        "~/usr/lib"
        "~/usr/mpfit/lib"
        "~/usr/local/lib"
        "~/usr/local/mpfit/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "Unsupported operating system.")

endif()

#
# Detect include paths based on the above search paths and hints.
#

find_path(MPFIT_INCLUDE_DIR
            NAMES
            "mpfit.h"
            HINTS
            ${INCLUDE_SEARCH_HINTS}
            PATHS
            ${INCLUDE_SEARCH_PATHS}
            DOC "Absolute path to mpfit include directory.")

#
# Set library names.
#

set(LIBRARY_LIB_NAME_LIST
    "mpfit"
)

#
# Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "MPFIT_LIBRARY"
)

#
# Detect the paths of the above libraries and save them in the appropriate
# variable.
#

list(LENGTH LIBRARY_LIB_VARIABLE_NAME_LIST LIBRARY_LIB_NAME_LIST_LENGTH)
math(EXPR LIBRARY_LIB_NAME_LIST_LENGTH "${LIBRARY_LIB_NAME_LIST_LENGTH}-1")
foreach(i RANGE ${LIBRARY_LIB_NAME_LIST_LENGTH})
    list(GET LIBRARY_LIB_VARIABLE_NAME_LIST ${i} LIB_VAR_NAME)
    list(GET LIBRARY_LIB_NAME_LIST ${i} LIB_NAME)
    find_library(${LIB_VAR_NAME}
                    NAMES
                    ${LIB_NAME}
                    HINTS
                    ${LIBRARY_SEARCH_HINTS}
                    PATHS
                    ${LIBRARY_SEARCH_PATHS}
                    DOC "Absolute path to ${LIB_NAME} library.")
    unset(LIB_VAR_NAME)
    unset(LIB_NAME)
endforeach(i)

#
# Combine all library paths into one variable.
#

set(MPFIT_INCLUDE_DIRS
    ${MPFIT_INCLUDE_DIR}
)

set(MPFIT_LIBRARIES
    ${MPFIT_LIBRARY}
)

#
# Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPFIT DEFAULT_MSG
                                  MPFIT_LIBRARIES MPFIT_INCLUDE_DIRS)

#
# Unset all the temporary variables.
#

unset(LIBRARY_LIB_VARIABLE_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST_LENGTH)
unset(INCLUDE_SEARCH_PATHS)
unset(INCLUDE_SEARCH_HINTS)
unset(LIBRARY_SEARCH_PATHS)
unset(LIBRARY_SEARCH_HINTS)
