
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find multinest libraries and include directories.
#
#   http://ccpforge.cse.rl.ac.uk/gf/project/multinest/
#
#   Variables defined after execution:
#
#	MULTINEST_FOUND
#	MULTINEST_INCLUDE_DIR
#	MULTINEST_INCLUDE_DIRS
#       MULTINEST_MODULE_DIR
#       MULTINEST_MODULE_DIRS
#       MULTINEST_LIBRARY
#       MULTINEST_MPI_LIBRARY
#	MULTINEST_LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{MULTINEST_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(MODULE_SEARCH_PATHS
        "$ENV{MULTINEST_ROOT}/modules"
    )
    set(MODULE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{MULTINEST_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{MULTINEST_ROOT}/include"
        "/usr/include"
        "/usr/multinest/include"
        "/usr/local/include"
        "/usr/local/multinest/include"
        "~/usr/include"
        "~/usr/multinest/include"
        "~/usr/local/include"
        "~/usr/local/multinest/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(MODULE_SEARCH_PATHS
        "$ENV{MULTINEST_ROOT}/modules"
        "/usr/modules"
        "/usr/multinest/modules"
        "/usr/local/modules"
        "/usr/local/multinest/modules"
        "~/usr/modules"
        "~/usr/multinest/modules"
        "~/usr/local/modules"
        "~/usr/local/multinest/modules"
    )
    set(MODULE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{MULTINEST_ROOT}/lib"
        "/usr/lib"
        "/usr/multinest/lib"
        "/usr/local/lib"
        "/usr/local/multinest/lib"
        "~/usr/lib"
        "~/usr/multinest/lib"
        "~/usr/local/lib"
        "~/usr/local/multinest/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "Unsupported operating system.")

endif()

#
# Detect include paths based on the above search paths and hints.
#

find_path(MULTINEST_INCLUDE_DIR
            NAMES
            "multinest.h"
            HINTS
            ${INCLUDE_SEARCH_HINTS}
            PATHS
            ${INCLUDE_SEARCH_PATHS}
            DOC "Absolute path to multinest include directory.")

#
# Detect module paths based on the above search paths and hints.
#

find_path(MULTINEST_MODULE_DIR
            NAMES
            "nested.mod"
            HINTS
            ${MODULE_SEARCH_HINTS}
            PATHS
            ${MODULE_SEARCH_PATHS}
            DOC "Absolute path to multinest module directory.")

#
# Set library names.
#

set(LIBRARY_LIB_NAME_LIST
    "multinest"
    "multinest_mpi"
)

#
# Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "MULTINEST_LIBRARY"
    "MULTINEST_MPI_LIBRARY"
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

set(MULTINEST_INCLUDE_DIRS
    ${MULTINEST_INCLUDE_DIR}
)

set(MULTINEST_MODULE_DIRS
    ${MULTINEST_MODULE_DIR}
)

set(MULTINEST_LIBRARIES
    ${MULTINEST_LIBRARY}
    ${MULTINEST_MPI_LIBRARY}
)

#
# Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MULTINEST DEFAULT_MSG
                                  MULTINEST_LIBRARIES MULTINEST_INCLUDE_DIRS MULTINEST_MODULE_DIRS)

#
# Unset all the temporary variables.
#

unset(LIBRARY_LIB_VARIABLE_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST)
unset(LIBRARY_LIB_NAME_LIST_LENGTH)
unset(INCLUDE_SEARCH_PATHS)
unset(INCLUDE_SEARCH_HINTS)
unset(MODULE_SEARCH_PATHS)
unset(MODULE_SEARCH_HINTS)
unset(LIBRARY_SEARCH_PATHS)
unset(LIBRARY_SEARCH_HINTS)
