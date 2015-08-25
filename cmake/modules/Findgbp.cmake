
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find gbpCode libraries and include directories.
#
#   https://github.com/gbpoole/gbpCode/
#
#   Variables defined after execution:
#
#	GBP_FOUND
#	GBP_INCLUDE_DIR
#	GBP_INCLUDE_DIRS
#       GBP_LIBRARY
#       GBP_COSMO_LIBRARY
#       GBP_HALOS_LIBRARY
#       GBP_MATH_LIBRARY
#       GBP_SPH_LIBRARY
#       GBP_UTILS_LIBRARY
#	GBP_LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{GBP_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{GBP_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{GBP_ROOT}/include"
        "/usr/include"
        "/usr/gbp/include"
        "/usr/local/include"
        "/usr/local/gbp/include"
        "~/usr/include"
        "~/usr/gbp/include"
        "~/usr/local/include"
        "~/usr/local/gbp/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{GBP_ROOT}/lib"
        "/usr/lib"
        "/usr/gbp/lib"
        "/usr/local/lib"
        "/usr/local/gbp/lib"
        "~/usr/lib"
        "~/usr/gbp/lib"
        "~/usr/local/lib"
        "~/usr/local/gbp/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "Unsupported operating system.")

endif()

#
# Detect include paths based on the above search paths and hints.
#

find_path(GBP_INCLUDE_DIR
            NAMES
            "gbpCommon.h"
            HINTS
            ${INCLUDE_SEARCH_HINTS}
            PATHS
            ${INCLUDE_SEARCH_PATHS}
            DOC "Absolute path to gbpCode include directory.")

#
# Set library names.
#

set(LIBRARY_LIB_NAME_LIST
    "gbpLib"
    "gbpCosmo"
    "gbpHalos"
    "gbpMath"
    "gbpSPH"
    "gbpUtils"
)

#
# Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "GBP_LIBRARY"
    "GBP_COSMO_LIBRARY"
    "GBP_HALOS_LIBRARY"
    "GBP_MATH_LIBRARY"
    "GBP_SPH_LIBRARY"
    "GBP_UTILS_LIBRARY"
)

#
# Detect the paths of the above libraries and save them in the appropriate
# variable.
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
                    DOC "Absolute path to ${LIB_NAME} library.")
    unset(LIB_NAME_VAR)
    unset(LIB_NAME)
endforeach(i)

#
# Combine all library paths into one variable.
#

set(GBP_INCLUDE_DIRS
    ${GBP_INCLUDE_DIR}
)

set(GBP_LIBRARIES
    ${GBP_LIBRARY}
    ${GBP_COSMO_LIBRARY}
    ${GBP_HALOS_LIBRARY}
    ${GBP_MATH_LIBRARY}
    ${GBP_SPH_LIBRARY}
    ${GBP_UTILS_LIBRARY}
)

#
# Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GBP DEFAULT_MSG
                                  GBP_LIBRARIES GBP_INCLUDE_DIRS)

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
