
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find gsl libraries and include directories.
#
#   http://www.gnu.org/software/gsl/
#
#   Variables defined after execution:
#
#	GSL_FOUND
#	GSL_INCLUDE_DIR
#	GSL_INCLUDE_DIRS
#       GSL_LIBRARY
#       GSL_CBLAS_LIBRARY
#	GSL_LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{GSL_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{GSL_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{GSL_ROOT}/include"
	"/usr/include"
	"/usr/gsl/include"
	"/usr/local/include"
	"/usr/local/gsl/include"
	"~/usr/include"
	"~/usr/gsl/include"
	"~/usr/local/include"
	"~/usr/local/gsl/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{GSL_ROOT}/lib"
	"/usr/lib"
	"/usr/gsl/lib"
	"/usr/local/lib"
	"/usr/local/gsl/lib"
	"~/usr/lib"
	"~/usr/gsl/lib"
	"~/usr/local/lib"
	"~/usr/local/gsl/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "Unsupported operating system.")

endif()

#
# Detect include paths based on the above search paths and hints.
#

find_path(GSL_INCLUDE_DIR
            NAMES
            "gsl/gsl_math.h"
            HINTS
            ${INCLUDE_SEARCH_HINTS}
            PATHS
            ${INCLUDE_SEARCH_PATHS}
            DOC "Absolute path to GSL include directory.")

#
# Set library names.
#

set(LIBRARY_LIB_NAME_LIST
    "gsl"
    "gslcblas"
)

#
# Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "GSL_LIBRARY"
    "GSL_CBLAS_LIBRARY"
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

set(GSL_INCLUDE_DIRS
    ${GSL_INCLUDE_DIR}
)

set(GSL_LIBRARIES
    ${GSL_LIBRARY}
    ${GSL_CBLAS_LIBRARY}
)

#
# Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GSL DEFAULT_MSG
                                  GSL_LIBRARIES GSL_INCLUDE_DIRS)

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
