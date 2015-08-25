
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find cfitsio libraries and include directories.
#
#   http://heasarc.gsfc.nasa.gov/fitsio/fitsio.html
#
#   Variables defined after execution:
#
#	CFITSIO_FOUND
#	CFITSIO_INCLUDE_DIR
#	CFITSIO_INCLUDE_DIRS
#       CFITSIO_LIBRARY
#	CFITSIO_LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{CFITSIO_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{CFITSIO_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{CFITSIO_ROOT}/include"
	"/usr/include"
	"/usr/cfitsio/include"
	"/usr/local/include"
	"/usr/local/cfitsio/include"
	"~/usr/include"
	"~/usr/cfitsio/include"
	"~/usr/local/include"
	"~/usr/local/cfitsio/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{CFITSIO_ROOT}/lib"
	"/usr/lib"
	"/usr/cfitsio/lib"
	"/usr/local/lib"
	"/usr/local/cfitsio/lib"
	"~/usr/lib"
	"~/usr/cfitsio/lib"
	"~/usr/local/lib"
	"~/usr/local/cfitsio/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "Unsupported operating system.")

endif()

#
# Detect include paths based on the above search paths and hints.
#

find_path(CFITSIO_INCLUDE_DIR
	    NAMES
            "fitsio.h"
	    HINTS
	    ${INCLUDE_SEARCH_HINTS}
	    PATHS
	    ${INCLUDE_SEARCH_PATHS}
            DOC "Absolute path to cfitsio include directory.")


#
# Set library names.
#
set(LIBRARY_LIB_NAME_LIST
    "cfitsio"
)

#
# Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "CFITSIO_LIBRARY"
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

set(CFITSIO_INCLUDE_DIRS
    ${CFITSIO_INCLUDE_DIR}
)

set(CFITSIO_LIBRARIES
    ${CFITSIO_LIBRARY}
)

#
# Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CFITSIO DEFAULT_MSG
				  CFITSIO_LIBRARIES CFITSIO_INCLUDE_DIRS)

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
