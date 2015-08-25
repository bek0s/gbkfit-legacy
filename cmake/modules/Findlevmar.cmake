
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find levmar libraries and include directories.
#
#   http://users.ics.forth.gr/~lourakis/levmar/
#
#   Variables defined after execution:
#
#	LEVMAR_FOUND
#	LEVMAR_INCLUDE_DIR
#	LEVMAR_INCLUDE_DIRS
#	LEVMAR_LIBRARY
#	LEVMAR_LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
#   Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{LEVMAR_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{LEVMAR_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{LEVMAR_ROOT}/include"
        "/usr/include"
        "/usr/levmar/include"
        "/usr/local/include"
        "/usr/local/levmar/include"
        "~/usr/include"
        "~/usr/levmar/include"
        "~/usr/local/include"
        "~/usr/local/levmar/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{LEVMAR_ROOT}/lib"
        "/usr/lib"
        "/usr/levmar/lib"
        "/usr/local/lib"
        "/usr/local/levmar/lib"
        "~/usr/lib"
        "~/usr/levmar/lib"
        "~/usr/local/lib"
        "~/usr/local/levmar/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "unsupported operating system")

endif()

#
#   Detect include paths based on the above search paths and hints.
#

find_path(LEVMAR_INCLUDE_DIR
            NAMES
            "levmar.h"
            HINTS
            ${INCLUDE_SEARCH_HINTS}
            PATHS
            ${INCLUDE_SEARCH_PATHS}
            DOC "absolute path to levmar include directory")

#
#   Set library names.
#

set(LIBRARY_LIB_NAME_LIST
    "levmar"
)

#
#   Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "LEVMAR_LIBRARY"
)

#
#   Detect the paths of the above libraries and save them in the appropriate 
#   variable.
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
                    DOC "absolute path to ${LIB_NAME} library")
    unset(LIB_VAR_NAME)
    unset(LIB_NAME)
endforeach(i)

#
#   Combine all library paths into one variable.
#

set(LEVMAR_INCLUDE_DIRS
    ${LEVMAR_INCLUDE_DIR}
)

set(LEVMAR_LIBRARIES
    ${LEVMAR_LIBRARY}
)

#
#   Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LEVMAR DEFAULT_MSG
                                  LEVMAR_LIBRARIES LEVMAR_INCLUDE_DIRS)

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
