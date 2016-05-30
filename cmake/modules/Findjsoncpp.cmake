
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#
#   Try to find jsoncpp libraries and include directories.
#
#   https://github.com/open-source-parsers/jsoncpp
#
#   Variables defined after execution:
#
#	JSONCPP_FOUND
#	JSONCPP_INCLUDE_DIR
#	JSONCPP_INCLUDE_DIRS
#       JSONCPP_LIBRARY
#	JSONCPP_LIBRARIES
#
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

#
# Setup default search paths and hints to help CMake find the dependencies.
#

if(WIN32)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{JSONCPP_ROOT}/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{JSONCPP_ROOT}/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

elseif(UNIX)

    set(INCLUDE_SEARCH_PATHS
        "$ENV{JSONCPP_ROOT}/include"
        "/usr/include"
        "/usr/jsoncpp/include"
        "/usr/local/include"
        "/usr/local/jsoncpp/include"
        "~/usr/include"
        "~/usr/jsoncpp/include"
        "~/usr/local/include"
        "~/usr/local/jsoncpp/include"
    )
    set(INCLUDE_SEARCH_HINTS
    )
    set(LIBRARY_SEARCH_PATHS
        "$ENV{JSONCPP_ROOT}/lib"
        "/usr/lib"
        "/usr/jsoncpp/lib"
        "/usr/local/lib"
        "/usr/local/jsoncpp/lib"
        "~/usr/lib"
        "~/usr/jsoncpp/lib"
        "~/usr/local/lib"
        "~/usr/local/jsoncpp/lib"
    )
    set(LIBRARY_SEARCH_HINTS
    )

else()

    message(FATAL_ERROR "Unsupported operating system.")

endif()

#
# Detect include paths based on the above search paths and hints.
#

find_path(JSONCPP_INCLUDE_DIR
            NAMES
            jsoncpp/json/json.h json/json.h
            HINTS
            ${INCLUDE_SEARCH_HINTS}
            PATHS
            ${INCLUDE_SEARCH_PATHS}
            DOC "Absolute path to jsoncpp include directory.")

#
# Set library names.
#

set(LIBRARY_LIB_NAME_LIST
    "jsoncpp"
)

#
# Set a variable name for each library.
#

set(LIBRARY_LIB_VARIABLE_NAME_LIST
    "JSONCPP_LIBRARY"
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

set(JSONCPP_INCLUDE_DIRS
    ${JSONCPP_INCLUDE_DIR}
)

set(JSONCPP_LIBRARIES
    ${JSONCPP_LIBRARY}
)

#
# Deal with the find_module() args and some other stuff.
#

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(JSONCPP DEFAULT_MSG
                                  JSONCPP_LIBRARIES JSONCPP_INCLUDE_DIRS)

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
