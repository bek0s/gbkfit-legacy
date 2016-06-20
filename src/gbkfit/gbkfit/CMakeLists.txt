
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit-project)

# ==============================================================================

set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED)
find_package(cfitsio REQUIRED)

# ==============================================================================

set(HEADER_FILES
    ${gbkfit-workspace_BINARY_DIR}/include/gbkfit/build_settings.hpp
    include/gbkfit/array_util.hpp
    include/gbkfit/config.hpp
    include/gbkfit/core.hpp
    include/gbkfit/dataset.hpp
    include/gbkfit/dmodel.hpp
    include/gbkfit/fits.hpp
    include/gbkfit/fitter.hpp
    include/gbkfit/fitter_result.hpp
    include/gbkfit/gmodel.hpp
    include/gbkfit/instrument.hpp
    include/gbkfit/json.hpp
    include/gbkfit/math.hpp
    include/gbkfit/ndarray.hpp
    include/gbkfit/ndarray_host.hpp
    include/gbkfit/ndshape.hpp
    include/gbkfit/params.hpp
    include/gbkfit/prerequisites.hpp
    include/gbkfit/spread_functions.hpp
    include/gbkfit/string_util.hpp
    include/gbkfit/utility.hpp
    include/gbkfit/variable_map.hpp
)

set(SOURCE_FILES
    src/array_util.cpp
    src/core.cpp
    src/dataset.cpp
    src/dmodel.cpp
    src/fits.cpp
    src/fitter.cpp
    src/fitter_result.cpp
    src/gmodel.cpp
    src/instrument.cpp
    src/ndarray.cpp
    src/ndarray_host.cpp
    src/ndshape.cpp
    src/params.cpp
    src/spread_functions.cpp
    src/utility.cpp
)

# ==============================================================================

set(GBKFIT_LIBRARY_STATIC $<TARGET_FILE:gbkfit_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_LIBRARY_SHARED $<TARGET_FILE:gbkfit_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_object>
)

add_library(gbkfit_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_object>
)

# ==============================================================================

target_include_directories(gbkfit_object
    PUBLIC
    "include"
    PRIVATE
    ${Boost_INCLUDE_DIRS}
    ${CFITSIO_INCLUDE_DIRS}
)

target_link_libraries(gbkfit_static
    PRIVATE
    ${CFITSIO_LIBRARIES}
)

target_link_libraries(gbkfit_shared
    PRIVATE
    ${CFITSIO_LIBRARIES}
)

target_compile_definitions(gbkfit_object
    PRIVATE
    "GBKFIT_SELF_BUILD"
)

target_compile_options(gbkfit_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_static
    gbkfit_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
