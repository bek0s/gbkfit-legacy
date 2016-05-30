
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit-project)

# ==============================================================================

find_package(cfitsio)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/array_util.hpp
    include/gbkfit/build_settings.hpp
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
    include/gbkfit/model.hpp
    include/gbkfit/ndarray.hpp
    include/gbkfit/ndarray_host.hpp
    include/gbkfit/ndshape.hpp
    include/gbkfit/parameters.hpp
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
    src/model.cpp
    src/ndarray.cpp
    src/ndarray_host.cpp
    src/ndshape.cpp
    src/parameters.cpp
    src/spread_functions.cpp
    src/utility.cpp
)

# ==============================================================================

set(GBKFIT_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_static STATIC $<TARGET_OBJECTS:target_gbkfit_object>)
add_library(target_gbkfit_shared SHARED $<TARGET_OBJECTS:target_gbkfit_object>)

target_include_directories(target_gbkfit_object PUBLIC "include")
target_include_directories(target_gbkfit_object PRIVATE ${Boost_INCLUDE_DIRS})
target_include_directories(target_gbkfit_object PRIVATE ${CFITSIO_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_static PRIVATE ${CFITSIO_LIBRARIES} jsoncpp)
target_link_libraries(target_gbkfit_shared PRIVATE ${CFITSIO_LIBRARIES} jsoncpp)

target_compile_definitions(target_gbkfit_object PRIVATE "GBKFIT_SELF_BUILD")

target_compile_options(target_gbkfit_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_object PROPERTIES CXX_STANDARD 14)
set_target_properties(target_gbkfit_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_static PROPERTIES OUTPUT_NAME "gbkfit_static")
set_target_properties(target_gbkfit_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_shared PROPERTIES OUTPUT_NAME "gbkfit_shared")
set_target_properties(target_gbkfit_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_gbkfit_static
        target_gbkfit_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
