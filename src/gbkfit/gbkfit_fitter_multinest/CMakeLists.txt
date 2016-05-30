
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_fitter_multinest-project)

# ==============================================================================

find_package(multinest REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/fitter/multinest/fitter_multinest.hpp
    include/gbkfit/fitter/multinest/fitter_multinest_factory.hpp
)

set(SOURCE_FILES
    src/fitter_multinest.cpp
    src/fitter_multinest_factory.cpp
)

# ==============================================================================

set(GBKFIT_FITTER_MULTINEST_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_fitter_multinest_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MULTINEST_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_fitter_multinest_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MULTINEST_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_fitter_multinest_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_fitter_multinest_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_fitter_multinest_static STATIC $<TARGET_OBJECTS:target_gbkfit_fitter_multinest_object>)
add_library(target_gbkfit_fitter_multinest_shared SHARED $<TARGET_OBJECTS:target_gbkfit_fitter_multinest_object>)

# ==============================================================================

target_include_directories(target_gbkfit_fitter_multinest_object
    PUBLIC
    "include"
    ${GBKFIT_INCLUDE_DIR}
    ${MULTINEST_INCLUDE_DIRS}
)

target_link_libraries(target_gbkfit_fitter_multinest_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
    ${MULTINEST_LIBRARIES}
)

target_link_libraries(target_gbkfit_fitter_multinest_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${MULTINEST_LIBRARIES}
)

target_compile_definitions(target_gbkfit_fitter_multinest_object
    PRIVATE
    "GBKFIT_FITTER_MULTINEST_SELF_BUILD"
)

target_compile_options(target_gbkfit_fitter_multinest_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(target_gbkfit_fitter_multinest_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(target_gbkfit_fitter_multinest_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_fitter_multinest_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(target_gbkfit_fitter_multinest_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_fitter_multinest_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(
    TARGETS
    target_gbkfit_fitter_multinest_static
    target_gbkfit_fitter_multinest_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
