
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_fitter_mpfit-project)

# ==============================================================================

#find_package(mpfit)

# ==============================================================================

set(HEADER_FILES
    ${gbkfit-workspace_SOURCE_DIR}/deps/mpfit/mpfit.h
    include/gbkfit/fitter/mpfit/fitter_mpfit.hpp
    include/gbkfit/fitter/mpfit/fitter_mpfit_factory.hpp
)

set(SOURCE_FILES
    ${gbkfit-workspace_SOURCE_DIR}/deps/mpfit/mpfit.c
    src/fitter_mpfit.cpp
    src/fitter_mpfit_factory.cpp
)

# ==============================================================================

set(GBKFIT_FITTER_MPFIT_LIBRARY_STATIC $<TARGET_FILE:gbkfit_fitter_mpfit_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MPFIT_LIBRARY_SHARED $<TARGET_FILE:gbkfit_fitter_mpfit_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MPFIT_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_fitter_mpfit_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_fitter_mpfit_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_fitter_mpfit_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_fitter_mpfit_object>
)

add_library(gbkfit_fitter_mpfit_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_fitter_mpfit_object>
)

# ==============================================================================

target_include_directories(gbkfit_fitter_mpfit_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
    ${gbkfit-workspace_SOURCE_DIR}/deps/mpfit
#   ${MPFIT_INCLUDE_DIRS}
)

target_link_libraries(gbkfit_fitter_mpfit_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
#   ${MPFIT_LIBRARIES}
)

target_link_libraries(gbkfit_fitter_mpfit_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
#   ${MPFIT_LIBRARIES}
)

target_compile_definitions(gbkfit_fitter_mpfit_object
    PRIVATE
    "GBKFIT_FITTER_MPFIT_SELF_BUILD"
)

target_compile_options(gbkfit_fitter_mpfit_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_fitter_mpfit_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_fitter_mpfit_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_fitter_mpfit_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_fitter_mpfit_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_fitter_mpfit_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_fitter_mpfit_static
    gbkfit_fitter_mpfit_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
