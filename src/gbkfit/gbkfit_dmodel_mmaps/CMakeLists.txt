
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_dmodel_mmaps-project)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/dmodel/mmaps/mmaps.hpp
    include/gbkfit/dmodel/mmaps/mmaps_factory.hpp
)

set(SOURCE_FILES
    src/mmaps.cpp
    src/mmaps_factory.cpp
)

# ==============================================================================

set(GBKFIT_DMODEL_MMAPS_LIBRARY_STATIC $<TARGET_FILE:gbkfit_dmodel_mmaps_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_MMAPS_LIBRARY_SHARED $<TARGET_FILE:gbkfit_dmodel_mmaps_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_MMAPS_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_dmodel_mmaps_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_dmodel_mmaps_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_dmodel_mmaps_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_dmodel_mmaps_object>
)

add_library(gbkfit_dmodel_mmaps_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_dmodel_mmaps_object>
)

# ==============================================================================

target_include_directories(gbkfit_dmodel_mmaps_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
)

target_link_libraries(gbkfit_dmodel_mmaps_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
)

target_link_libraries(gbkfit_dmodel_mmaps_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
)

target_compile_definitions(gbkfit_dmodel_mmaps_object
    PRIVATE
    "GBKFIT_DMODEL_CUBE_SELF_BUILD"
)

target_compile_options(gbkfit_dmodel_mmaps_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_dmodel_mmaps_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_dmodel_mmaps_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_mmaps_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_dmodel_mmaps_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_mmaps_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_dmodel_mmaps_static
    gbkfit_dmodel_mmaps_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
