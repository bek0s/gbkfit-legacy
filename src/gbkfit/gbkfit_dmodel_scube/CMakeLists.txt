
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_dmodel_scube-project)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/dmodel/scube/scube.hpp
    include/gbkfit/dmodel/scube/scube_factory.hpp
)

set(SOURCE_FILES
    src/scube.cpp
    src/scube_factory.cpp
)

# ==============================================================================

set(GBKFIT_DMODEL_SCUBE_LIBRARY_STATIC $<TARGET_FILE:gbkfit_dmodel_scube_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_SCUBE_LIBRARY_SHARED $<TARGET_FILE:gbkfit_dmodel_scube_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_SCUBE_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_dmodel_scube_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_dmodel_scube_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_dmodel_scube_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_dmodel_scube_object>
)

add_library(gbkfit_dmodel_scube_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_dmodel_scube_object>
)

# ==============================================================================

target_include_directories(gbkfit_dmodel_scube_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
)

target_link_libraries(gbkfit_dmodel_scube_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
)

target_link_libraries(gbkfit_dmodel_scube_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
)

target_compile_definitions(gbkfit_dmodel_scube_object
    PRIVATE
    "GBKFIT_DMODEL_SCUBE_SELF_BUILD"
)

target_compile_options(gbkfit_dmodel_scube_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_dmodel_scube_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_dmodel_scube_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_scube_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_dmodel_scube_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_scube_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_dmodel_scube_static
    gbkfit_dmodel_scube_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
