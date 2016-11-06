
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_gmodel-project)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/gmodel/gmodel1/gmodel1.hpp
    include/gbkfit/gmodel/gmodel1/gmodel1_factory.hpp
)

set(SOURCE_FILES
    src/gmodel1.cpp
    src/gmodel1_factory.cpp
)

# ==============================================================================

set(GBKFIT_GMODEL_LIBRARY_STATIC $<TARGET_FILE:gbkfit_gmodel_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_LIBRARY_SHARED $<TARGET_FILE:gbkfit_gmodel_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_gmodel_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_gmodel_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_gmodel_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_gmodel_object>
)

add_library(gbkfit_gmodel_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_gmodel_object>
)

# ==============================================================================

target_include_directories(gbkfit_gmodel_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
)

target_link_libraries(gbkfit_gmodel_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
)

target_link_libraries(gbkfit_gmodel_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
)

target_compile_definitions(gbkfit_gmodel_object
    PRIVATE
    "GBKFIT_GMODEL_SELF_BUILD"
)

target_compile_options(gbkfit_gmodel_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_gmodel_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                11
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_gmodel_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_gmodel_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_gmodel_static
    gbkfit_gmodel_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
