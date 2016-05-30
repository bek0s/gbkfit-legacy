
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_gmodel_gmodel01-project)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/gmodel/gmodel01/gmodel01.hpp
    include/gbkfit/gmodel/gmodel01/gmodel01_factory.hpp
)

set(SOURCE_FILES
    src/gmodel01.cpp
    src/gmodel01_factory.cpp
)

# ==============================================================================

set(GBKFIT_GMODEL_GMODEL01_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_gmodel_gmodel01_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_GMODEL01_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_gmodel_gmodel01_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_GMODEL01_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_gmodel_gmodel01_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_gmodel_gmodel01_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_gmodel_gmodel01_static STATIC $<TARGET_OBJECTS:target_gbkfit_gmodel_gmodel01_object>)
add_library(target_gbkfit_gmodel_gmodel01_shared SHARED $<TARGET_OBJECTS:target_gbkfit_gmodel_gmodel01_object>)

# ==============================================================================

target_include_directories(target_gbkfit_gmodel_gmodel01_object
    PUBLIC
    "include"
    ${GBKFIT_INCLUDE_DIR}
)

target_link_libraries(target_gbkfit_gmodel_gmodel01_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
)

target_link_libraries(target_gbkfit_gmodel_gmodel01_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
)

target_compile_definitions(target_gbkfit_gmodel_gmodel01_object
    PRIVATE
    "GBKFIT_GMODEL_GMODEL01_SELF_BUILD"
)

target_compile_options(target_gbkfit_gmodel_gmodel01_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(target_gbkfit_gmodel_gmodel01_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(target_gbkfit_gmodel_gmodel01_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_gmodel01_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(target_gbkfit_gmodel_gmodel01_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_gmodel01_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(
    TARGETS
    target_gbkfit_gmodel_gmodel01_static
    target_gbkfit_gmodel_gmodel01_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
