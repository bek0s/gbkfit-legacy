
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_gmodel_gmodel01_omp-project)

# ==============================================================================

find_package(OpenMP)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/gmodel/gmodel01/gmodel01_omp.hpp
    include/gbkfit/gmodel/gmodel01/gmodel01_omp_factory.hpp
    include/gbkfit/gmodel/gmodel01/gmodel01_omp_kernels.hpp
)

set(SOURCE_FILES
    src/gmodel01_omp.cpp
    src/gmodel01_omp_factory.cpp
    src/gmodel01_omp_kernels.cpp
)

# ==============================================================================

set(GBKFIT_GMODEL_GMODEL01_OMP_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_gmodel_gmodel01_omp_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_GMODEL01_OMP_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_gmodel_gmodel01_omp_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_GMODEL01_OMP_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_gmodel_gmodel01_omp_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_gmodel_gmodel01_omp_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_gmodel_gmodel01_omp_static STATIC $<TARGET_OBJECTS:target_gbkfit_gmodel_gmodel01_omp_object>)
add_library(target_gbkfit_gmodel_gmodel01_omp_shared SHARED $<TARGET_OBJECTS:target_gbkfit_gmodel_gmodel01_omp_object>)

# ==============================================================================

target_include_directories(target_gbkfit_gmodel_gmodel01_omp_object
    PUBLIC
    "include"
    ${GBKFIT_GMODEL_GMODEL01_INCLUDE_DIR}
)

target_link_libraries(target_gbkfit_gmodel_gmodel01_omp_static
    PRIVATE
    ${GBKFIT_GMODEL_GMODEL01_LIBRARY_STATIC}
    gomp
)

target_link_libraries(target_gbkfit_gmodel_gmodel01_omp_shared
    PRIVATE
    ${GBKFIT_GMODEL_GMODEL01_LIBRARY_SHARED}
    gomp
)

target_compile_definitions(target_gbkfit_gmodel_gmodel01_omp_object
    PRIVATE
    "GBKFIT_GMODEL_GMODEL01_OMP_SELF_BUILD"
)

target_compile_options(target_gbkfit_gmodel_gmodel01_omp_object
    PRIVATE
    ${OpenMP_CXX_FLAGS}
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(target_gbkfit_gmodel_gmodel01_omp_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(target_gbkfit_gmodel_gmodel01_omp_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_gmodel01_omp_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(target_gbkfit_gmodel_gmodel01_omp_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_gmodel01_omp_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(
    TARGETS
    target_gbkfit_gmodel_gmodel01_omp_static
    target_gbkfit_gmodel_gmodel01_omp_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
