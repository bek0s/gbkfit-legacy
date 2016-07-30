
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_gmodel_omp-project)

# ==============================================================================

find_package(OpenMP)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/gmodel/gmodel1/gmodel1_omp.hpp
    include/gbkfit/gmodel/gmodel1/gmodel1_omp_factory.hpp
    include/gbkfit/gmodel/gmodel1/gmodel1_omp_kernels.hpp
)

set(SOURCE_FILES
    src/gmodel1_omp.cpp
    src/gmodel1_omp_factory.cpp
    src/gmodel1_omp_kernels.cpp
)

# ==============================================================================

set(GBKFIT_GMODEL_OMP_LIBRARY_STATIC $<TARGET_FILE:gbkfit_gmodel_omp_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_OMP_LIBRARY_SHARED $<TARGET_FILE:gbkfit_gmodel_omp_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_GMODEL_OMP_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_gmodel_omp_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_gmodel_omp_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_gmodel_omp_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_gmodel_omp_object>
)

add_library(gbkfit_gmodel_omp_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_gmodel_omp_object>
)

# ==============================================================================

target_include_directories(gbkfit_gmodel_omp_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
    ${GBKFIT_GMODEL_INCLUDE_DIR}
)

target_link_libraries(gbkfit_gmodel_omp_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
    ${GBKFIT_GMODEL_LIBRARY_STATIC}
    ${OpenMP_CXX_FLAGS}
)

target_link_libraries(gbkfit_gmodel_omp_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${GBKFIT_GMODEL_LIBRARY_SHARED}
    ${OpenMP_CXX_FLAGS}
)

target_compile_definitions(gbkfit_gmodel_omp_object
    PRIVATE
    "GBKFIT_GMODEL_OMP_SELF_BUILD"
)

target_compile_options(gbkfit_gmodel_omp_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
    ${OpenMP_CXX_FLAGS}
)

set_target_properties(gbkfit_gmodel_omp_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_gmodel_omp_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_omp_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_gmodel_omp_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_gmodel_omp_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_gmodel_omp_static
    gbkfit_gmodel_omp_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
