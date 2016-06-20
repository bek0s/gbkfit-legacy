
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_dmodel_scube_omp-project)

# ==============================================================================

find_package(OpenMP)
find_package(fftw3 COMPONENTS SINGLE)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/dmodel/scube/scube_omp.hpp
    include/gbkfit/dmodel/scube/scube_omp_factory.hpp
    include/gbkfit/dmodel/scube/scube_omp_kernels.hpp
)

set(SOURCE_FILES
    src/scube_omp.cpp
    src/scube_omp_factory.cpp
    src/scube_omp_kernels.cpp
)

# ==============================================================================

set(GBKFIT_DMODEL_SCUBE_OMP_LIBRARY_STATIC $<TARGET_FILE:gbkfit_dmodel_scube_omp_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_SCUBE_OMP_LIBRARY_SHARED $<TARGET_FILE:gbkfit_dmodel_scube_omp_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_SCUBE_OMP_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_dmodel_scube_omp_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_dmodel_scube_omp_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_dmodel_scube_omp_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_dmodel_scube_omp_object>
)

add_library(gbkfit_dmodel_scube_omp_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_dmodel_scube_omp_object>
)

# ==============================================================================

target_include_directories(gbkfit_dmodel_scube_omp_object
    PUBLIC
    "include"
    ${FFTW3_INCLUDE_DIRS}
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
    ${GBKFIT_DMODEL_SCUBE_INCLUDE_DIR}
)

target_link_libraries(gbkfit_dmodel_scube_omp_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
    ${GBKFIT_DMODEL_SCUBE_LIBRARY_STATIC}
    ${FFTW3_SINGLE_THREADS_LIBRARY}
    ${OpenMP_CXX_FLAGS}
)

target_link_libraries(gbkfit_dmodel_scube_omp_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_SCUBE_LIBRARY_SHARED}
    ${FFTW3_SINGLE_THREADS_LIBRARY}
    ${OpenMP_CXX_FLAGS}
)

target_compile_definitions(gbkfit_dmodel_scube_omp_object
    PRIVATE
    "GBKFIT_DMODEL_SCUBE_OMP_SELF_BUILD"
)

target_compile_options(gbkfit_dmodel_scube_omp_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
    ${OpenMP_CXX_FLAGS}
)

set_target_properties(gbkfit_dmodel_scube_omp_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_dmodel_scube_omp_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_scube_omp_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_dmodel_scube_omp_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_scube_omp_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_dmodel_scube_omp_static
    gbkfit_dmodel_scube_omp_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
