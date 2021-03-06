
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_dmodel_cuda-project)

# ==============================================================================

find_package(CUDA REQUIRED)

# ==============================================================================

set(HEADER_FILES_NVCC
    include/gbkfit/dmodel/mmaps/mmaps_cuda_kernels_d.cuh
    include/gbkfit/dmodel/mmaps/mmaps_cuda_kernels_h.hpp
    include/gbkfit/dmodel/scube/scube_cuda_kernels_d.cuh
    include/gbkfit/dmodel/scube/scube_cuda_kernels_h.hpp
)

set(SOURCE_FILES_NVCC
    src/mmaps_cuda_kernels_h.cu
    src/scube_cuda_kernels_h.cu
)

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -keep -arch=compute_30 -code=sm_30,sm_35 -Xcompiler -fPIC")

include_directories(include)

cuda_add_library(gbkfit_dmodel_cuda_nvcc_static
    STATIC
    ${HEADER_FILES_NVCC}
    ${SOURCE_FILES_NVCC}
)

cuda_add_library(gbkfit_dmodel_cuda_nvcc_shared
    SHARED
    ${HEADER_FILES_NVCC}
    ${SOURCE_FILES_NVCC}
)

set_target_properties(gbkfit_dmodel_cuda_nvcc_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_cuda_nvcc_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_dmodel_cuda_nvcc_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_cuda_nvcc_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/dmodel/mmaps/mmaps_cuda.hpp
    include/gbkfit/dmodel/mmaps/mmaps_cuda_factory.hpp
    include/gbkfit/dmodel/scube/scube_cuda.hpp
    include/gbkfit/dmodel/scube/scube_cuda_factory.hpp
)

set(SOURCE_FILES
    src/mmaps_cuda.cpp
    src/mmaps_cuda_factory.cpp
    src/scube_cuda.cpp
    src/scube_cuda_factory.cpp
)

# ==============================================================================

set(GBKFIT_DMODEL_CUDA_LIBRARY_STATIC $<TARGET_FILE:gbkfit_dmodel_cuda_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_CUDA_LIBRARY_SHARED $<TARGET_FILE:gbkfit_dmodel_cuda_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_CUDA_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_dmodel_cuda_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_dmodel_cuda_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_dmodel_cuda_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_dmodel_cuda_object>
)

add_library(gbkfit_dmodel_cuda_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_dmodel_cuda_object>
)

# ==============================================================================

target_include_directories(gbkfit_dmodel_cuda_object
    PUBLIC
    "include"
    ${GBKFIT_CUDA_INCLUDE_DIR}
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
    ${GBKFIT_DMODEL_INCLUDE_DIR}
)

target_link_libraries(gbkfit_dmodel_cuda_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
    ${GBKFIT_CUDA_LIBRARY_STATIC}
    ${GBKFIT_DMODEL_LIBRARY_STATIC}
    ${CUDA_CUFFT_LIBRARIES}
    gbkfit_dmodel_cuda_nvcc_static
)

target_link_libraries(gbkfit_dmodel_cuda_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${GBKFIT_CUDA_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_LIBRARY_SHARED}
    ${CUDA_CUFFT_LIBRARIES}
    gbkfit_dmodel_cuda_nvcc_shared
)

target_compile_definitions(gbkfit_dmodel_cuda_object
    PRIVATE
    "GBKFIT_DMODEL_CUDA_SELF_BUILD"
)

target_compile_options(gbkfit_dmodel_cuda_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_dmodel_cuda_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                11
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_dmodel_cuda_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_cuda_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_dmodel_cuda_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_cuda_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_dmodel_cuda_static
    gbkfit_dmodel_cuda_shared
    gbkfit_dmodel_cuda_nvcc_static
    gbkfit_dmodel_cuda_nvcc_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
