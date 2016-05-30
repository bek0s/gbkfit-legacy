
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_dmodel_scube_cuda-project)

# ==============================================================================

find_package(CUDA REQUIRED)

# ==============================================================================

set(HEADER_FILES_CUDA
    include/gbkfit/dmodel/scube/scube_cuda_kernels_d.cuh
    include/gbkfit/dmodel/scube/scube_cuda_kernels_h.hpp
)

set(SOURCE_FILES_CUDA
    src/scube_cuda_kernels_h.cu
)

SET(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -keep -arch=compute_20 -code=sm_20,sm_30,sm_35 -Xcompiler -fPIC")

include_directories(include)

cuda_add_library(target_gbkfit_dmodel_scube_cuda_nvcc ${HEADER_FILES_CUDA} ${SOURCE_FILES_CUDA})
set_target_properties(target_gbkfit_dmodel_scube_cuda_nvcc PROPERTIES OUTPUT_NAME "gbkfit_dmodel_scube_cuda_nvcc")
set_target_properties(target_gbkfit_dmodel_scube_cuda_nvcc PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/dmodel/scube/scube_cuda.hpp
    include/gbkfit/dmodel/scube/scube_cuda_factory.hpp
)

set(SOURCE_FILES
    src/scube_cuda.cpp
    src/scube_cuda_factory.cpp
)

# ==============================================================================

set(GBKFIT_DMODEL_SCUBE_CUDA_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_dmodel_scube_cuda_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_SCUBE_CUDA_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_dmodel_scube_cuda_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_DMODEL_SCUBE_CUDA_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_dmodel_scube_cuda_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_dmodel_scube_cuda_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_dmodel_scube_cuda_static STATIC $<TARGET_OBJECTS:target_gbkfit_dmodel_scube_cuda_object>)
add_library(target_gbkfit_dmodel_scube_cuda_shared SHARED $<TARGET_OBJECTS:target_gbkfit_dmodel_scube_cuda_object>)

# ==============================================================================

target_include_directories(target_gbkfit_dmodel_scube_cuda_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
    ${GBKFIT_CUDA_INCLUDE_DIR}
    ${GBKFIT_DMODEL_SCUBE_INCLUDE_DIR}
)

target_link_libraries(target_gbkfit_dmodel_scube_cuda_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
    ${GBKFIT_CUDA_LIBRARY_STATIC}
    ${GBKFIT_DMODEL_SCUBE_LIBRARY_STATIC}
    target_gbkfit_dmodel_scube_cuda_nvcc
    cufft
)

target_link_libraries(target_gbkfit_dmodel_scube_cuda_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${GBKFIT_CUDA_LIBRARY_SHARED}
    ${GBKFIT_DMODEL_SCUBE_LIBRARY_SHARED}
    target_gbkfit_dmodel_scube_cuda_nvcc
    cufft
)

target_compile_definitions(target_gbkfit_dmodel_scube_cuda_object
    PRIVATE
    "GBKFIT_DMODEL_SCUBE_CUDA_SELF_BUILD"
)

target_compile_options(target_gbkfit_dmodel_scube_cuda_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(target_gbkfit_dmodel_scube_cuda_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(target_gbkfit_dmodel_scube_cuda_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_scube_cuda_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(target_gbkfit_dmodel_scube_cuda_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_dmodel_scube_cuda_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
        target_gbkfit_dmodel_scube_cuda_static
        target_gbkfit_dmodel_scube_cuda_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
