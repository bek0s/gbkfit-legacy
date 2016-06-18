
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_cuda-project)

# ==============================================================================

find_package(CUDA REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/cuda/ndarray.hpp
)

set(SOURCE_FILES
    src/ndarray.cpp
)

# ==============================================================================

set(GBKFIT_CUDA_LIBRARY_STATIC $<TARGET_FILE:gbkfit_cuda_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_CUDA_LIBRARY_SHARED $<TARGET_FILE:gbkfit_cuda_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_CUDA_INCLUDE_DIR $<TARGET_PROPERTY:gbkfit_cuda_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(gbkfit_cuda_object
    OBJECT
    ${HEADER_FILES}
    ${SOURCE_FILES}
)

add_library(gbkfit_cuda_static
    STATIC
    $<TARGET_OBJECTS:gbkfit_cuda_object>
)

add_library(gbkfit_cuda_shared
    SHARED
    $<TARGET_OBJECTS:gbkfit_cuda_object>
)

# ==============================================================================

target_include_directories(gbkfit_cuda_object
    PUBLIC
    "include"
    PRIVATE
    ${GBKFIT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(gbkfit_cuda_static
    PRIVATE
    ${GBKFIT_LIBRARY_STATIC}
    ${CUDA_LIBRARIES}
)

target_link_libraries(gbkfit_cuda_shared
    PRIVATE
    ${GBKFIT_LIBRARY_SHARED}
    ${CUDA_LIBRARIES}
)

target_compile_definitions(gbkfit_cuda_object
    PRIVATE
    "GBKFIT_CUDA_SELF_BUILD"
)

target_compile_options(gbkfit_cuda_object
    PRIVATE
    "-Wall"
    "-Wextra"
    "-Wpedantic"
    "-Wsign-conversion"
)

set_target_properties(gbkfit_cuda_object
    PROPERTIES
    CXX_EXTENSIONS              OFF
    CXX_STANDARD                14
    CXX_STANDARD_REQUIRED       ON
    POSITION_INDEPENDENT_CODE   ON
)

set_target_properties(gbkfit_cuda_static
    PROPERTIES
    OUTPUT_NAME     "gbkfit_cuda_static"
    DEBUG_POSTFIX   "_d"
)

set_target_properties(gbkfit_cuda_shared
    PROPERTIES
    OUTPUT_NAME     "gbkfit_cuda_shared"
    DEBUG_POSTFIX   "_d"
)

# ==============================================================================

install(TARGETS
    gbkfit_cuda_static
    gbkfit_cuda_shared
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
