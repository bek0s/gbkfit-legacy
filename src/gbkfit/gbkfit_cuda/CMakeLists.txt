
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
    include/gbkfit/cuda/memory_buffer_cuda.hpp
    include/gbkfit/cuda/ndarray_cuda.hpp
)

set(SOURCE_FILES
    src/memory_buffer_cuda.cpp
    src/ndarray_cuda.cpp
)

# ==============================================================================

set(GBKFIT_CUDA_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_cuda_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_CUDA_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_cuda_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_CUDA_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_cuda_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_CUDA_INCLUDE_DIRS ${GBKFIT_CUDA_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_cuda_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_cuda_static STATIC $<TARGET_OBJECTS:target_gbkfit_cuda_object>)
add_library(target_gbkfit_cuda_shared SHARED $<TARGET_OBJECTS:target_gbkfit_cuda_object>)

target_include_directories(target_gbkfit_cuda_object PUBLIC "include")
target_include_directories(target_gbkfit_cuda_object PRIVATE ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_cuda_object PRIVATE ${CUDA_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_cuda_static PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(target_gbkfit_cuda_shared PRIVATE ${CUDA_LIBRARIES})

target_compile_definitions(target_gbkfit_cuda_object PRIVATE "GBKFIT_CUDA_SELF_BUILD")

target_compile_options(target_gbkfit_cuda_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_cuda_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_cuda_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_cuda_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_cuda_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_cuda_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_cuda_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_cuda_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_cuda_static PROPERTIES OUTPUT_NAME "gbkfit_cuda_static")
set_target_properties(target_gbkfit_cuda_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_cuda_shared PROPERTIES OUTPUT_NAME "gbkfit_cuda_shared")
set_target_properties(target_gbkfit_cuda_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_gbkfit_cuda_static
        target_gbkfit_cuda_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
