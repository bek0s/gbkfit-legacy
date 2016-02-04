
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_ocl-project)

# ==============================================================================

find_package(OpenCL REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/ocl/ndarray_ocl.hpp
)

set(SOURCE_FILES
    src/ndarray_ocl.cpp
)

# ==============================================================================

set(GBKFIT_OCL_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_ocl_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_OCL_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_ocl_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_OCL_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_ocl_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_OCL_INCLUDE_DIRS ${GBKFIT_OCL_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_ocl_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_ocl_static STATIC $<TARGET_OBJECTS:target_gbkfit_ocl_object>)
add_library(target_gbkfit_ocl_shared SHARED $<TARGET_OBJECTS:target_gbkfit_ocl_object>)

target_include_directories(target_gbkfit_ocl_object PUBLIC "include")
target_include_directories(target_gbkfit_ocl_object PRIVATE ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_ocl_object PRIVATE ${OCL_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_ocl_static PRIVATE ${OCL_LIBRARIES})
target_link_libraries(target_gbkfit_ocl_shared PRIVATE ${OCL_LIBRARIES})

target_compile_definitions(target_gbkfit_ocl_object PRIVATE "GBKFIT_OCL_SELF_BUILD")

target_compile_options(target_gbkfit_ocl_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_ocl_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_ocl_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_ocl_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_ocl_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_ocl_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_ocl_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_ocl_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_ocl_static PROPERTIES OUTPUT_NAME "gbkfit_ocl_static")
set_target_properties(target_gbkfit_ocl_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_ocl_shared PROPERTIES OUTPUT_NAME "gbkfit_ocl_shared")
set_target_properties(target_gbkfit_ocl_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_gbkfit_ocl_static
        target_gbkfit_ocl_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
