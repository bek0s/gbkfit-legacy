
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_fftw3-project)

# ==============================================================================

find_package(fftw3 REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/fftw3/memory_buffer_fftw3.hpp
)

set(SOURCE_FILES
    src/memory_buffer_fftw3.cpp
)

# ==============================================================================

set(GBKFIT_FFTW3_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_fftw3_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_FFTW3_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_fftw3_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_FFTW3_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_fftw3_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_FFTW3_INCLUDE_DIRS ${GBKFIT_FFTW3_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_fftw3_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_fftw3_static STATIC $<TARGET_OBJECTS:target_gbkfit_fftw3_object>)
add_library(target_gbkfit_fftw3_shared SHARED $<TARGET_OBJECTS:target_gbkfit_fftw3_object>)

target_include_directories(target_gbkfit_fftw3_object PUBLIC "include")
target_include_directories(target_gbkfit_fftw3_object PRIVATE ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_fftw3_object PRIVATE ${FFTW3_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_fftw3_static PRIVATE ${FFTW3_LIBRARIES})
target_link_libraries(target_gbkfit_fftw3_shared PRIVATE ${FFTW3_LIBRARIES})

target_compile_definitions(target_gbkfit_fftw3_object PRIVATE "GBKFIT_FFTW3_SELF_BUILD")

target_compile_options(target_gbkfit_fftw3_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_fftw3_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_fftw3_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_fftw3_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_fftw3_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_fftw3_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_fftw3_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_fftw3_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_fftw3_static PROPERTIES OUTPUT_NAME "gbkfit_fftw3_static")
set_target_properties(target_gbkfit_fftw3_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_fftw3_shared PROPERTIES OUTPUT_NAME "gbkfit_fftw3_shared")
set_target_properties(target_gbkfit_fftw3_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_gbkfit_fftw3_static
        target_gbkfit_fftw3_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
