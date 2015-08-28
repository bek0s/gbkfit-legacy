
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_fitters_multinest-project)

# ==============================================================================

find_package(multinest REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/fitters/multinest/fitter_multinest.hpp
)

set(SOURCE_FILES
    src/fitter_multinest.cpp
)

# ==============================================================================

set(GBKFIT_FITTER_MULTINEST_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_fitter_multinest_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MULTINEST_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_fitter_multinest_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MULTINEST_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_fitter_multinest_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MULTINEST_INCLUDE_DIRS ${GBKFIT_FITTER_MULTINEST_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_fitter_multinest_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_fitter_multinest_static STATIC $<TARGET_OBJECTS:target_gbkfit_fitter_multinest_object>)
add_library(target_gbkfit_fitter_multinest_shared SHARED $<TARGET_OBJECTS:target_gbkfit_fitter_multinest_object>)

target_include_directories(target_gbkfit_fitter_multinest_object PUBLIC "./include")
target_include_directories(target_gbkfit_fitter_multinest_object PUBLIC ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_fitter_multinest_object PRIVATE ${MULTINEST_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_fitter_multinest_static PRIVATE ${MULTINEST_LIBRARIES})
target_link_libraries(target_gbkfit_fitter_multinest_shared PRIVATE ${MULTINEST_LIBRARIES})

target_compile_definitions(target_gbkfit_fitter_multinest_object PRIVATE "GBKFIT_FITTER_MULTINEST_SELF_BUILD")

target_compile_options(target_gbkfit_fitter_multinest_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_fitter_multinest_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_fitter_multinest_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_fitter_multinest_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_fitter_multinest_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_fitter_multinest_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_fitter_multinest_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_fitter_multinest_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_fitter_multinest_static PROPERTIES OUTPUT_NAME "gbkfit_fitter_multinest_static")
set_target_properties(target_gbkfit_fitter_multinest_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_fitter_multinest_shared PROPERTIES OUTPUT_NAME "gbkfit_fitter_multinest_shared")
set_target_properties(target_gbkfit_fitter_multinest_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS target_gbkfit_fitter_multinest_static target_gbkfit_fitter_multinest_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
