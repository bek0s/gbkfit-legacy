
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_fitter_mpfit-project)

# ==============================================================================

#find_package()

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/fitters/mpfit/fitter_mpfit.hpp

    include/gbkfit/fitters/mpfit/mpfit/mpfit.h
)

set(SOURCE_FILES
    src/fitter_mpfit.cpp

    src/mpfit/mpfit.c
)

# ==============================================================================

set(GBKFIT_FITTER_MPFIT_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_fitter_mpfit_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MPFIT_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_fitter_mpfit_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MPFIT_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_fitter_mpfit_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_FITTER_MPFIT_INCLUDE_DIRS ${GBKFIT_FITTER_MPFIT_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_fitter_mpfit_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_fitter_mpfit_static STATIC $<TARGET_OBJECTS:target_gbkfit_fitter_mpfit_object>)
add_library(target_gbkfit_fitter_mpfit_shared SHARED $<TARGET_OBJECTS:target_gbkfit_fitter_mpfit_object>)

target_include_directories(target_gbkfit_fitter_mpfit_object PUBLIC "./include")
target_include_directories(target_gbkfit_fitter_mpfit_object PUBLIC ${GBKFIT_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_fitter_mpfit_static PUBLIC ${GBKFIT_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_fitter_mpfit_shared PUBLIC ${GBKFIT_LIBRARY_SHARED})

target_compile_definitions(target_gbkfit_fitter_mpfit_object PRIVATE "GBKFIT_FITTER_MPFIT_SELF_BUILD")

target_compile_options(target_gbkfit_fitter_mpfit_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_fitter_mpfit_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_fitter_mpfit_object PRIVATE "-Wpedantic")

set_target_properties(target_gbkfit_fitter_mpfit_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_fitter_mpfit_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_fitter_mpfit_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_fitter_mpfit_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_fitter_mpfit_static PROPERTIES OUTPUT_NAME "gbkfit_fitter_mpfit_static")
set_target_properties(target_gbkfit_fitter_mpfit_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_fitter_mpfit_shared PROPERTIES OUTPUT_NAME "gbkfit_fitter_mpfit_shared")
set_target_properties(target_gbkfit_fitter_mpfit_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS target_gbkfit_fitter_mpfit_static target_gbkfit_fitter_mpfit_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
