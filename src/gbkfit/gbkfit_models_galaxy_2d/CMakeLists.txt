
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_models_galaxy_2d-project)

# ==============================================================================

#find_package(foo REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/models/galaxy_2d/model_factory_galaxy_2d.hpp
    include/gbkfit/models/galaxy_2d/model_galaxy_2d.hpp
)

set(SOURCE_FILES
    src/model_factory_galaxy_2d.cpp
    src/model_galaxy_2d.cpp
)

# ==============================================================================

set(GBKFIT_MODELS_GALAXY_2D_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_models_galaxy_2d_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODELS_GALAXY_2D_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_models_galaxy_2d_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODELS_GALAXY_2D_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_models_galaxy_2d_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODELS_GALAXY_2D_INCLUDE_DIRS ${GBKFIT_MODELS_GALAXY_2D_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_models_galaxy_2d_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_models_galaxy_2d_static STATIC $<TARGET_OBJECTS:target_gbkfit_models_galaxy_2d_object>)
add_library(target_gbkfit_models_galaxy_2d_shared SHARED $<TARGET_OBJECTS:target_gbkfit_models_galaxy_2d_object>)

target_include_directories(target_gbkfit_models_galaxy_2d_object PUBLIC "./include")
target_include_directories(target_gbkfit_models_galaxy_2d_object PUBLIC ${GBKFIT_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_models_galaxy_2d_static PRIVATE ${GBKFIT_LIBRARIES})
target_link_libraries(target_gbkfit_models_galaxy_2d_shared PRIVATE ${GBKFIT_LIBRARIES})

target_compile_options(target_gbkfit_models_galaxy_2d_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_models_galaxy_2d_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_models_galaxy_2d_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_models_galaxy_2d_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_models_galaxy_2d_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_models_galaxy_2d_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_models_galaxy_2d_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_models_galaxy_2d_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_models_galaxy_2d_static PROPERTIES OUTPUT_NAME "gbkfit_models_galaxy_2d_static")
set_target_properties(target_gbkfit_models_galaxy_2d_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_models_galaxy_2d_shared PROPERTIES OUTPUT_NAME "gbkfit_models_galaxy_2d_shared")
set_target_properties(target_gbkfit_models_galaxy_2d_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS target_gbkfit_models_galaxy_2d_static target_gbkfit_models_galaxy_2d_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
