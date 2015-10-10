
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_models_model01_omp-project)

# ==============================================================================

find_package(fftw3)
find_package(OpenMP)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/models/model01/kernels_omp.hpp
    include/gbkfit/models/model01/model_model01_omp.hpp
)

set(SOURCE_FILES
    src/kernels_omp.cpp
    src/model_model01_omp.cpp
)

# ==============================================================================

set(GBKFIT_MODELS_MODEL01_OMP_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_models_model01_omp_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODELS_MODEL01_OMP_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_models_model01_omp_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODELS_MODEL01_OMP_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_models_model01_omp_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODELS_MODEL01_OMP_INCLUDE_DIRS ${GBKFIT_MODELS_MODEL01_OMP_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_models_model01_omp_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_models_model01_omp_static STATIC $<TARGET_OBJECTS:target_gbkfit_models_model01_omp_object>)
add_library(target_gbkfit_models_model01_omp_shared SHARED $<TARGET_OBJECTS:target_gbkfit_models_model01_omp_object>)

target_include_directories(target_gbkfit_models_model01_omp_object PUBLIC include)
target_include_directories(target_gbkfit_models_model01_omp_object PRIVATE ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_models_model01_omp_object PRIVATE ${GBKFIT_MODELS_MODEL01_INCLUDE_DIRS})
target_include_directories(target_gbkfit_models_model01_omp_object PRIVATE ${FFTW3_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_models_model01_omp_static PRIVATE ${GBKFIT_LIBRARIES})
target_link_libraries(target_gbkfit_models_model01_omp_static PRIVATE ${GBKFIT_MODELS_MODEL01_LIBRARIES})
target_link_libraries(target_gbkfit_models_model01_omp_static PRIVATE ${FFTW3_LIBRARIES})
target_link_libraries(target_gbkfit_models_model01_omp_shared PRIVATE ${GBKFIT_LIBRARIES})
target_link_libraries(target_gbkfit_models_model01_omp_shared PRIVATE ${GBKFIT_MODELS_MODEL01_LIBRARIES})
target_link_libraries(target_gbkfit_models_model01_omp_shared PRIVATE ${FFTW3_LIBRARIES})

target_compile_definitions(target_gbkfit_models_model01_omp_object PRIVATE "GBKFIT_MODELS_MODEL01_OMP_SELF_BUILD")

target_compile_options(target_gbkfit_models_model01_omp_object PRIVATE "-fopenmp")
target_compile_options(target_gbkfit_models_model01_omp_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_models_model01_omp_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_models_model01_omp_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_models_model01_omp_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_models_model01_omp_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_models_model01_omp_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_models_model01_omp_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_models_model01_omp_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_models_model01_omp_static PROPERTIES OUTPUT_NAME "gbkfit_models_model01_omp_static")
set_target_properties(target_gbkfit_models_model01_omp_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_models_model01_omp_shared PROPERTIES OUTPUT_NAME "gbkfit_models_model01_omp_shared")
set_target_properties(target_gbkfit_models_model01_omp_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_gbkfit_models_model01_omp_static
        target_gbkfit_models_model01_omp_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
