
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_model_thindisk-project)

# ==============================================================================

find_package(CUDA REQUIRED)
find_package(fftw3 REQUIRED)
find_package(OpenMP REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/gbkfit/model_thindisk/model_thindisk.hpp
    include/gbkfit/model_thindisk/model_thindisk_arctan.hpp
    include/gbkfit/model_thindisk/model_thindisk_epinat.hpp

    include/gbkfit/model_thindisk/model_thindisk_kernels_omp.hpp

)

set(SOURCE_FILES
    src/model_thindisk.cpp
    src/model_thindisk_arctan.cpp
    src/model_thindisk_epinat.cpp

    src/model_thindisk_kernels_omp.cpp
)

# ==============================================================================

set(GBKFIT_MODEL_THINDISK_LIBRARY_STATIC $<TARGET_FILE:target_gbkfit_model_thindisk_static> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODEL_THINDISK_LIBRARY_SHARED $<TARGET_FILE:target_gbkfit_model_thindisk_shared> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODEL_THINDISK_INCLUDE_DIR $<TARGET_PROPERTY:target_gbkfit_model_thindisk_object,INTERFACE_INCLUDE_DIRECTORIES> CACHE INTERNAL "" FORCE)
set(GBKFIT_MODEL_THINDISK_INCLUDE_DIRS ${GBKFIT_MODEL_THINDISK_INCLUDE_DIR} CACHE INTERNAL "" FORCE)

# ==============================================================================

add_library(target_gbkfit_model_thindisk_object OBJECT ${HEADER_FILES} ${SOURCE_FILES})
add_library(target_gbkfit_model_thindisk_static STATIC $<TARGET_OBJECTS:target_gbkfit_model_thindisk_object>)
add_library(target_gbkfit_model_thindisk_shared SHARED $<TARGET_OBJECTS:target_gbkfit_model_thindisk_object>)

target_include_directories(target_gbkfit_model_thindisk_object PUBLIC "./include")
target_include_directories(target_gbkfit_model_thindisk_object PUBLIC ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_model_thindisk_object PRIVATE ${FFTW3_INCLUDE_DIRS})

target_link_libraries(target_gbkfit_model_thindisk_static PRIVATE ${GBKFIT_LIBRARIES})
target_link_libraries(target_gbkfit_model_thindisk_shared PRIVATE ${GBKFIT_LIBRARIES})

target_link_libraries(target_gbkfit_model_thindisk_static PRIVATE ${FFTW3_LIBRARIES})
target_link_libraries(target_gbkfit_model_thindisk_shared PRIVATE ${FFTW3_LIBRARIES})

target_compile_definitions(target_gbkfit_model_thindisk_object PRIVATE "GBKFIT_MODEL_THINDISK_SELF_BUILD")

target_compile_options(target_gbkfit_model_thindisk_object PRIVATE "-fopenmp")
target_compile_options(target_gbkfit_model_thindisk_object PRIVATE "-Wall")
target_compile_options(target_gbkfit_model_thindisk_object PRIVATE "-Wextra")
target_compile_options(target_gbkfit_model_thindisk_object PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_model_thindisk_object PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_model_thindisk_object PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_model_thindisk_object PROPERTIES CXX_STANDARD 11)
set_target_properties(target_gbkfit_model_thindisk_object PROPERTIES CXX_STANDARD_REQUIRED ON)
set_target_properties(target_gbkfit_model_thindisk_object PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(target_gbkfit_model_thindisk_static PROPERTIES OUTPUT_NAME "gbkfit_model_thindisk_static")
set_target_properties(target_gbkfit_model_thindisk_static PROPERTIES DEBUG_POSTFIX "_d")
set_target_properties(target_gbkfit_model_thindisk_shared PROPERTIES OUTPUT_NAME "gbkfit_model_thindisk_shared")
set_target_properties(target_gbkfit_model_thindisk_shared PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS target_gbkfit_model_thindisk_static target_gbkfit_model_thindisk_shared
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

install(DIRECTORY include/ DESTINATION include)

# ==============================================================================
