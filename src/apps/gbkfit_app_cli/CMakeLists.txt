
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

project(gbkfit_app_cli-project)

# ==============================================================================

find_package(cfitsio)
set(Boost_NO_BOOST_CMAKE ON)
find_package(Boost COMPONENTS program_options system REQUIRED)

# ==============================================================================

set(HEADER_FILES
    include/application.hpp
    include/prerequisites.hpp
)

set(SOURCE_FILES
    src/application.cpp
    src/main.cpp
)

# ==============================================================================

add_executable(target_gbkfit_app_cli ${HEADER_FILES} ${SOURCE_FILES})

target_include_directories(target_gbkfit_app_cli PRIVATE "include")
target_include_directories(target_gbkfit_app_cli PRIVATE ${GBKFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_app_cli PRIVATE ${GBKFIT_CUDA_INCLUDE_DIRS})
target_include_directories(target_gbkfit_app_cli PRIVATE ${GBKFIT_FITTER_MPFIT_INCLUDE_DIRS})
target_include_directories(target_gbkfit_app_cli PRIVATE ${GBKFIT_FITTER_MULTINEST_INCLUDE_DIRS})
target_include_directories(target_gbkfit_app_cli PRIVATE ${GBKFIT_MODEL_MODEL01_INCLUDE_DIRS})
target_include_directories(target_gbkfit_app_cli PRIVATE ${GBKFIT_MODEL_MODEL01_OMP_INCLUDE_DIRS})
target_include_directories(target_gbkfit_app_cli PRIVATE ${GBKFIT_MODEL_MODEL01_CUDA_INCLUDE_DIRS})
target_include_directories(target_gbkfit_app_cli PRIVATE ${Boost_INCLUDE_DIRS})


target_link_libraries(target_gbkfit_app_cli PRIVATE ${GBKFIT_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_app_cli PRIVATE ${GBKFIT_CUDA_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_app_cli PRIVATE ${GBKFIT_FITTER_MPFIT_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_app_cli PRIVATE ${GBKFIT_FITTER_MULTINEST_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_app_cli PRIVATE ${GBKFIT_MODEL_MODEL01_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_app_cli PRIVATE ${GBKFIT_MODEL_MODEL01_OMP_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_app_cli PRIVATE ${GBKFIT_MODEL_MODEL01_CUDA_LIBRARY_SHARED})
target_link_libraries(target_gbkfit_app_cli PRIVATE ${Boost_LIBRARIES})
target_link_libraries(target_gbkfit_app_cli PRIVATE cufft)

target_compile_options(target_gbkfit_app_cli PRIVATE "-Wall")
target_compile_options(target_gbkfit_app_cli PRIVATE "-Wextra")
target_compile_options(target_gbkfit_app_cli PRIVATE "-Wpedantic")
target_compile_options(target_gbkfit_app_cli PRIVATE "-Wsign-conversion")

set_target_properties(target_gbkfit_app_cli PROPERTIES CXX_EXTENSIONS OFF)
set_target_properties(target_gbkfit_app_cli PROPERTIES CXX_STANDARD 14)
set_target_properties(target_gbkfit_app_cli PROPERTIES CXX_STANDARD_REQUIRED ON)

set_target_properties(target_gbkfit_app_cli PROPERTIES OUTPUT_NAME "gbkfit_app_cli")
set_target_properties(target_gbkfit_app_cli PROPERTIES DEBUG_POSTFIX "_d")

# ==============================================================================

install(TARGETS
        target_gbkfit_app_cli
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib)

# ==============================================================================
