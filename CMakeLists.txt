
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

cmake_minimum_required(VERSION 3.4 FATAL_ERROR)

project(gbkfit-workspace)

# ==============================================================================
#
#   Setup paths and directories
#
# ==============================================================================

# Set module search directories
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules")

# Set output directories
set(GBKFIT_RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin")
set(GBKFIT_ARCHIVE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(GBKFIT_LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/lib")
set(GBKFIT_INCLUDE_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/include")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${GBKFIT_RUNTIME_OUTPUT_DIRECTORY}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${GBKFIT_ARCHIVE_OUTPUT_DIRECTORY}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${GBKFIT_LIBRARY_OUTPUT_DIRECTORY}")

# ==============================================================================
#
#   Include modules
#
# ==============================================================================

# ==============================================================================
#
#   Create user options
#
# ==============================================================================

option(GBKFIT_BUILD_APP_CLI "Build gbkfit_app_cli" TRUE)

option(GBKFIT_BUILD_CUDA "Build gbkfit_cuda" TRUE)
option(GBKFIT_BUILD_FFTW3 "Build gbkfit_fftw3" TRUE)

option(GBKFIT_BUILD_DMODEL_SCUBE "Build gbkfit_dmodel_scube" TRUE)
option(GBKFIT_BUILD_DMODEL_SCUBE_CUDA "Build gbkfit_dmodel_scube_cuda" TRUE)
option(GBKFIT_BUILD_DMODEL_SCUBE_OMP "Build gbkfit_dmodel_scube_omp" TRUE)

option(GBKFIT_BUILD_DMODEL_MMAPS "Build gbkfit_dmodel_mmaps" TRUE)
option(GBKFIT_BUILD_DMODEL_MMAPS_CUDA "Build gbkfit_dmodel_mmaps_cuda" TRUE)
option(GBKFIT_BUILD_DMODEL_MMAPS_OMP "Build gbkfit_dmodel_mmaps_omp" TRUE)

option(GBKFIT_BUILD_GMODEL_GMODEL1 "Build gbkfit_gmodel_gmodel1" TRUE)
option(GBKFIT_BUILD_GMODEL_GMODEL1_CUDA "Build gbkfit_gmodel_gmodel1_cuda" TRUE)
option(GBKFIT_BUILD_GMODEL_GMODEL1_OMP "Build gbkfit_gmodel_gmodel1_omp" TRUE)

option(GBKFIT_BUILD_FITTER_MPFIT "Build gbkfit_fitter_mpfit" TRUE)
option(GBKFIT_BUILD_FITTER_MULTINEST "Build gbkfit_fitter_multinest" TRUE)

option(GBKFIT_BUILD_UNITTESTS_GBKFIT "Build unit tests for gbkfit" FALSE)

# ==============================================================================
#
#   Configure build
#
# ==============================================================================

# Build the correct RPATHs for both the build and installation trees
SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(GBKFIT_VERSION_MAJOR 0)
set(GBKFIT_VERSION_MINOR 0)
set(GBKFIT_VERSION_PATCH 1)

configure_file(cmake/templates/build_settings.hpp.in ${PROJECT_BINARY_DIR}/include/gbkfit/build_settings.hpp)
configure_file(cmake/templates/version.cpp.in ${PROJECT_BINARY_DIR}/src/version.cpp)

include_directories(${PROJECT_BINARY_DIR}/include)

# ==============================================================================
#
#   Add subdirectories
#
# ==============================================================================

add_subdirectory(src)

# ==============================================================================
#
#   Setup uninstall target
#
# ==============================================================================

configure_file("cmake/templates/cmake_uninstall.cmake.in"
               "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
	       @ONLY)

add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
