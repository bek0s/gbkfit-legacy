
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

option(GBKFIT_BUILD_CUDA "Build gbkfit_cuda" FALSE)
option(GBKFIT_BUILD_FFTW3 "Build gbkfit_fftw3" TRUE)

option(GBKFIT_BUILD_FITTER_MPFIT "Build gbkfit_fitter_mpfit" TRUE)
option(GBKFIT_BUILD_FITTER_MULTINEST "Build gbkfit_fitter_multinest" FALSE)

option(GBKFIT_BUILD_MODEL_MODEL01 "Build gbkfit_model_model01" TRUE)
option(GBKFIT_BUILD_MODEL_MODEL01_CUDA "Build gbkfit_model_model01_cuda" FALSE)
option(GBKFIT_BUILD_MODEL_MODEL01_OMP "Build gbkfit_model_model01_omp" TRUE)

#option(GBKFIT_BUILD_UNITTESTS_GBKFIT "Build unit tests for gbkfit" TRUE)

# ==============================================================================
#
#   Configure build
#
# ==============================================================================

configure_file(cmake/templates/build_settings.hpp.in src/gbkfit/gbkfit/include/gbkfit/build_settings.hpp)

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
