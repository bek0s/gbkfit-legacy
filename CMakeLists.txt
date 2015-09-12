
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

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
#   Create user options
#
# ==============================================================================

option(GBKFIT_BUILD_GBKFIT_APP_CLI "build gbkfit_app_cli" TRUE)

option(GBKFIT_BUILD_GBKFIT_CUDA "build gbkfit_cuda" TRUE)
option(GBKFIT_BUILD_GBKFIT_FFTW3 "build gbkfit_fftw3" TRUE)

option(GBKFIT_BUILD_GBKFIT_FITTERS_MPFIT "build gbkfit_fitters_mpfit" TRUE)
option(GBKFIT_BUILD_GBKFIT_FITTERS_MULTINEST "build gbkfit_fitters_multinest" TRUE)

option(GBKFIT_BUILD_GBKFIT_DATAMODELS_SPECUBE "build gbkfit_datamodels_specube" TRUE)
option(GBKFIT_BUILD_GBKFIT_DATAMODELS_SPECUBE_CUDA "build gbkfit_datamodels_specube_cuda" TRUE)
option(GBKFIT_BUILD_GBKFIT_DATAMODELS_SPECUBE_OMP "build gbkfit_datamodels_specube_omp" TRUE)

option(GBKFIT_BUILD_GBKFIT_DATAMODELS_MOMENTS "build gbkfit_datamodels_moments" TRUE)
option(GBKFIT_BUILD_GBKFIT_DATAMODELS_MOMENTS_CUDA "build gbkfit_datamodels_moments_cuda" TRUE)
option(GBKFIT_BUILD_GBKFIT_DATAMODELS_MOMENTS_OMP "build gbkfit_datamodels_moments_omp" TRUE)

option(GBKFIT_BUILD_GBKFIT_GALMODELS_GALMODEL01 "build gbkfit_galmodels_galmodel01" TRUE)
option(GBKFIT_BUILD_GBKFIT_GALMODELS_GALMODEL01_CUDA "build gbkfit_galmodels_galmodel01_cuda" TRUE)
option(GBKFIT_BUILD_GBKFIT_GALMODELS_GALMODEL01_OMP "build gbkfit_galmodels_galmodel01_omp" TRUE)

option(GBKFIT_BUILD_GBKFIT_MODELS_GALAXY_2D "build gbkfit_models_galaxy_2d" TRUE)
option(GBKFIT_BUILD_GBKFIT_MODELS_GALAXY_2D_CUDA "build gbkfit_models_galaxy_2d_cuda" TRUE)
option(GBKFIT_BUILD_GBKFIT_MODELS_GALAXY_2D_OMP "build gbkfit_models_galaxy_2d_omp" TRUE)

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
