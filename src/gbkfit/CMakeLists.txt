
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

add_subdirectory(gbkfit)

if(${GBKFIT_BUILD_GBKFIT_CUDA})
    add_subdirectory(gbkfit_cuda)
endif()

if(${GBKFIT_BUILD_GBKFIT_FFTW3})
    add_subdirectory(gbkfit_fftw3)
endif()

if(${GBKFIT_BUILD_GBKFIT_FITTERS_MPFIT})
    add_subdirectory(gbkfit_fitters_mpfit)
endif()

if(${GBKFIT_BUILD_GBKFIT_FITTERS_MULTINEST})
    add_subdirectory(gbkfit_fitters_multinest)
endif()

if(${GBKFIT_BUILD_GBKFIT_MODELS_GALAXY_2D})
    add_subdirectory(gbkfit_models_galaxy_2d)
endif()

if(${GBKFIT_BUILD_GBKFIT_MODELS_GALAXY_2D_CUDA})
    add_subdirectory(gbkfit_models_galaxy_2d_cuda)
endif()

if(${GBKFIT_BUILD_GBKFIT_MODELS_GALAXY_2D_OMP})
    add_subdirectory(gbkfit_models_galaxy_2d_omp)
endif()

#add_subdirectory(gbkfit_model_thindisk)
