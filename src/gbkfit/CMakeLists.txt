
# ==============================================================================
#
#   CMake configuration file
#
# ==============================================================================

add_subdirectory(gbkfit)

if(${GBKFIT_BUILD_CUDA})
    add_subdirectory(gbkfit_cuda)
endif()

if(${GBKFIT_BUILD_FFTW3})
    add_subdirectory(gbkfit_fftw3)
endif()

#
# Fitters
#

if(${GBKFIT_BUILD_FITTERS_MPFIT})
    add_subdirectory(gbkfit_fitters_mpfit)
endif()

if(${GBKFIT_BUILD_FITTERS_MULTINEST})
    add_subdirectory(gbkfit_fitters_multinest)
endif()

#
# Models
#

if(${GBKFIT_BUILD_MODELS_MODEL01})
    add_subdirectory(gbkfit_models_model01)
endif()

if(${GBKFIT_BUILD_MODELS_MODEL01_CUDA})
    add_subdirectory(gbkfit_models_model01_cuda)
endif()

if(${GBKFIT_BUILD_MODELS_MODEL01_OMP})
    add_subdirectory(gbkfit_models_model01_omp)
endif()
