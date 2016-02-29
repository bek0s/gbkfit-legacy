
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

if(${GBKFIT_BUILD_FITTER_MPFIT})
    add_subdirectory(gbkfit_fitter_mpfit)
endif()

if(${GBKFIT_BUILD_FITTER_MULTINEST})
    add_subdirectory(gbkfit_fitter_multinest)
endif()

#
# Models
#

if(${GBKFIT_BUILD_MODEL_MODEL01})
    add_subdirectory(gbkfit_model_model01)
endif()

if(${GBKFIT_BUILD_MODEL_MODEL01_CUDA})
    add_subdirectory(gbkfit_model_model01_cuda)
endif()

if(${GBKFIT_BUILD_MODEL_MODEL01_OMP})
    add_subdirectory(gbkfit_model_model01_omp)
endif()
