
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
# DModels
#

if(${GBKFIT_BUILD_DMODEL})
    add_subdirectory(gbkfit_dmodel)
endif()

if(${GBKFIT_BUILD_DMODEL_OMP})
    add_subdirectory(gbkfit_dmodel_omp)
endif()

if(${GBKFIT_BUILD_DMODEL_CUDA})
    add_subdirectory(gbkfit_dmodel_cuda)
endif()

#
# GModels
#

if(${GBKFIT_BUILD_GMODEL})
    add_subdirectory(gbkfit_gmodel)
endif()

if(${GBKFIT_BUILD_GMODEL_OMP})
    add_subdirectory(gbkfit_gmodel_omp)
endif()

if(${GBKFIT_BUILD_GMODEL_CUDA})
    add_subdirectory(gbkfit_gmodel_cuda)
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
