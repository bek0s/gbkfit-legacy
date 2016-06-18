
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

if(${GBKFIT_BUILD_DMODEL_SCUBE})
    add_subdirectory(gbkfit_dmodel_scube)
endif()

if(${GBKFIT_BUILD_DMODEL_SCUBE_CUDA})
    add_subdirectory(gbkfit_dmodel_scube_cuda)
endif()

if(${GBKFIT_BUILD_DMODEL_SCUBE_OMP})
    add_subdirectory(gbkfit_dmodel_scube_omp)
endif()

if(${GBKFIT_BUILD_DMODEL_MMAPS})
    add_subdirectory(gbkfit_dmodel_mmaps)
endif()

if(${GBKFIT_BUILD_DMODEL_MMAPS_CUDA})
    add_subdirectory(gbkfit_dmodel_mmaps_cuda)
endif()

if(${GBKFIT_BUILD_DMODEL_MMAPS_OMP})
    add_subdirectory(gbkfit_dmodel_mmaps_omp)
endif()

#
# GModels
#

if(${GBKFIT_BUILD_GMODEL_GMODEL1})
    add_subdirectory(gbkfit_gmodel_gmodel1)
endif()

if(${GBKFIT_BUILD_GMODEL_GMODEL1_CUDA})
    add_subdirectory(gbkfit_gmodel_gmodel1_cuda)
endif()

if(${GBKFIT_BUILD_GMODEL_GMODEL1_OMP})
    add_subdirectory(gbkfit_gmodel_gmodel1_omp)
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

