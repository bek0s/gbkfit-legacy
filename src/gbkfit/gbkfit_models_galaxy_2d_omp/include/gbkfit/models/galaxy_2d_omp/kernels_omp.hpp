#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP
#define GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP

#include <complex>
#include <fftw3.h>
#include <omp.h>

namespace gbkfit {
namespace models {
namespace galaxy_2d {
namespace kernels_omp {


void model_image_2d_evaluate(float* out_flxmap,
                             float* out_velmap,
                             float* out_sigmap,
                             int model_flux_id,
                             int model_rcur_id,
                             int data_size_x,
                             int data_size_y,
                             float step_x,
                             float step_y,
                             const float model_parameter_vsys,
                             const float* model_parameters_proj,
                             int model_parameters_proj_length,
                             const float* model_parameters_flux,
                             int model_parameters_flux_length,
                             const float* model_parameters_rcur,
                             int model_parameters_rcur_length,
                             const float* model_parameters_vsig,
                             int model_parameters_vsig_length);

void model_image_3d_evaluate(float* out_cube,
                             int model_flux_id,
                             int model_rcur_id,
                             int data_width,
                             int data_height,
                             int data_depth,
                             int data_aligned_width,
                             int data_aligned_height,
                             int data_aligned_depth,
                             float sampling_x,
                             float sampling_y,
                             float sampling_z,
                             const float model_parameter_vsys,
                             const float* model_parameters_proj,
                             int model_parameters_proj_length,
                             const float* model_parameters_flux,
                             int model_parameters_flux_length,
                             const float* model_parameters_rcur,
                             int model_parameters_rcur_length,
                             const float* model_parameters_vsig,
                             int model_parameters_vsig_length);


} // namespace kernels_omp
} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif  //  GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP
