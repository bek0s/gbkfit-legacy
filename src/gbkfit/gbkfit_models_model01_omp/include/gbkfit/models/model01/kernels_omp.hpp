#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP
#define GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP

#include <fftw3.h>

namespace gbkfit {
namespace models {
namespace model01 {
namespace kernels_omp {

void array_3d_fill(int size_x,
                   int size_y,
                   int size_z,
                   float value,
                   float* data);

void model_image_3d_evaluate(int profile_flx_id,
                             int profile_vel_id,
                             const float param_vsig,
                             const float param_vsys,
                             const float* params_prj,
                             const float* params_flx,
                             const float* params_vel,
                             int size_u_x,
                             int size_u_y,
                             int size_u_z,
                             int size_up_x,
                             int size_up_y,
                             int size_up_z,
                             float step_u_x,
                             float step_u_y,
                             float step_u_z,
                             int padding_u_x0,
                             int padding_u_y0,
                             int padding_u_z0,
                             int padding_u_x1,
                             int padding_u_y1,
                             int padding_u_z1,
                             float* flxcube_up);

void model_image_3d_convolve_fft(float* image,
                                 float* image_complex,
                                 float* kernel_complex,
                                 int size_x,
                                 int size_y,
                                 int size_z,
                                 fftwf_plan plan_r2c,
                                 fftwf_plan plan_c2r);

void model_image_3d_downsample_and_copy(const float* flxcube_src,
                                        int size_x,
                                        int size_y,
                                        int size_z,
                                        int upsampled_padding_x0,
                                        int upsampled_padding_y0,
                                        int upsampled_padding_z0,
                                        int upsampled_padded_size_x,
                                        int upsampled_padded_size_y,
                                        int upsampled_padded_size_z,
                                        int downsample_x,
                                        int downsample_y,
                                        int downsample_z,
                                        float* flxcube_dst);

void model_image_3d_extract_moment_maps(const float* flxcube,
                                        int size_x,
                                        int size_y,
                                        int size_z,
                                        float step_x,
                                        float step_y,
                                        float step_z,
                                        float* flxmap,
                                        float* velmap,
                                        float* sigmap);

} // namespace kernels_omp
} // namespace model01
} // namespace models
} // namespace gbkfit

#endif  //  GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP
