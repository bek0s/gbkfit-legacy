#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_HOST_HPP
#define GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_HOST_HPP

#include <cufft.h>

namespace gbkfit {
namespace models {
namespace model01 {
namespace kernels_cuda_host {

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

void model_image_3d_convolve_fft(float* flxcube,
                                 cufftComplex* flxcube_complex,
                                 cufftComplex* psfcube_complex,
                                 int size_x,
                                 int size_y,
                                 int size_z,
                                 cufftHandle plan_r2c,
                                 cufftHandle plan_c2r);

void model_image_3d_downsample_and_copy(const float* flxcube_up,
                                        int size_x,
                                        int size_y,
                                        int size_z,
                                        int size_up_x,
                                        int size_up_y,
                                        int size_up_z,
                                        int padding_u_x0,
                                        int padding_u_y0,
                                        int padding_u_z0,
                                        int downsample_x,
                                        int downsample_y,
                                        int downsample_z,
                                        float* flxcube);

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

} // namespace kernels_cuda_host
} // namespace model01
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_HOST_HPP
