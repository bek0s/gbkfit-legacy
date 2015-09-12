#pragma once
#ifndef GBKFIT_MODEL_THINDISK_KERNELS_OMP_HPP
#define GBKFIT_MODEL_THINDISK_KERNELS_OMP_HPP

#include <complex>
#include <fftw3.h>
#include <omp.h>

namespace gbkfit {
namespace model_thindisk {
namespace kernels_omp {

void model_image_3d_evaluate(float* out,
                             int model_id,
                             int data_width,
                             int data_height,
                             int data_depth,
                             int data_aligned_width,
                             int data_aligned_height,
                             int data_aligned_depth,
                             int data_margin_width_0,
                             int data_margin_width_1,
                             int data_margin_height_0,
                             int data_margin_height_1,
                             int data_margin_depth_0,
                             int data_margin_depth_1,
                             const float* model_params_proj,
                             int model_params_proj_length,
                             const float* model_params_flux,
                             int model_params_flux_length,
                             const float* model_params_rcur,
                             int model_params_rcur_length,
                             const float* model_params_vsig,
                             int model_params_vsig_length,
                             const float* cube_sampling,
                             int cube_sampling_length);

void model_image_3d_convolve_fft(float* inout_img,
                                 std::complex<float>* img_fft,
                                 const std::complex<float>* krl_fft,
                                 int width,
                                 int height,
                                 int depth,
                                 int batch,
                                 fftwf_plan plan_r2c,
                                 fftwf_plan plan_c2r);

void model_image_3d_downsample(float* data_dst,
                               float* data_src,
                               int data_downsampled_width,
                               int data_downsampled_height,
                               int data_downsampled_depth,
                               int data_aligned_width,
                               int data_aligned_height,
                               int data_aligned_depth,
                               int data_margin_width_0,
                               int data_margin_height_0,
                               int data_margin_depth_0,
                               int downsample_x,
                               int downsample_y,
                               int downsample_z);

void model_image_3d_extract_moment_maps(float* out_velmap,
                                        float* out_sigmap,
                                        const float* cube,
                                        int data_width,
                                        int data_height,
                                        int data_depth,
                                        const float* cube_sampling,
                                        int cube_sampling_length,
                                        float velmap_offset,
                                        float sigmap_offset);

} // namespace kernels_omp
} // namespace model_thindisk
} // namespace gbkfit

#endif  //  GBKFIT_MODEL_THINDISK_KERNELS_OMP_HPP
