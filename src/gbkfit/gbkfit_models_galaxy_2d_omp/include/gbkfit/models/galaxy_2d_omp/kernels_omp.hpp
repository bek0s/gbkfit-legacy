#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP
#define GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP

namespace gbkfit {
namespace models {
namespace galaxy_2d {
namespace kernels_omp {

void model_image_3d_evaluate(int profile_flx_id,
                             int profile_vel_id,
                             const float param_vsig,
                             const float param_vsys,
                             const float* params_prj,
                             const float* params_flx,
                             const float* params_vel,
                             int data_size_x,
                             int data_size_y,
                             int data_size_z,
                             int data_size_x_padded,
                             int data_size_y_padded,
                             int data_size_z_padded,

                             int data_padding_x,
                             int data_padding_y,
                             int data_padding_z,

                             float data_step_x,
                             float data_step_y,
                             float data_step_z,

                             int upsampling_x,
                             int upsampling_y,
                             int upsampling_z,

                             float* flxcube_padded);

void model_image_3d_copy(const float* flxcube_src,
                         int data_size_x,
                         int data_size_y,
                         int data_size_z,
                         int data_size_x_padded,
                         int data_size_y_padded,
                         int data_size_z_padded,
                         float* flxcube_dst);

void model_image_3d_downsample_and_copy(const float* flxcube_src,
                                        int data_size_x,
                                        int data_size_y,
                                        int data_size_z,
                                        int data_size_x_padded,
                                        int data_size_y_padded,
                                        int data_size_z_padded,
                                        int data_padding_x,
                                        int data_padding_y,
                                        int data_padding_z,
                                        int downsample_x,
                                        int downsample_y,
                                        int downsample_z,
                                        float* flxcube_dst);

void model_image_3d_extract_moment_maps(const float* flxcube,
                                        int data_size_x,
                                        int data_size_y,
                                        int data_size_z,
                                        float data_step_x,
                                        float data_step_y,
                                        float data_step_z,
                                        float* flxmap,
                                        float* velmap,
                                        float* sigmap);

} // namespace kernels_omp
} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif  //  GBKFIT_MODELS_GALAXY_2D_KERNELS_OMP_HPP
