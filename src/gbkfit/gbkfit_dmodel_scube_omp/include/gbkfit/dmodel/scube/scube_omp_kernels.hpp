#pragma once
#ifndef GBKFIT_DMODEL_SCUBE_SCUBE_OMP_KERNELS_HPP
#define GBKFIT_DMODEL_SCUBE_SCUBE_OMP_KERNELS_HPP

#include "gbkfit/prerequisites.hpp"

#include <fftw3.h>

namespace gbkfit {
namespace dmodel {
namespace scube {
namespace kernels_omp {

void clear_cube(int size_x,
                int size_y,
                int size_z,
                float value,
                float* data);

void downsample_cube(int size_x,
                     int size_y,
                     int size_z,
                     int size_up_x,
                     int size_up_y,
                     int size_up_z,
                     int offset_x,
                     int offset_y,
                     int offset_z,
                     int downsample_x,
                     int downsample_y,
                     int downsample_z,
                     const float* cube_up,
                     float* cube);

void model_image_3d_convolve_fft(float* flxcube,
                                 fftwf_complex *flxcube_complex,
                                 fftwf_complex *psfcube_complex,
                                 int size_x,
                                 int size_y,
                                 int size_z,
                                 fftwf_plan plan_r2c,
                                 fftwf_plan plan_c2r);

} // namespace kernels_omp
} // namespace scube
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_SCUBE_SCUBE_OMP_KERNELS_HPP
