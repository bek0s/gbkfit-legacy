#pragma once
#ifndef GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_KERNELS_H_HPP
#define GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_KERNELS_H_HPP

#include <cufft.h>

namespace gbkfit {
namespace dmodel {
namespace scube {
namespace kernels_cuda_h
{

void fill_cube(int size_x,
               int size_y,
               int size_z,
               float value,
               float* data);

void convolve_cube(float* flxcube,
                   cufftComplex* flxcube_complex,
                   cufftComplex* psfcube_complex,
                   int size_x,
                   int size_y,
                   int size_z,
                   cufftHandle plan_r2c,
                   cufftHandle plan_c2r);

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

} // namespace kernels_cuda_h
} // namespace scube
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_KERNELS_H_HPP
