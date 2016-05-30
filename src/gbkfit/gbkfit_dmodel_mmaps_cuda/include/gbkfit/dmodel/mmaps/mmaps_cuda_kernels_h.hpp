#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_KERNELS_H_HPP
#define GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_KERNELS_H_HPP

namespace gbkfit {
namespace dmodel {
namespace mmaps {
namespace kernels_cuda_h
{

void extract_maps_mmnt(const float* cube,
                       int size_x,
                       int size_y,
                       int size_z,
                       float zerp_z,
                       float step_z,
                       float* mom0,
                       float* mom1,
                       float* mom2);

} // namespace kernels_cuda_h
} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_KERNELS_H_HPP
