#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_OMP_KERNELS_HPP
#define GBKFIT_DMODEL_MMAPS_MMAPS_OMP_KERNELS_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {
namespace kernels_omp {

void extract_maps_mmnt(const float* cube,
                       int size_x,
                       int size_y,
                       int size_z,
                       float zero_z,
                       float step_z,
                       float* mom0,
                       float* mom1,
                       float* mom2);

void extract_maps_gfit(const float* cube,
                       int size_x,
                       int size_y,
                       int size_z,
                       float zero_z,
                       float step_z,
                       float* mom0,
                       float* mom1,
                       float* mom2);

} // namespace kernels_omp
} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_OMP_KERNELS_HPP
