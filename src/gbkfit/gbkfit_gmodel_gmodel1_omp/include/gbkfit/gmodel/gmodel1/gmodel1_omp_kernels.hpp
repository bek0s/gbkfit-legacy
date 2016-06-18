#pragma once
#ifndef GBKFIT_GMODEL_GMODEL1_GMODEL1_OMP_KERNELS_HPP
#define GBKFIT_GMODEL_GMODEL1_GMODEL1_OMP_KERNELS_HPP

#include "gbkfit/gmodel/gmodel1/gmodel1.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {
namespace kernels_omp {

void evaluate_model(int flx_profile,
                    int vel_profile,
                    const float* params_prj,
                    const float* params_flx,
                    const float* params_vel,
                    const float param_vsys,
                    const float param_vsig,
                    int data_size_x,
                    int data_size_y,
                    int data_size_z,
                    float data_zero_x,
                    float data_zero_y,
                    float data_zero_z,
                    float data_step_x,
                    float data_step_y,
                    float data_step_z,
                    float* data);

} // namespace kernels_omp
} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit

#endif // GBKFIT_GMODEL_GMODEL1_GMODEL1_OMP_KERNELS_HPP
