#pragma once
#ifndef GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_HPP
#define GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_HPP

#include "gbkfit/gmodel/gmodel01/gmodel01.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel01 {

class GModel01Cuda : public GModel01
{

public:

    GModel01Cuda(FlxProfileType flx_profile, VelProfileType vel_profile);

    ~GModel01Cuda() {}

    const std::string& get_type(void) const override final;

    void evaluate(const std::vector<float>& params_prj,
                  const std::vector<float>& params_flx,
                  const std::vector<float>& params_vel,
                  float param_vsys,
                  float param_vsig,
                  const std::vector<float>& data_zero,
                  const std::vector<float>& data_step,
                  NDArray* data) const override final;

};

} // namespace gmodel01
} // namespace gmodel
} // namespace gbkfit

#endif // GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_HPP
