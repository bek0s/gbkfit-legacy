#pragma once
#ifndef GBKFIT_GMODEL_GMODEL1_GMODEL1_CUDA_HPP
#define GBKFIT_GMODEL_GMODEL1_GMODEL1_CUDA_HPP

#include "gbkfit/gmodel/gmodel1/gmodel1.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {

class GModel1Cuda : public GModel1
{

public:

    GModel1Cuda(FlxProfileType flx_profile, VelProfileType vel_profile);

    ~GModel1Cuda() {}

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

} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit

#endif // GBKFIT_GMODEL_GMODEL1_GMODEL1_CUDA_HPP
