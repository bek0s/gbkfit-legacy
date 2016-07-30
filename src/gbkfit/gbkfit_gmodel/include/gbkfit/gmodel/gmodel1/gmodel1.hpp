#pragma once
#ifndef GBKFIT_GMODEL_GMODEL1_GMODEL1_HPP
#define GBKFIT_GMODEL_GMODEL1_GMODEL1_HPP

#include "gbkfit/gmodel.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {

enum FlxProfileType {
    exponential = 1
};

enum VelProfileType {
    lramp = 1,
    boissier = 2,
    arctan = 3,
    epinat = 4
};

class GModel1 : public GModel
{

protected:

    FlxProfileType m_flx_profile;
    VelProfileType m_vel_profile;
    std::vector<std::string> m_param_names;

public:

    GModel1(FlxProfileType flx_profile, VelProfileType vel_profile);

    ~GModel1();

    const std::vector<std::string>& get_param_names(void) const override final;

    void evaluate(const std::map<std::string, float>& params,
                  const std::vector<float>& data_zero,
                  const std::vector<float>& data_step,
                  NDArray* data) const override final;

private:

    virtual void evaluate(const std::vector<float>& params_prj,
                          const std::vector<float>& params_flx,
                          const std::vector<float>& params_vel,
                          float param_vsys,
                          float param_vsig,
                          const std::vector<float>& data_zero,
                          const std::vector<float>& data_step,
                          NDArray* data) const = 0;

};

} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit

#endif // GBKFIT_GMODEL_GMODEL1_GMODEL1_HPP
