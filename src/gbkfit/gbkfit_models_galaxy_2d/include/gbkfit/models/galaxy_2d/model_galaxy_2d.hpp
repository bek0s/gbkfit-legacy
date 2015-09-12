#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_HPP
#define GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_HPP

#include "gbkfit/model.hpp"

namespace gbkfit {
namespace models {
namespace galaxy_2d {

enum profile_flx_type {
    exponential = 1
};

enum profile_vel_type {
    lramp = 1,
    arctan = 2,
    epinat = 3,
};

class model_galaxy_2d : public model
{

protected:


    float m_step_x;
    float m_step_y;
    float m_step_z;

    profile_flx_type m_profile_flx;
    profile_vel_type m_profile_vel;

    std::vector<std::string> m_parameter_names;
    std::vector<float> m_parameter_values;

public:

    model_galaxy_2d(profile_flx_type profile_flx, profile_vel_type profile_vel);

    ~model_galaxy_2d();

    const std::vector<std::string>& get_parameter_names(void) const final;

    const std::vector<float>& get_parameter_values(void) const final;

    const std::map<std::string,ndarray*>& evaluate(const std::map<std::string,float>& parameters) final;

private:

    virtual const std::map<std::string,ndarray*>& evaluate(int profile_flx_id,
                                                           int profile_vel_id,
                                                           const float param_vsig,
                                                           const float param_vsys,
                                                           const std::vector<float>& params_prj,
                                                           const std::vector<float>& params_flx,
                                                           const std::vector<float>& params_vel) = 0;

}; // class model_galaxy_2d


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_HPP
