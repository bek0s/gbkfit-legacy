#pragma once
#ifndef GBKFIT_MODELS_MODEL01_MODEL_MODEL01_HPP
#define GBKFIT_MODELS_MODEL01_MODEL_MODEL01_HPP

#include "gbkfit/model.hpp"

namespace gbkfit {
namespace models {
namespace model01 {

enum profile_flx_type {
    exponential = 1
};

enum profile_vel_type {
    lramp = 1,
    david = 2,
    arctan = 3,
    epinat = 4,
};

//!
//! \brief The ModelModel01 class
//!
class ModelModel01 : public Model
{

protected:

    profile_flx_type m_profile_flx;
    profile_vel_type m_profile_vel;

    std::vector<std::string> m_parameter_names;
    std::vector<float> m_parameter_values;

public:

    ModelModel01(profile_flx_type profile_flx, profile_vel_type profile_vel);

    ~ModelModel01();

    const std::vector<std::string>& get_parameter_names(void) const final;

    const std::vector<float>& get_parameter_values(void) const final;

    const std::map<std::string,NDArray*>& evaluate(const std::map<std::string,float>& parameters) final;

private:

    virtual const std::map<std::string,NDArray*>& evaluate(int profile_flx_id,
                                                           int profile_vel_id,
                                                           const std::vector<float>& params_prj,
                                                           const std::vector<float>& params_flx,
                                                           const std::vector<float>& params_vel,
                                                           const float param_vsys,
                                                           const float patam_vsig) = 0;

}; // class ModelModel01

//!
//! \brief The ModelFactoryModel01 class
//!
class ModelFactoryModel01 : public ModelFactory
{

public:

    ModelFactoryModel01(void);

    ~ModelFactoryModel01();

}; // class ModelFactoryModel01

} // namespace model01
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_MODEL01_MODEL_MODEL01_HPP
