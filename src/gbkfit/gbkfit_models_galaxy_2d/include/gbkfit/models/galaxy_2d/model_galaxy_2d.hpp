#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_HPP
#define GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_HPP

#include "gbkfit/model.hpp"

namespace gbkfit {
namespace models {
namespace galaxy_2d {

//!
//! \brief The model_galaxy_2d class
//!
class model_galaxy_2d : public model
{

//  static const std::map<std::string,std::vector<std::string>> m_

//  static const std::vector<std::vector<std::string>> ms_rcur_arctan;

public:

    /*
    enum class profile_flux_type : int {
        exponential = 1
    };

    enum class profile_rcur_type : int {
        lramp = 1,
        arctan = 2,
        epinat = 3,
    };
    */

    enum profile_flux_type {
        exponential = 1
    };

    enum profile_rcur_type {
        lramp = 1,
        arctan = 2,
        epinat = 3,
    };

protected:

    int m_model_size_x;
    int m_model_size_y;
    int m_model_size_z;
    float m_step_x;
    float m_step_y;
    float m_step_z;
    int m_upsampling_x;
    int m_upsampling_y;
    int m_upsampling_z;
    int m_model_size_x_aligned;
    int m_model_size_y_aligned;
    int m_model_size_z_aligned;

    profile_flux_type m_profile_flux;
    profile_rcur_type m_profile_rcur;

    std::vector<std::string> m_parameter_names;
    std::vector<float> m_parameter_values;

public:

    model_galaxy_2d(int size_x, int size_y, float step_x, float step_y, int upsampling_x, int upsampling_y,
                    profile_flux_type profile_flux, profile_rcur_type profile_rcur_type);

    ~model_galaxy_2d();

    const std::vector<std::string>& get_parameter_names(void) const final;

    const std::vector<float>& get_parameter_values(void) const final;

    const std::map<std::string,ndarray*>& evaluate(const std::map<std::string,float>& parameters) final;

private:

    virtual const std::map<std::string,ndarray*>& evaluate(int model_flux_id,
                                                           int model_rcur_id,
                                                           const float parameter_vsys,
                                                           const std::vector<float>& parameters_proj,
                                                           const std::vector<float>& parameters_flux,
                                                           const std::vector<float>& parameters_rcur,
                                                           const std::vector<float>& parameters_vsig) = 0;

}; // class model_galaxy_2d


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_HPP
