
#include "gbkfit/models/galaxy_2d/model_galaxy_2d.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {
namespace models {
namespace galaxy_2d {


/*
const std::vector<std::vector<std::string>> model_galaxy_2d::ms_rcur_arctan = {{"1","2"}
                                                                              ,{"3","$"}};
                                                                              */

model_galaxy_2d::model_galaxy_2d(int size_x, int size_y, float step_x, float step_y, int upsampling_x, int upsampling_y,
                                 profile_flux_type profile_flux, profile_rcur_type profile_rcur)
    : m_model_size_x(size_x)
    , m_model_size_y(size_y)
    , m_model_size_z(1)
    , m_step_x(step_x)
    , m_step_y(step_y)
    , m_step_z(1)
    , m_upsampling_x(upsampling_x)
    , m_upsampling_y(upsampling_y)
    , m_upsampling_z(1)
{
    m_model_size_x_aligned = m_model_size_x * m_upsampling_x;
    m_model_size_y_aligned = m_model_size_y * m_upsampling_y;
    m_model_size_z_aligned = m_model_size_z * m_upsampling_z;

    m_profile_flux = profile_flux;
    m_profile_rcur = profile_rcur;
}

model_galaxy_2d::~model_galaxy_2d()
{
}

const std::vector<std::string>& model_galaxy_2d::get_parameter_names(void) const
{
    return m_parameter_names;
}

const std::vector<float>& model_galaxy_2d::get_parameter_values(void) const
{
    return m_parameter_values;
}

const std::map<std::string,ndarray*>& model_galaxy_2d::evaluate(const std::map<std::string,float>& parameters)
{
    // Projection parameters.
    std::vector<float> params_proj;
    params_proj.push_back(parameters.at("xo"));
    params_proj.push_back(parameters.at("yo"));
    params_proj.push_back(parameters.at("pa"));
    params_proj.push_back(parameters.at("incl"));

    // Flux model parameters.
    std::vector<float> params_flux;
    if(m_profile_flux == exponential) {
        params_flux.push_back(parameters.at("i0"));
        params_flux.push_back(parameters.at("r0"));
    }

    // Rotation curve model parameters.
    std::vector<float> params_rcur;
    if(m_profile_rcur == arctan) {
        params_rcur.push_back(parameters.at("rt"));
        params_rcur.push_back(parameters.at("vt"));
    }

    // Velocity dispersion model parameters
    std::vector<float> params_vsig;
    params_vsig.push_back(parameters.at("vsig"));

    // velocity offset parameter
    float param_vsys = parameters.at("vsys");

    // ...
    return evaluate(1,1,param_vsys,params_proj,params_flux,params_rcur,params_vsig);
}


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
