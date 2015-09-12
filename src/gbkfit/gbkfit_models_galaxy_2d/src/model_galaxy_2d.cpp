
#include "gbkfit/models/galaxy_2d/model_galaxy_2d.hpp"

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

namespace gbkfit {
namespace models {
namespace galaxy_2d {


model_galaxy_2d::model_galaxy_2d(profile_flx_type profile_flx, profile_vel_type profile_vel)
    : m_profile_flx(profile_flx)
    , m_profile_vel(profile_vel)
{
    //
    // Projection parameters.
    //

    m_parameter_names.push_back("xo");
    m_parameter_names.push_back("yo");
    m_parameter_names.push_back("pa");
    m_parameter_names.push_back("incl");

    //
    // Flux model parameters.
    //

    if (m_profile_flx == exponential) {
        m_parameter_names.push_back("i0");
        m_parameter_names.push_back("r0");
    }

    //
    // Rotation curve model parameters.
    //

    if (m_profile_vel == lramp) {
        m_parameter_names.push_back("rt");
        m_parameter_names.push_back("vt");
    }
    else if (m_profile_vel == arctan) {
        m_parameter_names.push_back("rt");
        m_parameter_names.push_back("vt");
    }
    else if (m_profile_vel == epinat) {
        m_parameter_names.push_back("rt");
        m_parameter_names.push_back("vt");
        m_parameter_names.push_back("a");
        m_parameter_names.push_back("b");
    }

    //
    // Velocity dispersion model parameters.
    //

    m_parameter_names.push_back("vsig");

    //
    // Systemic velocity parameter.
    //

    m_parameter_names.push_back("vsys");
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
    //
    // Projection parameters.
    //

    std::vector<float> params_prj;
    params_prj.push_back(parameters.at("xo"));
    params_prj.push_back(parameters.at("yo"));
    params_prj.push_back(parameters.at("pa"));
    params_prj.push_back(parameters.at("incl"));

    //
    // Flux model parameters.
    //

    std::vector<float> params_flx;
    if (m_profile_flx == exponential) {
        params_flx.push_back(parameters.at("i0"));
        params_flx.push_back(parameters.at("r0"));
    }

    //
    // Rotation curve model parameters.
    //

    std::vector<float> params_vel;
    if (m_profile_vel == lramp) {
        params_vel.push_back(parameters.at("rt"));
        params_vel.push_back(parameters.at("vt"));
    }
    else if (m_profile_vel == arctan) {
        params_vel.push_back(parameters.at("rt"));
        params_vel.push_back(parameters.at("vt"));
    }
    else if (m_profile_vel == epinat) {
        params_vel.push_back(parameters.at("rt"));
        params_vel.push_back(parameters.at("vt"));
        params_vel.push_back(parameters.at("a"));
        params_vel.push_back(parameters.at("b"));
    }

    //
    // Velocity dispersion model parameters.
    //

    float param_vsig = parameters.at("vsig");

    //
    // Systemic velocity parameter.
    //

    float param_vsys = parameters.at("vsys");

    // ...
    return evaluate(m_profile_flx,m_profile_vel,param_vsig,param_vsys,params_prj,params_flx,params_vel);
}


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
