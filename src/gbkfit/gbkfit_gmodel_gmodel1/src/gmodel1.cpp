
#include "gbkfit/gmodel/gmodel1/gmodel1.hpp"

#include "gbkfit/ndarray.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {

GModel1::GModel1(FlxProfileType flx_profile, VelProfileType vel_profile)
    : m_flx_profile(flx_profile)
    , m_vel_profile(vel_profile)
{
    // Projection parameters
    m_param_names.push_back("xo");
    m_param_names.push_back("yo");
    m_param_names.push_back("pa");
    m_param_names.push_back("incl");

    // Flux profile parameters
    switch (m_flx_profile) {
        case exponential: {
            m_param_names.push_back("i0");
            m_param_names.push_back("r0");
            break;
        }
    }

    // Velocity profile parameters
    switch (m_vel_profile) {
        case lramp: {
            m_param_names.push_back("rt");
            m_param_names.push_back("vt");
            break;
        }
        case boissier: {
            m_param_names.push_back("rt");
            m_param_names.push_back("vt");
            break;
        }
        case arctan: {
            m_param_names.push_back("rt");
            m_param_names.push_back("vt");
            break;
        }
        case epinat: {
            m_param_names.push_back("rt");
            m_param_names.push_back("vt");
            m_param_names.push_back("a");
            m_param_names.push_back("b");
            break;
        }
    }

    // Systemic velocity parameter
    m_param_names.push_back("vsys");

    // Velocity dispersion model parameter
    m_param_names.push_back("vsig");
}

GModel1::~GModel1()
{
}

const std::vector<std::string>& GModel1::get_param_names(void) const
{
    return m_param_names;
}

void GModel1::evaluate(const std::map<std::string, float>& params,
                        const std::vector<float>& data_zero,
                        const std::vector<float>& data_step,
                        NDArray* data) const
{
    // Make sure the data storage has 3 dimensions
    if (data->get_shape().get_dim_count() != 3) {
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    }

    // Projection parameters
    std::vector<float> params_prj;
    params_prj.push_back(params.at("xo"));
    params_prj.push_back(params.at("yo"));
    params_prj.push_back(params.at("pa"));
    params_prj.push_back(params.at("incl"));

    // Flux profile parameters
    std::vector<float> params_flx;
    switch(m_flx_profile) {
        case exponential: {
            params_flx.push_back(params.at("i0"));
            params_flx.push_back(params.at("r0"));
            break;
        }
    }

    // Velocity profile parameters
    std::vector<float> params_vel;
    switch(m_vel_profile) {
        case lramp: {
            params_vel.push_back(params.at("rt"));
            params_vel.push_back(params.at("vt"));
            break;
        }
        case boissier: {
            params_vel.push_back(params.at("rt"));
            params_vel.push_back(params.at("vt"));
            break;
        }
        case arctan: {
            params_vel.push_back(params.at("rt"));
            params_vel.push_back(params.at("vt"));
            break;
        }
        case epinat: {
            params_vel.push_back(params.at("rt"));
            params_vel.push_back(params.at("vt"));
            params_vel.push_back(params.at("a"));
            params_vel.push_back(params.at("b"));
            break;
        }
    }

    // Systemic velocity parameter
    float param_vsys = params.at("vsys");

    // Velocity dispersion model parameter
    float param_vsig = params.at("vsig");

    // Evaluate the damn thing already :)
    evaluate(params_prj,
             params_flx,
             params_vel,
             param_vsys,
             param_vsig,
             data_zero,
             data_step,
             data);
}

} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit
