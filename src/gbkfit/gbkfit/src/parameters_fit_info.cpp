
#include "gbkfit/parameters_fit_info.hpp"

namespace gbkfit {

parameters_fit_info::parameter_fit_info& parameters_fit_info::add_parameter(const std::string& name)
{
    if(m_parameters.find(name) != m_parameters.end())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return m_parameters[name];
}

parameters_fit_info::parameter_fit_info& parameters_fit_info::get_parameter(const std::string& name)
{
    if(m_parameters.find(name) == m_parameters.end())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return m_parameters[name];
}

} // namespace gbkfit
