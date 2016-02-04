
#include "gbkfit/parameters.hpp"

namespace gbkfit {

Parameters::Parameter& Parameters::add_parameter(const std::string& name)
{
    if(m_parameters.find(name) != m_parameters.end())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return m_parameters[name];
}

Parameters::Parameter& Parameters::get_parameter(const std::string& name)
{
    if(m_parameters.find(name) == m_parameters.end())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);
    return m_parameters[name];
}

} // namespace gbkfit
