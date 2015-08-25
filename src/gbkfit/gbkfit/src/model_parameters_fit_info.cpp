
#include "gbkfit/model_parameters_fit_info.hpp"

namespace gbkfit {


model_parameter_fit_info& model_parameters_fit_info::add_parameter(const std::string& name)
{
    if(m_parameters.find(name) != m_parameters.end())
        throw std::runtime_error("parameter already exists");
    return m_parameters[name] = model_parameter_fit_info();

}

model_parameter_fit_info& model_parameters_fit_info::get_parameter(const std::string& name)
{
    if(m_parameters.find(name) == m_parameters.end())
        throw std::runtime_error("parameter does not exist");
    return m_parameters[name];
}


} // namespace gbkfit
