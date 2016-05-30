
#include "gbkfit/parameters.hpp"

namespace gbkfit {

Parameter::Parameter(const std::string& name)
    : m_name(name)
{
}

Parameter::~Parameter()
{
}

const std::string& Parameter::get_name(void) const
{
    return m_name;
}

bool Parameters::has(const std::string& name) const
{
    return m_params.find(name) != m_params.end();
}

Parameter& Parameters::add(const std::string& name)
{
    if (has(name))
        throw std::runtime_error("parameter already exists: '" + name + "'");

    m_params.emplace(name, Parameter(name));
    return m_params.at(name);
}

void Parameters::del(const std::string& name)
{
    if (!has(name))
        throw std::runtime_error("parameter does not exist: '" + name + "'");

    m_params.erase(name);
}

Parameter& Parameters::get(const std::string& name)
{
    if (!has(name))
        throw std::runtime_error("parameter does not exist: '" + name + "'");

    return m_params.at(name);
}

const Parameter& Parameters::get(const std::string& name) const
{
    if (!has(name))
        throw std::runtime_error("parameter does not exist: '" + name + "'");

    return m_params.at(name);
}

Parameters::iterator Parameters::begin(void)
{
    return m_params.begin();
}

Parameters::iterator Parameters::end(void)
{
    return m_params.end();
}

Parameters::const_iterator Parameters::begin(void) const
{
    return m_params.begin();
}

Parameters::const_iterator Parameters::end(void) const
{
    return m_params.end();
}

} // namespace gbkfit
