
#include "gbkfit/params.hpp"

namespace gbkfit {

Param::Param(const std::string& name)
    : m_name(name)
{
}

Param::~Param()
{
}

const std::string& Param::get_name(void) const
{
    return m_name;
}

bool Params::has(const std::string& name) const
{
    return m_params.find(name) != m_params.end();
}

Param& Params::add(const std::string& name)
{
    if (has(name))
        throw std::runtime_error("parameter already exists: '" + name + "'");

    m_params.emplace(name, Param(name));
    return m_params.at(name);
}

void Params::del(const std::string& name)
{
    if (!has(name))
        throw std::runtime_error("parameter does not exist: '" + name + "'");

    m_params.erase(name);
}

Param& Params::get(const std::string& name)
{
    if (!has(name))
        throw std::runtime_error("parameter does not exist: '" + name + "'");

    return m_params.at(name);
}

const Param& Params::get(const std::string& name) const
{
    if (!has(name))
        throw std::runtime_error("parameter does not exist: '" + name + "'");

    return m_params.at(name);
}

Params::iterator Params::begin(void)
{
    return m_params.begin();
}

Params::iterator Params::end(void)
{
    return m_params.end();
}

Params::const_iterator Params::begin(void) const
{
    return m_params.begin();
}

Params::const_iterator Params::end(void) const
{
    return m_params.end();
}

} // namespace gbkfit
