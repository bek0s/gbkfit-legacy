#pragma once
#ifndef GBKFIT_PARAMETERS_HPP
#define GBKFIT_PARAMETERS_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/variable_map.hpp"

namespace gbkfit {

class Parameter : public VariableMap
{

private:

    std::string m_name;

public:

    Parameter(const std::string& name);

    virtual ~Parameter();

    const std::string& get_name(void) const;

};

class Parameters
{

public:

    typedef std::map<std::string, Parameter>::iterator       iterator;
    typedef std::map<std::string, Parameter>::const_iterator const_iterator;

private:

    std::map<std::string, Parameter> m_params;

public:

    bool has(const std::string& name) const;

    Parameter& add(const std::string& name);

    void del(const std::string& name);

    Parameter& get(const std::string& name);

    const Parameter& get(const std::string& name) const;

    iterator begin(void);

    iterator end(void);

    const_iterator begin(void) const;

    const_iterator end(void) const;

};

} // namespace gbkfit

#endif // GBKFIT_PARAMETERS_HPP
