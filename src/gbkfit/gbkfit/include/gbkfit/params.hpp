#pragma once
#ifndef GBKFIT_PARAMS_HPP
#define GBKFIT_PARAMS_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/variable_map.hpp"

namespace gbkfit {

class Param : public VariableMap
{

private:

    std::string m_name;

public:

    Param(const std::string& name);

    virtual ~Param();

    const std::string& get_name(void) const;

};

class Params
{

public:

    typedef std::map<std::string, Param>::iterator       iterator;
    typedef std::map<std::string, Param>::const_iterator const_iterator;

private:

    std::map<std::string, Param> m_params;

public:

    bool has(const std::string& name) const;

    Param& add(const std::string& name);

    void del(const std::string& name);

    Param& get(const std::string& name);

    const Param& get(const std::string& name) const;

    iterator begin(void);

    iterator end(void);

    const_iterator begin(void) const;

    const_iterator end(void) const;

    template<typename T>
    std::vector<T> get_array(const std::string& property) const
    {
        std::vector<T> array;
        for(auto& param : m_params)
        {
            array.push_back(param.second.get<T>(property));
        }
        return array;
    }

    template<typename T>
    std::map<std::string, T> get_map(const std::string& property) const
    {
        std::map<std::string, T> map;
        for(auto& param : m_params)
        {
            map.emplace(param.first, param.second.get<T>(property));
        }
        return map;
    }

};

} // namespace gbkfit

#endif // GBKFIT_PARAMS_HPP
