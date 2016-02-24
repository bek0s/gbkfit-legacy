#pragma once
#ifndef GBKFIT_PROPERTY_MAP_HPP
#define GBKFIT_PROPERTY_MAP_HPP

#include "gbkfit/prerequisites.hpp"

#include <boost/lexical_cast.hpp>

namespace gbkfit {

class PropertyMap
{

private:

    std::map<std::string, std::string> m_options;
    std::vector<std::string> m_keys;

public:

    PropertyMap(void)
    {
    }

    virtual ~PropertyMap()
    {
    }

    const std::vector<std::string>& keys(void) const
    {
        return m_keys;
    }

    bool has(const std::string& key) const
    {
        return m_options.count(key) > 0;
    }

    template<typename T>
    T get(const std::string& key)
    {
        if (!has(key))
            throw std::runtime_error(BOOST_CURRENT_FUNCTION);
        return boost::lexical_cast<T>(m_options[key]);
    }

    template<typename T>
    T get(const std::string& key, const T& dvalue)
    {
        if (!has(key))
            return dvalue;
        return boost::lexical_cast<T>(m_options[key]);
    }

    template<typename T>
    PropertyMap& add(const std::string& key, const T& value)
    {
        if (has(key))
            throw std::runtime_error(BOOST_CURRENT_FUNCTION);
        m_options[key] = boost::lexical_cast<std::string>(value);
        m_keys.push_back(key);
        return *this;
    }

    template<typename T>
    PropertyMap& set(const std::string& key, const T& value)
    {
        if (!has(key))
            throw std::runtime_error(BOOST_CURRENT_FUNCTION);
        m_options[key] = boost::lexical_cast<std::string>(value);
        return *this;
    }

    PropertyMap& remove(const std::string& key)
    {
        if (!has(key))
            throw std::runtime_error(BOOST_CURRENT_FUNCTION);
        m_options.erase(key);
        m_keys.erase(std::find(m_keys.begin(), m_keys.end(), key));
        return *this;
    }

}; // class PropertyMap

} // namespace gbkfit

#endif // GBKFIT_PROPERTY_MAP_HPP
