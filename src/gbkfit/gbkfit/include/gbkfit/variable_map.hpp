#pragma once
#ifndef GBKFIT_VARIABLE_MAP_HPP
#define GBKFIT_VARIABLE_MAP_HPP

#include "gbkfit/prerequisites.hpp"
#include "gbkfit/string_util.hpp"

namespace gbkfit {

class VariableMap
{

private:

    std::map<std::string, std::string> m_items;

public:

    VariableMap(void) {}

    virtual ~VariableMap() {}

    std::vector<std::string> keys(void) const
    {
        std::vector<std::string> keys;
        for(const auto& item : m_items)
            keys.push_back(item.first);
        return keys;
    }

    bool has(const std::string& key) const
    {
        return m_items.find(key) != m_items.end();
    }

    template<typename T>
    VariableMap& add(const std::string& key, const T& value)
    {
        if (has(key))
            throw std::runtime_error("key already exists: '" + key + "'");

        m_items[key] = string_util::lexical_cast<std::string>(value);
        return *this;
    }

    template<typename T>
    VariableMap& set(const std::string& key, const T& value)
    {
        if (!has(key))
            throw std::runtime_error("key does not exist: '" + key + "'");

        m_items[key] = string_util::lexical_cast<std::string>(value);
        return *this;
    }

    VariableMap& del(const std::string& key)
    {
        if (!has(key))
            throw std::runtime_error("key does not exist: '" + key + "'");

        m_items.erase(key);
        return *this;
    }

    template<typename T>
    T get(const std::string& key) const
    {
        if (!has(key))
            throw std::runtime_error("key does not exist: '" + key + "'");

        return string_util::lexical_cast<T>(m_items.at(key));
    }

    template<typename T>
    T get(const std::string& key, const T& dvalue) const
    {
        if (!has(key))
            return dvalue;

        return string_util::lexical_cast<T>(m_items.at(key));
    }
};

} // namespace gbkfit

#endif // GBKFIT_VARIABLE_MAP_HPP
