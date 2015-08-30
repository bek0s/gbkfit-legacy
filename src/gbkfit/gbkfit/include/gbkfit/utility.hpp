#pragma once
#ifndef GBKFIT_UTILITY_HPP
#define GBKFIT_UTILITY_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


template<typename TKey,typename TValue>
std::map<TKey,TValue> vectors_to_map(const std::vector<TKey>& keys, const std::vector<TValue>& values)
{
    if (keys.size() != values.size())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    std::map<TKey,TValue> map;
    std::transform(keys.begin(), keys.end(), values.begin(), std::inserter(map,map.begin()), [] (TKey key, TValue value) { return std::make_pair(key, value); });
    return map;
}


template<typename T>
std::string to_string(T value)
{
    std::ostringstream stream;
    stream << value;
    return stream.str();
}

template<typename Tkey, typename Tvalue>
std::string to_string(const std::map<Tkey, Tvalue>& map)
{
    std::ostringstream stream;
    stream << "std::map [";
    for(auto iter = map.begin(); iter != map.end(); ++iter)
    {
        stream << "(" << to_string((*iter).first) << ", " << to_string((*iter).second) << ")" << (std::next(iter) != map.end() ? ", " : "");
    }
    stream << "]";
    return stream.str();
}

} // namespace gbkfit

#endif // GBKFIT_UTILITY_HPP
