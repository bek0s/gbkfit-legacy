#pragma once
#ifndef GBKFIT_STRING_UTIL_HPP
#define GBKFIT_STRING_UTIL_HPP

#include <boost/lexical_cast.hpp>

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {
namespace string_util {

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

template<typename Target, typename Source>
Target lexical_cast(const Source& value)
{
    return boost::lexical_cast<Target>(value);
}

} // namespace string_util
} // namespace gbkfit

#endif // GBKFIT_STRING_UTIL_HPP
