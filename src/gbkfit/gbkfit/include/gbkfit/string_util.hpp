#pragma once
#ifndef GBKFIT_STRING_UTIL_HPP
#define GBKFIT_STRING_UTIL_HPP

#include "gbkfit/prerequisites.hpp"

#include <boost/lexical_cast.hpp>


namespace gbkfit {
namespace string_util {

template<typename T>
T parse(const std::string& value)
{
    return boost::lexical_cast<T>(value);
}

template<typename T>
T parse(const std::string& value, T dvalue)
{
    T pvalue;

    try
    {
        pvalue = string_util::parse<T>(value);
    }
    catch(std::exception&)
    {
        pvalue = dvalue;
    }

    return pvalue;
}

} // namespace string_util
} // namespace gbkfit

#endif // GBKFIT_STRING_UTIL_HPP
