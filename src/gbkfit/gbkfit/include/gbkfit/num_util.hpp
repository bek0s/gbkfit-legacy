#pragma once
#ifndef GBKFIT_NUM_UTIL_HPP
#define GBKFIT_NUM_UTIL_HPP

#include <cassert>

namespace gbkfit {
namespace num_util {

template<typename T>
T roundup_multiple(T value, T multiple)
{
    assert(multiple >= 0);

    return multiple == 0 ? value : static_cast<T>(std::ceil(static_cast<double>(value)/static_cast<double>(multiple))*static_cast<double>(multiple));
}

unsigned int roundup_po2(unsigned int num)
{
    num--;
    num |= num >> 1;
    num |= num >> 2;
    num |= num >> 4;
    num |= num >> 8;
    num |= num >> 16;
    num++;
    return num;
}

int roundup_to_odd(float n)
{
    int num = static_cast<int>(std::ceil(n));
    if(num % 2 == 0)
        num++;
    return num;
}

} // namespace num_util
} // namespace gbkfit

#endif // GBKFIT_NUM_UTIL_HPP
