#pragma once
#ifndef GBKFIT_UTILITY_HPP
#define GBKFIT_UTILITY_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {

namespace util_num
{

std::uint32_t roundu_po2(std::uint32_t num)
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

std::uint32_t roundd_po2 (std::uint32_t num)
{
    num = num | (num >> 1);
    num = num | (num >> 2);
    num = num | (num >> 4);
    num = num | (num >> 8);
    num = num | (num >> 16);
    return num - (num >> 1);
}

std::uint32_t roundu_multiple(std::uint32_t num, std::uint32_t multiple)
{
    return (num % multiple != 0) ? (num - num % multiple + multiple) : num;
}

std::uint32_t roundd_multiple(std::uint32_t num, std::uint32_t multiple)
{
    return (num / multiple) * multiple;
}

bool is_odd(int num)
{
    return num & 1;
}

bool is_even(int num)
{
    return !is_odd(num);
}

int roundd_even(float num)
{
    return 2.0 * std::floor(num * 0.5);
}

int roundu_even(float num)
{
    return 2.0 * std::ceil(num * 0.5);
}

int roundd_odd(float num)
{
    int num_even = roundd_even(num);
    return num_even + 1.0 > num ? num_even - 1.0 : num_even + 1.0;
}

int roundu_odd(float num)
{
    int num_even = roundu_even(num);
    return num_even - 1.0 < num ? num_even + 1.0 : num_even - 1.0;
}

} // namespace util_num

namespace util_fft
{

std::uint32_t calculate_optimal_dim_length(std::uint32_t length, std::uint32_t po2_length_max, std::uint32_t multiple)
{
    std::uint32_t length_new = util_num::roundu_po2(length);
    if (length_new > po2_length_max)
        length_new = util_num::roundu_multiple(length,multiple);
    return length_new;
}

} // namespace util_fft

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
