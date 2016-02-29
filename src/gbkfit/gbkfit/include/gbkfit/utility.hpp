#pragma once
#ifndef GBKFIT_UTILITY_HPP
#define GBKFIT_UTILITY_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {


namespace util_num
{

inline std::uint32_t roundu_po2(std::uint32_t num)
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

inline std::uint32_t roundd_po2 (std::uint32_t num)
{
    num = num | (num >> 1);
    num = num | (num >> 2);
    num = num | (num >> 4);
    num = num | (num >> 8);
    num = num | (num >> 16);
    return num - (num >> 1);
}

inline std::uint32_t roundu_multiple(std::uint32_t num, std::uint32_t multiple)
{
    return (num % multiple != 0) ? (num - num % multiple + multiple) : num;
}

inline std::uint32_t roundd_multiple(std::uint32_t num, std::uint32_t multiple)
{
    return (num / multiple) * multiple;
}

inline bool is_odd(int num)
{
    return num & 1;
}

inline bool is_even(int num)
{
    return !is_odd(num);
}

inline int roundd_even(float num)
{
    return 2.0 * std::floor(num * 0.5);
}

inline int roundu_even(float num)
{
    return 2.0 * std::ceil(num * 0.5);
}

inline int roundd_odd(float num)
{
    int num_even = roundd_even(num);
    return num_even + 1.0 > num ? num_even - 1.0 : num_even + 1.0;
}

inline int roundu_odd(float num)
{
    int num_even = roundu_even(num);
    return num_even - 1.0 < num ? num_even + 1.0 : num_even - 1.0;
}

} // namespace util_num

namespace util_fft
{

inline int calculate_optimal_dim_length(std::uint32_t length, std::uint32_t po2_length_max, std::uint32_t multiple)
{
    std::uint32_t length_new = util_num::roundu_po2(length);
    if (length_new > po2_length_max)
        length_new = util_num::roundu_multiple(length,multiple);
    return static_cast<int>(length_new);
}

} // namespace util_fft

namespace util_image
{
    void image_shift(float* data, int size_x, int size_y, int size_z, int shift_x, int shift_y, int shift_z);

    void image_copy_padded(const float* src, int size_x, int size_y, int size_z, int size_x_padded, int size_y_padded, int size_z_padded, float* dst);

    void image_fill_padded(float value, int size_x, int size_y, int size_z, int size_x_padded, int size_y_padded, int size_z_padded, float* data);

    void image_flip_2d(float* data, int size_x, int size_y);

} // namespace util_image

template<typename TKey,typename TValue>
std::map<TKey,TValue> vectors_to_map(const std::vector<TKey>& keys, const std::vector<TValue>& values)
{
    if (keys.size() != values.size())
        throw std::runtime_error(BOOST_CURRENT_FUNCTION);

    std::map<TKey,TValue> map;
    std::transform(keys.begin(), keys.end(), values.begin(), std::inserter(map,map.begin()), [] (TKey key, TValue value) { return std::make_pair(key, value); });
    return map;
}

} // namespace gbkfit

#endif // GBKFIT_UTILITY_HPP
