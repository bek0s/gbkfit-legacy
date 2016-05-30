#pragma once
#ifndef GBKFIT_ARRAY_UTIL_HPP
#define GBKFIT_ARRAY_UTIL_HPP

#include "gbkfit/prerequisites.hpp"

namespace gbkfit {
namespace array_util {

void array_shift(int size_x,
                 int size_y,
                 int size_z,
                 int shift_x,
                 int shift_y,
                 int shift_z,
                 float* data);

void array_copy(int size_x,
                int size_y,
                int size_z,
                int size_x_padded,
                int size_y_padded,
                int size_z_padded,
                const float* src,
                float* dst);

void array_fill(float value,
                int size_x,
                int size_y,
                int size_z,
                int size_x_padded,
                int size_y_padded,
                int size_z_padded,
                float* data);

void array_flip(int size_x, int size_y, int size_z, float* data);

} // namespace array_util
} // namespace gbkfit

#endif // GBKFIT_ARRAY_UTIL_HPP
