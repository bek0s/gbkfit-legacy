#pragma once
#ifndef GKFIT_IMAGE_UTIL_HPP
#define GKFIT_IMAGE_UTIL_HPP

namespace gbkfit
{
namespace image_util
{

    void img_pad_copy(float* out_data_padded,
                      const float* data,
                      unsigned int width_padded,
                      unsigned int height_padded,
                      unsigned int depth_padded,
                      unsigned int width,
                      unsigned int height,
                      unsigned int depth);

    void img_pad_fill_value(float* out_data_padded,
                            unsigned int width_padded,
                            unsigned int height_padded,
                            unsigned int depth_padded,
                            unsigned int width,
                            unsigned int height,
                            unsigned int depth,
                            float value);

    void img_pad_fill_edge(float* out_data_padded,
                           const float* data,
                           unsigned int width_padded,
                           unsigned int height_padded,
                           unsigned int depth_padded,
                           unsigned int width,
                           unsigned int height,
                           unsigned int depth);

    void img_pad_fill_mirror(float* out_data_padded,
                             const float* data,
                             unsigned int width_padded,
                             unsigned int height_padded,
                             unsigned int depth_padded,
                             unsigned int width,
                             unsigned int height,
                             unsigned int depth);

    void img_circular_shift(float* inout_data,
                            unsigned int width,
                            unsigned int height,
                            unsigned int depth,
                            int x_shift,
                            int y_shift,
                            int z_shift);

}   //  namespace image_util
}   //  namespace gbkfit

#endif  //  GKFIT_IMAGE_UTIL_HPP
