
#include "gbkfit/image_util.hpp"
#include <cstring>

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
                      unsigned int depth)
    {
        for(unsigned int z = 0; z < depth; ++z) {
            for(unsigned int y = 0; y < height; ++y) {
                for(unsigned int x = 0; x < width; ++x) {
                    out_data_padded[z*height_padded*width_padded+y*width_padded+x] = data[z*height*width+y*width+x];
                }
            }
        }
    }

    void img_pad_fill_value(float* out_data_padded,
                            unsigned int width_padded,
                            unsigned int height_padded,
                            unsigned int depth_padded,
                            unsigned int width,
                            unsigned int height,
                            unsigned int depth,
                            float value)
    {
        for(unsigned int z = 0; z < depth_padded; ++z) {
            for(unsigned int y = 0; y < height_padded; ++y) {
                for(unsigned int x = 0; x < width_padded; ++x) {
                    if(z >= depth || y >= height || x >= width)
                        out_data_padded[z*height_padded*width_padded+y*width_padded+x] = value;
                }
            }
        }
    }

    void img_pad_fill_edge(float* out_data_padded,
                           const float* data,
                           unsigned int width_padded,
                           unsigned int height_padded,
                           unsigned int depth_padded,
                           unsigned int width,
                           unsigned int height,
                           unsigned int depth)
    {
    }

    void img_pad_fill_mirror(float* out_data_padded,
                             const float* data,
                             unsigned int width_padded,
                             unsigned int height_padded,
                             unsigned int depth_padded,
                             unsigned int width,
                             unsigned int height,
                             unsigned int depth)
    {

    }

    void img_circular_shift(float* inout_data,
                            unsigned int width,
                            unsigned int height,
                            unsigned int depth,
                            int x_shift,
                            int y_shift,
                            int z_shift)
    {
        float* data_tmp = new float[width*height*depth];
        memcpy(data_tmp,inout_data,width*height*depth*sizeof(float));

        //  TODO: handle over and under flows

        for(int z = 0; z < (int)depth; ++z)
        {
            for(int y = 0; y < (int)height; ++y)
            {
                for(int x = 0; x < (int)width; ++x)
                {
                    int xidx = x + x_shift;
                    int yidx = y + y_shift;
                    int zidx = z + z_shift;
                    if(xidx < 0) xidx = (int)width  + xidx;
                    if(yidx < 0) yidx = (int)height + yidx;
                    if(zidx < 0) zidx = (int)depth  + zidx;
                    if(xidx >= (int)width)  xidx = (xidx-(int)width);
                    if(yidx >= (int)height) yidx = (yidx-(int)height);
                    if(zidx >= (int)depth)  zidx = (zidx-(int)depth);
                    inout_data[zidx*height*width+yidx*width+xidx] = data_tmp[z*height*width+y*width+x];
                }
            }
        }

        delete [] data_tmp;
    }

}   //  namespace image_util
}   //  namespace gbkfit
