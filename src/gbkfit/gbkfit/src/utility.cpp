
#include "gbkfit/utility.hpp"

namespace gbkfit {

namespace util_image
{
    void image_shift(float* data, int size_x, int size_y, int size_z, int shift_x, int shift_y, int shift_z)
    {
        int size = size_x * size_y * size_z;

        float* data_tmp = new float[size];
        std::copy(data, data+size, data_tmp);

        for(int z = 0; z < size_z; ++z)
        {
            for(int y = 0; y < size_y; ++y)
            {
                for(int x = 0; x < size_x; ++x)
                {
                    int xidx = x + shift_x;
                    int yidx = y + shift_y;
                    int zidx = z + shift_z;

                    if (xidx < 0) xidx = size_x + xidx;
                    if (yidx < 0) yidx = size_y + yidx;
                    if (zidx < 0) zidx = size_z + zidx;
                    if (xidx >= size_x) xidx = xidx-size_x;
                    if (yidx >= size_y) yidx = yidx-size_y;
                    if (zidx >= size_z) zidx = zidx-size_z;

                    int idx_src = z*size_x*size_y + y*size_x + x;
                    int idx_dst = zidx*size_x*size_y + yidx*size_x + xidx;

                    data[idx_dst] = data_tmp[idx_src];
                }
            }
        }

        delete [] data_tmp;
    }

} // namespace util_image

} // namespace gbkfit
