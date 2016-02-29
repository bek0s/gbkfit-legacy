
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

    void image_copy_padded(const float* src, int size_x, int size_y, int size_z, int size_x_padded, int size_y_padded, int size_z_padded, float* dst)
    {
        (void)size_z_padded;

        for(int z = 0; z < size_z; ++z) {
            for(int y = 0; y < size_y; ++y) {
                for(int x = 0; x < size_x; ++x) {
                    int idx_src = z*size_x*size_y + y*size_x + x;
                    int idx_dst = z*size_x_padded*size_y_padded + y*size_x_padded + x;
                    dst[idx_dst] = src[idx_src];
                }
            }
        }
    }

    void image_fill_padded(float value, int size_x, int size_y, int size_z, int size_x_padded, int size_y_padded, int size_z_padded, float* data)
    {
        for(int z = 0; z < size_z_padded; ++z) {
            for(int y = 0; y < size_y_padded; ++y) {
                for(int x = 0; x < size_x_padded; ++x) {
                    if(z >= size_z || y >= size_y || x >= size_x) {
                        int idx = z*size_x_padded*size_y_padded + y*size_x_padded + x;
                        data[idx] = value;
                    }
                }
            }
        }
    }

    void image_flip_2d(float* data, int size_x, int size_y)
    {
        for(int y = 0; y < size_y/2; ++y)
        {
            int ya = y;
            int yb = size_y - ya - 1;

            for (int x = 0; x < size_x/2; ++x)
            {
                int xa = x;
                int xb = size_x - xa - 1;

                int idxa = ya*size_x + xa;
                int idxb = yb*size_x + xb;

                float tmp;
                tmp = data[idxa];
                data[idxa] = data[idxb];
                data[idxb] = tmp;
            }
        }
    }

} // namespace util_image

} // namespace gbkfit
