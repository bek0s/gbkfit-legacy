#pragma once
#ifndef GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_KERNELS_D_CUH
#define GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_KERNELS_D_CUH

#include <cufft.h>

namespace gbkfit {
namespace dmodel {
namespace scube {
namespace kernels_cuda_d
{
__device__
void map_index_1d_to_3d(int& out_xidx,
                        int& out_yidx,
                        int& out_zidx,
                        int idx,
                        int width,
                        int height,
                        int depth)
{
    out_zidx = idx/(width*height);
    idx -= out_zidx*width*height;

    out_yidx = idx/width;
    idx -= out_yidx*width;

    out_xidx = idx/1;
}

__global__
void fill_cube(int size_x,
               int size_y,
               int size_z,
               float value,
               float* data)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int grid_size = gridDim.x*blockDim.x;

    const int size = size_x*size_y*size_z;

    int x, y, z;

    while(idx < size)
    {
        // Calculate cube indices.
        map_index_1d_to_3d(x, y, z, idx, size_x, size_y, size_z);

        data[z*size_x*size_y + y*size_x + x] = value;

        idx += grid_size;
    }
}

__global__
void array_complex_multiply_and_scale(cufftComplex* array1,
                                      cufftComplex* array2,
                                      int length, float scale)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop until there are no elements left.
    while(idx < length)
    {
        cufftComplex a,b,t;

        a.x = array1[idx].x;
        a.y = array1[idx].y;

        b.x = array2[idx%length].x;
        b.y = array2[idx%length].y;

        // Perform complex multiplication and scalar scale.
        t.x = (a.x*b.x-a.y*b.y)*scale;
        t.y = (a.x*b.y+a.y*b.x)*scale;

        // Store result.
        array1[idx].x = t.x;
        array1[idx].y = t.y;

        // Proceed to the next element.
        idx += gridDim.x * blockDim.x;
    }
}


__global__
void downsample_cube(int size_x,
                     int size_y,
                     int size_z,
                     int size_up_x,
                     int size_up_y,
                     int size_up_z,
                     int offset_x,
                     int offset_y,
                     int offset_z,
                     int downsample_x,
                     int downsample_y,
                     int downsample_z,
                     const float* cube_up,
                     float* cube)
{
    (void)size_up_z;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int grid_size = gridDim.x*blockDim.x;
    const int data_size = size_x*size_y*size_z;

    // Calculate normalization constant
    const float nfactor = 1.0f/(downsample_x*downsample_y*downsample_z);

    //  loop until there are no pixels left
    while(idx < data_size)
    {
        int x,y,z;

        //  calculate pixel's cube indices
        map_index_1d_to_3d(x,y,z,idx,size_x,size_y,size_z);

        // Calculate the indices of the source cube
        int nx = offset_x + x * downsample_x;
        int ny = offset_y + y * downsample_y;
        int nz = offset_z + z * downsample_z;

        // Calculate the final 1d index of the destination cube
        int idx_dst = z*size_x*size_y +
                      y*size_x +
                      x;

        // Calculate and store the average value under the current position
        float sum = 0;

        for(int dsz = 0; dsz < downsample_z; ++dsz)
        {
            for(int dsy = 0; dsy < downsample_y; ++dsy)
            {
                for(int dsx = 0; dsx < downsample_x; ++dsx)
                {
                    int idx_src = (nz+dsz)*size_up_x*size_up_y +
                                  (ny+dsy)*size_up_x +
                                  (nx+dsx);

                    sum += cube_up[idx_src];
                }
            }
        }
        cube[idx_dst] = sum * nfactor;

        //  proceed to the next pixel
        idx += grid_size;
    }
}

} // namespace kernels_cuda_d
} // namespace scube
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_SCUBE_SCUBE_CUDA_KERNELS_D_CUH
