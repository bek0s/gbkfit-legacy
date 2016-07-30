#pragma once
#ifndef GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_KERNELS_D_CUH
#define GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_KERNELS_D_CUH

namespace gbkfit {
namespace dmodel {
namespace mmaps {
namespace kernels_cuda_d
{

__device__
void map_index_1d_to_2d(int& out_xidx,
                        int& out_yidx,
                        int idx,
                        int width,
                        int height)
{
    out_yidx = idx/width;
    idx -= out_yidx*width;

    out_xidx = idx/1;
}

__global__
void extract_maps_mmnt(const float* cube,
                       int size_x,
                       int size_y,
                       int size_z,
                       float zero_z,
                       float step_z,
                       float* mom0,
                       float* mom1,
                       float* mom2)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int grid_size = gridDim.x*blockDim.x;
    int data_size = size_x*size_y;

    //  loop until there are no pixels left
    while(idx < data_size)
    {
        int x, y;

        // Calculate pixel's cube indices
        map_index_1d_to_2d(x, y, idx, size_x, size_y);

        float m0, m1, m2;
        float flx_sum = 0;
        float flxvel_sum = 0;
        float flxdvel2_sum = 0;
        int idx_map = y*size_x + x;

        //
        // Calculate moment 0 for the current spatial position.
        //

        for(int z = 0; z < size_z; ++z)
        {
            int idx_cube = z*size_x*size_y + y*size_x + x;
            float flx = fmaxf(0.0f, cube[idx_cube]);
            flx_sum += flx;
        }
        m0 = flx_sum;
        mom0[idx_map] = m0;

        //
        // Calculate moment 1 for the current spatial position.
        //

        for(int z = 0; z < size_z; ++z)
        {
            int idx_cube = z*size_x*size_y + y*size_x + x;
            float flx = fmaxf(0.0f, cube[idx_cube]);
            float vel = zero_z + z*step_z;
            flxvel_sum += flx*vel;
        }
        m1 = flx_sum > 0 ? flxvel_sum/flx_sum : 0;
        m1 = m1 - zero_z - size_z/2*step_z; // ...
        mom1[idx_map] = m1;

        //
        // Calculate moment 2 for the current spatial position.
        //

        for(int z = 0; z < size_z; ++z)
        {
            int idx_cube = z*size_x*size_y + y*size_x + x;
            float flx = fmaxf(0.0f, cube[idx_cube]);
            float vel = zero_z + z*step_z;
            flxdvel2_sum += flx*(vel-m1)*(vel-m1);
        }
        m2 = flx_sum > 0 ? sqrtf(flxdvel2_sum/flx_sum) : 0;
        mom2[idx_map] = m2;

        //
        //  Proceed to the next pixel.
        //

        idx += grid_size;
    }
}

} // namespace kernels_cuda_d
} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit

#endif // GBKFIT_DMODEL_MMAPS_MMAPS_CUDA_KERNELS_D_CUH
