
#include "gbkfit/dmodel/mmaps/mmaps_omp_kernels.hpp"

namespace gbkfit {
namespace dmodel {
namespace mmaps {
namespace kernels_omp {

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
    #pragma omp parallel for
    for (int x = 0; x < size_x; ++x)
    {
        for (int y = 0; y < size_y; ++y)
        {
            float m0, m1, m2;
            float flx_sum = 0;
            float flxvel_sum = 0;
            float flxdvel2_sum = 0;
            int idx_map = y*size_x + x;

            //
            // Calculate moment 0 for the current spatial position.
            //

            #pragma omp simd
            for (int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y + y*size_x + x;
                float flx = std::max(0.0f, cube[idx_cube]);
                flx_sum += flx;
            }
            m0 = flx_sum;
            mom0[idx_map] = m0;

            //
            // Calculate moment 1 for the current spatial position.
            //

            #pragma omp simd
            for (int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y + y*size_x + x;
                float flx = std::max(0.0f, cube[idx_cube]);
                float vel = zero_z + z*step_z;
                flxvel_sum += flx*vel;
            }
            m1 = flx_sum > 0 ? flxvel_sum/flx_sum : 0;
            m1 = m1 - zero_z - size_z/2*step_z; // ...
            mom1[idx_map] = m1;

            //
            // Calculate moment 2 for the current spatial position.
            //

            #pragma omp simd
            for (int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y + y*size_x + x;
                float flx = std::max(0.0f, cube[idx_cube]);
                float vel = zero_z + z*step_z;
                flxdvel2_sum += flx*(vel-m1)*(vel-m1);
            }
            m2 = flx_sum > 0 ? std::sqrt(flxdvel2_sum/flx_sum) : 0;
            mom2[idx_map] = m2;
        }
    }
}

} // namespace kernels_omp
} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
