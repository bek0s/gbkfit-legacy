
#include "gbkfit/dmodel/mmaps/mmaps_cuda_kernels_h.hpp"
#include "gbkfit/dmodel/mmaps/mmaps_cuda_kernels_d.cuh"

namespace gbkfit {
namespace dmodel {
namespace mmaps {
namespace kernels_cuda_h
{

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
    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_d::extract_maps_mmnt<<<num_blocks,num_threads>>>(cube,
                                                                  size_x,
                                                                  size_y,
                                                                  size_z,
                                                                  zero_z,
                                                                  step_z,
                                                                  mom0,
                                                                  mom1,
                                                                  mom2);

    cudaDeviceSynchronize();

}

} // namespace kernels_cuda_h
} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
