
#include "gbkfit/models/galaxy_2d_cuda/kernels_cuda_host.hpp"
#include "gbkfit/models/galaxy_2d_cuda/kernels_cuda_device.cuh"

#include <iostream>
namespace gbkfit {
namespace models {
namespace galaxy_2d {
namespace kernels_cuda_host {

void foo(float* out_velmap,
         float* out_sigmap,
         int data_size_x,
         int data_size_y)
{
    std::cout << "Hello!" << std::endl;

    kernels_cuda_device::foo<<<1,1>>>(out_velmap,
                                      out_sigmap,
                                      data_size_x,
                                      data_size_y);
}



} // namespace kernels_cuda_host
} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
