
#include "gbkfit/models/model01/kernels_cuda_host.hpp"
#include "gbkfit/models/model01/kernels_cuda_device.cuh"

namespace gbkfit {
namespace models {
namespace model01 {
namespace kernels_cuda_host {

void foo(float* out_velmap,
         float* out_sigmap,
         int data_size_x,
         int data_size_y)
{
    kernels_cuda_device::foo<<<1,1>>>(out_velmap,
                                      out_sigmap,
                                      data_size_x,
                                      data_size_y);
}



} // namespace kernels_cuda_host
} // namespace model01
} // namespace models
} // namespace gbkfit
