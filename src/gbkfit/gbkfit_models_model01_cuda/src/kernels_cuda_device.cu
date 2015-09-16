
#include "gbkfit/models/model01/kernels_cuda_device.cuh"

namespace gbkfit {
namespace models {
namespace model01 {
namespace kernels_cuda_device {

__global__
void foo(float* out_velmap,
         float* out_sigmap,
         int data_size_x,
         int data_size_y)
{
    out_velmap[0] = 101;
}

} // namespace kernels_cuda_device
} // namespace model01
} // namespace models
} // namespace gbkfit
