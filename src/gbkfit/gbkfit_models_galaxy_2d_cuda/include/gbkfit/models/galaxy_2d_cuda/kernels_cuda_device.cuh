#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_DEVICE_CUH
#define GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_DEVICE_CUH

namespace gbkfit {
namespace models {
namespace galaxy_2d {
namespace kernels_cuda_device {

__global__
void foo(float* out_velmap,
         float* out_sigmap,
         int data_size_x,
         int data_size_y);

} // namespace kernels_cuda_device
} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_DEVICE_CUH
