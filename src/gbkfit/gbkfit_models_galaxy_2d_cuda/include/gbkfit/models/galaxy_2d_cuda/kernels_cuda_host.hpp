#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_HOST_HPP
#define GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_HOST_HPP

namespace gbkfit {
namespace models {
namespace galaxy_2d {
namespace kernels_cuda_host {


void foo(float* out_velmap,
         float* out_sigmap,
         int data_size_x,
         int data_size_y);

} // namespace kernels_cuda_host
} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_KERNELS_CUDA_HOST_HPP
