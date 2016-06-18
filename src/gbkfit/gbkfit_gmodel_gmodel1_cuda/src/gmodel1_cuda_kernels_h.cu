
#include "gbkfit/gmodel/gmodel1/gmodel1_cuda_kernels_h.hpp"
#include "gbkfit/gmodel/gmodel1/gmodel1_cuda_kernels_d.cuh"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {
namespace kernels_cuda_h {

void evaluate_model(int flx_profile,
                    int vel_profile,
                    const float* params_prj,
                    const float* params_flx,
                    const float* params_vel,
                    const float param_vsys,
                    const float param_vsig,
                    int data_size_x,
                    int data_size_y,
                    int data_size_z,
                    float data_zero_x,
                    float data_zero_y,
                    float data_zero_z,
                    float data_step_x,
                    float data_step_y,
                    float data_step_z,
                    float* data)
{
    cudaMemcpyToSymbol(kernels_cuda_d::params_prj, params_prj, 16*sizeof(float),
                       0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_d::params_flx, params_flx, 16*sizeof(float),
                       0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_d::params_vel, params_vel, 16*sizeof(float),
                       0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_d::param_vsys, &param_vsys, 1*sizeof(float),
                       0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_d::param_vsig, &param_vsig, 1*sizeof(float),
                       0, cudaMemcpyDefault);

    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_d::evaluate_model<<<num_blocks,num_threads>>>(flx_profile,
                                                               vel_profile,
                                                               data_size_x,
                                                               data_size_y,
                                                               data_size_z,
                                                               data_zero_x,
                                                               data_zero_y,
                                                               data_zero_z,
                                                               data_step_x,
                                                               data_step_y,
                                                               data_step_z,
                                                               data);

    cudaDeviceSynchronize();
}

} // namespace kernels_cuda_h
} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit
