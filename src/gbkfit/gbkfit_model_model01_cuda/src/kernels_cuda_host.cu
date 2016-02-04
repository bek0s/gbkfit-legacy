
#include "gbkfit/models/model01/kernels_cuda_host.hpp"
#include "gbkfit/models/model01/kernels_cuda_device.cuh"
#include <iostream>

namespace gbkfit {
namespace models {
namespace model01 {
namespace kernels_cuda_host {

void array_3d_fill(int size_x,
                   int size_y,
                   int size_z,
                   float value,
                   float* data)
{
    dim3 num_blocks(256,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_device::array_3d_fill<<<num_blocks,num_threads>>>(size_x, size_y, size_z, value, data);

    cudaDeviceSynchronize();
}

void model_image_3d_evaluate(int profile_flx_id,
                             int profile_vel_id,
                             const float param_vsig,
                             const float param_vsys,
                             const float* params_prj,
                             const float* params_flx,
                             const float* params_vel,
                             int size_u_x,
                             int size_u_y,
                             int size_u_z,
                             int size_up_x,
                             int size_up_y,
                             int size_up_z,
                             float step_u_x,
                             float step_u_y,
                             float step_u_z,
                             int padding_u_x0,
                             int padding_u_y0,
                             int padding_u_z0,
                             int padding_u_x1,
                             int padding_u_y1,
                             int padding_u_z1,
                             float* flxcube_up)
{
    cudaMemcpyToSymbol(kernels_cuda_device::param_vsig, &param_vsig,  1*sizeof(float), 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_device::param_vsys, &param_vsys,  1*sizeof(float), 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_device::params_prj,  params_prj, 16*sizeof(float), 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_device::params_flx,  params_flx, 16*sizeof(float), 0, cudaMemcpyDefault);
    cudaMemcpyToSymbol(kernels_cuda_device::params_vel,  params_vel, 16*sizeof(float), 0, cudaMemcpyDefault);

    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_device::model_image_3d_evaluate<<<num_blocks,num_threads>>>(profile_flx_id,
                                                                             profile_vel_id,
                                                                             size_u_x,
                                                                             size_u_y,
                                                                             size_u_z,
                                                                             size_up_x,
                                                                             size_up_y,
                                                                             size_up_z,
                                                                             step_u_x,
                                                                             step_u_y,
                                                                             step_u_z,
                                                                             padding_u_x0,
                                                                             padding_u_y0,
                                                                             padding_u_z0,
                                                                             padding_u_x1,
                                                                             padding_u_y1,
                                                                             padding_u_z1,
                                                                             flxcube_up);

    cudaDeviceSynchronize();
}

void array_complex_multiply_and_scale(cufftComplex* array1,
                                      cufftComplex* array2,
                                      int length, float scale)
{
    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_device::array_complex_multiply_and_scale<<<num_blocks,num_threads>>>(array1,
                                                                                      array2,
                                                                                      length, scale);

    cudaDeviceSynchronize();
}

void model_image_3d_convolve_fft(float* flxcube,
                                 cufftComplex* flxcube_complex,
                                 cufftComplex* psfcube_complex,
                                 int size_x,
                                 int size_y,
                                 int size_z,
                                 cufftHandle plan_r2c,
                                 cufftHandle plan_c2r)
{

    // Perform real-to-complex transform.
    cufftExecR2C(plan_r2c, flxcube, reinterpret_cast<cufftComplex*>(flxcube_complex));

    // Perform complex multiplication on Fourier space (i.e., convolution).
    const int length = size_z*size_y*(size_x/2+1);
    const float nfactor = 1.0f/(size_x*size_y*size_z);
    array_complex_multiply_and_scale(reinterpret_cast<cufftComplex*>(flxcube_complex),
                                     reinterpret_cast<cufftComplex*>(psfcube_complex),
                                     length, nfactor);

    // Perform complex-to-real transform.
    cufftExecC2R(plan_c2r, reinterpret_cast<cufftComplex*>(flxcube_complex), flxcube);

}

void model_image_3d_downsample_and_copy(const float* flxcube_up,
                                        int size_x,
                                        int size_y,
                                        int size_z,
                                        int size_up_x,
                                        int size_up_y,
                                        int size_up_z,
                                        int padding_u_x0,
                                        int padding_u_y0,
                                        int padding_u_z0,
                                        int downsample_x,
                                        int downsample_y,
                                        int downsample_z,
                                        float* flxcube)
{
    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_device::model_image_3d_downsample_and_copy<<<num_blocks,num_threads>>>(flxcube_up,
                                                                                        size_x,
                                                                                        size_y,
                                                                                        size_z,
                                                                                        size_up_x,
                                                                                        size_up_y,
                                                                                        size_up_z,
                                                                                        padding_u_x0,
                                                                                        padding_u_y0,
                                                                                        padding_u_z0,
                                                                                        downsample_x,
                                                                                        downsample_y,
                                                                                        downsample_z,
                                                                                        flxcube);
    cudaDeviceSynchronize();
}

void model_image_3d_extract_moment_maps(const float* flxcube,
                                        int size_x,
                                        int size_y,
                                        int size_z,
                                        float step_x,
                                        float step_y,
                                        float step_z,
                                        float* flxmap,
                                        float* velmap,
                                        float* sigmap)
{
    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_device::model_image_3d_extract_moment_maps<<<num_blocks,num_threads>>>(flxcube,
                                                                                        size_x,
                                                                                        size_y,
                                                                                        size_z,
                                                                                        step_x,
                                                                                        step_y,
                                                                                        step_z,
                                                                                        flxmap,
                                                                                        velmap,
                                                                                        sigmap);

    cudaDeviceSynchronize();
}

} // namespace kernels_cuda_host
} // namespace model01
} // namespace models
} // namespace gbkfit
