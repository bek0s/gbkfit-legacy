
#include "gbkfit/dmodel/scube/scube_cuda_kernels_h.hpp"
#include "gbkfit/dmodel/scube/scube_cuda_kernels_d.cuh"

namespace gbkfit {
namespace dmodel {
namespace scube {
namespace kernels_cuda_h
{

void fill_cube(int size_x,
               int size_y,
               int size_z,
               float value,
               float* data)
{
    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_d::fill_cube<<<num_blocks, num_threads>>>(size_x,
                                                           size_y,
                                                           size_z,
                                                           value,
                                                           data);
    cudaDeviceSynchronize();
}

void array_complex_multiply_and_scale(cufftComplex* array1,
                                      cufftComplex* array2,
                                      int length, float scale)
{
    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_d::array_complex_multiply_and_scale<<<num_blocks,num_threads>>>(array1,
                                                                                 array2,
                                                                                 length,
                                                                                 scale);

    cudaDeviceSynchronize();
}

void convolve_cube(float* flxcube,
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

    cudaDeviceSynchronize();
}


void downsample_cube(int size_x,
                     int size_y,
                     int size_z,
                     int size_up_x,
                     int size_up_y,
                     int size_up_z,
                     int offset_x,
                     int offset_y,
                     int offset_z,
                     int downsample_x,
                     int downsample_y,
                     int downsample_z,
                     const float* flxcube_up,
                     float* flxcube)
{
    dim3 num_blocks(512,1,1);
    dim3 num_threads(128,1,1);

    kernels_cuda_d::downsample_cube<<<num_blocks, num_threads>>>(size_x,
                                                                 size_y,
                                                                 size_z,
                                                                 size_up_x,
                                                                 size_up_y,
                                                                 size_up_z,
                                                                 offset_x,
                                                                 offset_y,
                                                                 offset_z,
                                                                 downsample_x,
                                                                 downsample_y,
                                                                 downsample_z,
                                                                 flxcube_up,
                                                                 flxcube);
    cudaDeviceSynchronize();
}

} // namespace kernels_cuda_h
} // namespace scube
} // namespace dmodel
} // namespace gbkfit
