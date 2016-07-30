
#include "gbkfit/dmodel/scube/scube_omp_kernels.hpp"

namespace gbkfit {
namespace dmodel {
namespace scube {
namespace kernels_omp {

void clear_cube(int size_x,
               int size_y,
               int size_z,
               float value,
               float* data)
{
    #pragma omp parallel for
    for(int z = 0; z < size_z; ++z)
    {
        for(int y = 0; y < size_y; ++y)
        {
            #pragma omp simd
            for(int x = 0; x < size_x; ++x)
            {
                data[z*size_x*size_y + y*size_x + x] = value;
            }
        }
    }
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
                     const float* cube_up,
                     float* cube)
{
    (void)size_up_z;

    // Calculate normalization constant
    const float nfactor = 1.0f/(downsample_x*downsample_y*downsample_z);

    // Iterate over the indices of the destination cube
    #pragma omp parallel for
    for(int z = 0; z < size_z; ++z)
    {
        for(int y = 0; y < size_y; ++y)
        {
            for(int x = 0; x < size_x; ++x)
            {
                // Calculate the indices of the source cube
                int nx = offset_x + x * downsample_x;
                int ny = offset_y + y * downsample_y;
                int nz = offset_z + z * downsample_z;

                // Calculate the final 1d index of the destination cube
                int idx_dst = z*size_x*size_y +
                              y*size_x +
                              x;

                // Calculate and store the average value under the current
                // position
                float sum = 0;
                #pragma omp simd
                for(int dsz = 0; dsz < downsample_z; ++dsz)
                {
                    for(int dsy = 0; dsy < downsample_y; ++dsy)
                    {
                        for(int dsx = 0; dsx < downsample_x; ++dsx)
                        {
                            int idx_src = (nz+dsz)*size_up_x*size_up_y +
                                          (ny+dsy)*size_up_x +
                                          (nx+dsx);

                            sum += cube_up[idx_src];
                        }
                    }
                }
                cube[idx_dst] = sum * nfactor;
            }
        }
    }
}

void array_complex_multiply_and_scale(fftwf_complex* array1,
                                      fftwf_complex* array2,
                                      int size,
                                      float scale)
{
    #pragma omp parallel for
    for(int i = 0; i < size; ++i)
    {
        fftwf_complex a, b, t;

        a[0] = array1[i][0];
        a[1] = array1[i][1];

        b[0] = array2[i%size][0];
        b[1] = array2[i%size][1];

        // Perform complex multiplication and scalar scale.
        t[0] = (a[0]*b[0]-a[1]*b[1])*scale;
        t[1] = (a[0]*b[1]+a[1]*b[0])*scale;

        // Store result.
        array1[i][0] = t[0];
        array1[i][1] = t[1];
    }
}

void model_image_3d_convolve_fft(float* flxcube,
                                 fftwf_complex *flxcube_complex,
                                 fftwf_complex *psfcube_complex,
                                 int size_x,
                                 int size_y,
                                 int size_z,
                                 fftwf_plan plan_r2c,
                                 fftwf_plan plan_c2r)
{
    // Perform real-to-complex transform.
    fftwf_execute_dft_r2c(plan_r2c, flxcube, reinterpret_cast<fftwf_complex*>(flxcube_complex));

    // Perform complex multiplication on Fourier space (i.e., convolution).
    const int length = size_z*size_y*(size_x/2+1);
    const float nfactor = 1.0f/(size_x*size_y*size_z);
    array_complex_multiply_and_scale(reinterpret_cast<fftwf_complex*>(flxcube_complex),
                                     reinterpret_cast<fftwf_complex*>(psfcube_complex),
                                     length, nfactor);

    // Perform complex-to-real transform.
    fftwf_execute_dft_c2r(plan_c2r, reinterpret_cast<fftwf_complex*>(flxcube_complex), flxcube);
}

} // namespace kernels_omp
} // namespace scube
} // namespace dmodel
} // namespace gbkfit
