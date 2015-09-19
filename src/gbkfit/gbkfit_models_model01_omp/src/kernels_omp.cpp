
#include "gbkfit/models/model01/kernels_omp.hpp"

#include <algorithm>
#include <cmath>

namespace gbkfit {
namespace models {
namespace model01 {
namespace kernels_omp {

void evaluate_profile_gaussian(float& out, float x, float mu, float sigma)
{
    float a = 1.0f / sigma * std::sqrt(2.0f*(float)M_PI);
    out = a * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
}

void evaluate_profile_flx_expdisk(float& out, float r, float i0, float r0)
{
    out = i0 * std::exp(-r/r0);
}

void evaluate_profile_vel_lramp(float& out, float r, float rt, float vt)
{
    out = r <= rt ? vt * r/rt : vt;
}

void evaluate_profile_vel_arctan(float& out, float r, float rt, float vt)
{
    out = vt * (2.0f/(float)M_PI) * std::atan(r/rt);
}

void evaluate_profile_vel_epinat(float& out, float r, float rt, float vt, float a, float g)
{
    out = vt * std::pow(r/rt,g)/(1.0f+std::pow(r/rt,a));
}

void evaluate_model_flx(float& out, float r, int model_id, const float* params)
{
    out = 0;
    float ir = 0;

    if      (model_id == 1)
    {
        float i0 = params[0];
        float r0 = params[1];
        evaluate_profile_flx_expdisk(ir,r,i0,r0);
    }

    out += ir;
}

void evaluate_model_vel(float& out, float x, float r, float sini, int model_id, const float* params)
{
    out = 0;
    float vr = 0;
    float costheta = 0;

    if(r > 0)
    {
        if      (model_id == 1)
        {
            float rt = params[0];
            float vt = params[1];
            evaluate_profile_vel_lramp(vr,r,rt,vt);
        }
        else if (model_id == 2)
        {
            float rt = params[0];
            float vt = params[1];
            evaluate_profile_vel_arctan(vr,r,rt,vt);
        }
        else if (model_id == 3)
        {
            float rt = params[0];
            float vt = params[1];
            float a = params[2];
            float g = params[3];
            evaluate_profile_vel_epinat(vr,r,rt,vt,a,g);
        }

        costheta = x/r;
        out += vr * sini * costheta;
    }
}

void array_3d_fill(int size_x,
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
            for(int x = 0; x < size_x; ++x)
            {
                data[z*size_x*size_y + y*size_x + x] = value;
            }
        }
    }
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
    (void)size_up_z;

    const float DEG_TO_RAD_F = 0.017453293f;

    //
    // Extract parameters.
    //

    const float xo = params_prj[0];
    const float yo = params_prj[1];
    const float pa = params_prj[2] * DEG_TO_RAD_F;
    const float incl = params_prj[3] * DEG_TO_RAD_F;
    const float vsig = param_vsig;
    const float vsys = param_vsys;

    //
    // Evaluate spectral cube data.
    //

    #pragma omp parallel for
    for(int y = 0; y < padding_u_y0 + padding_u_y1 + size_u_y; ++y)
    {
        for(int x = 0; x < padding_u_x0 + padding_u_x1 + size_u_x; ++x)
        {
            float flx_spat, vel_spat, sig_spat, flx_spec;
            float xn, yn, zn, xe, ye, rn, sini, cosi, sinpa, cospa;

            // Transform image coordinates to spatial coordinates.
            xn = x - padding_u_x0;
            yn = y - padding_u_y0;
            xn += 0.5f;
            yn += 0.5f;
            xn *= step_u_x;
            yn *= step_u_y;

            // Transform spatial coordinates to disk coordinates.
            sini = std::sin(incl);
            cosi = std::cos(incl);
            sinpa = std::sin(pa);
            cospa = std::cos(pa);
            xe = -(xn-xo)*sinpa+(yn-yo)*cospa;
            ye = -(xn-xo)*cospa-(yn-yo)*sinpa;
            rn = std::sqrt((xe*xe)+(ye/cosi)*(ye/cosi));

            // Calculate spatial flux.
            evaluate_model_flx(flx_spat, rn, profile_flx_id, params_flx);

            // Calculate spatial velocity.
            evaluate_model_vel(vel_spat, xe, rn, sini, profile_vel_id, params_vel);
            vel_spat = vel_spat + vsys;

            // Calculate spatial velocity dispersion.
            sig_spat = vsig;

            // Evaluate a single spectrum.
            #pragma omp simd
            for(int z = 0; z < padding_u_z0 + padding_u_z1 + size_u_z; ++z)
            {
                // Evaluate spectrum.
                zn =  (z - padding_u_z0 - size_u_z/2.0+0.5f) * step_u_z;

                evaluate_profile_gaussian(flx_spec,zn,vel_spat,sig_spat);

                // Calculate the right index in the cube.
                int idx = z*size_up_x*size_up_y +
                          y*size_up_x +
                          x;

                // Save flux, we are done! Woohoo!
                flxcube_up[idx] = flx_spat * flx_spec;
#if 0
                float zero = 0.01;
                if(x < upsampled_padding_x0)
                    flxcube_padded[idx] = zero;
                if(y < upsampled_padding_y0)
                    flxcube_padded[idx] = zero;
                if(z < upsampled_padding_z0)
                    flxcube_padded[idx] = zero;
                if(x >= size_x*upsampling_x+upsampled_padding_x0)
                    flxcube_padded[idx] = zero;
                if(y >= size_y*upsampling_y+upsampled_padding_y0)
                    flxcube_padded[idx] = zero;
                if(z >= size_z*upsampling_z+upsampled_padding_z0)
                    flxcube_padded[idx] = zero;
#endif

            }
        }
    }
}



void array_complex_multiply_and_scale(fftwf_complex* array1,
                                      fftwf_complex* array2,
                                      int length, float scale)
{
    #pragma omp parallel for
    for(int idx = 0; idx < length; ++idx)
    {
        fftwf_complex a,b,t;

        a[0] = array1[idx][0];
        a[1] = array1[idx][1];

        b[0] = array2[idx%length][0];
        b[1] = array2[idx%length][1];

        // Perform complex multiplication and scalar scale.
        t[0] = (a[0]*b[0]-a[1]*b[1])*scale;
        t[1] = (a[0]*b[1]+a[1]*b[0])*scale;

        // Store result.
        array1[idx][0] = t[0];
        array1[idx][1] = t[1];
    }
}

void model_image_3d_convolve_fft(float* image,
                                 float* image_complex,
                                 float* kernel_complex,
                                 int size_x,
                                 int size_y,
                                 int size_z,
                                 fftwf_plan plan_r2c,
                                 fftwf_plan plan_c2r)
{
    // Perform real-to-complex transform.
    fftwf_execute_dft_r2c(plan_r2c, image, reinterpret_cast<fftwf_complex*>(image_complex));

    // Perform complex multiplication on Fourier space (i.e., convolution).
    const int length = size_z*size_y*(size_x/2+1);
    const float nfactor = 1.0f/(size_x*size_y*size_z);
    array_complex_multiply_and_scale(reinterpret_cast<fftwf_complex*>(image_complex),
                                     reinterpret_cast<fftwf_complex*>(kernel_complex),
                                     length, nfactor);

    // Perform complex-to-real transform.
    fftwf_execute_dft_c2r(plan_c2r, reinterpret_cast<fftwf_complex*>(image_complex), image);
}

void model_image_3d_downsample_and_copy(const float* flxcube_src,
                                        int size_x,
                                        int size_y,
                                        int size_z,
                                        int upsampled_padding_x0,
                                        int upsampled_padding_y0,
                                        int upsampled_padding_z0,
                                        int upsampled_padded_size_x,
                                        int upsampled_padded_size_y,
                                        int upsampled_padded_size_z,
                                        int downsample_x,
                                        int downsample_y,
                                        int downsample_z,
                                        float* flxcube_dst)
{
    (void)upsampled_padded_size_z;

    // Calculate normalization constant.
    const float nfactor = 1.0f/(downsample_x*downsample_y*downsample_z);

    #pragma omp parallel for
    // Iterate over the indices of the destination flxcube.
    for(int z = 0; z < size_z; ++z)
    {
        for(int y = 0; y < size_y; ++y)
        {
            for(int x = 0; x < size_x; ++x)
            {
                // Calculate the indices of the source flxcube.
                int nx = upsampled_padding_x0 + x * downsample_x;
                int ny = upsampled_padding_y0 + y * downsample_y;
                int nz = upsampled_padding_z0 + z * downsample_z;

                // Calculate final 1d index for destination flxcube.
                int idx_dst = z*size_x*size_y +
                              y*size_x +
                              x;

                // Iterate over the "subindices" and calculate their sum.
                float sum = 0;
                #pragma omp simd
                for(int dsz = 0; dsz < downsample_z; ++dsz)
                {
                    for(int dsy = 0; dsy < downsample_y; ++dsy)
                    {
                        for(int dsx = 0; dsx < downsample_x; ++dsx)
                        {
                            int idx_src = (nz+dsz)*upsampled_padded_size_x*upsampled_padded_size_y +
                                          (ny+dsy)*upsampled_padded_size_x +
                                          (nx+dsx);

                            sum += flxcube_src[idx_src];
                        }
                    }
                }

                // Normalize and store result.
                flxcube_dst[idx_dst] = sum * nfactor;
            }
        }
    }
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
    (void)step_x;
    (void)step_y;

    #pragma omp parallel for
    for(int y = 0; y < size_y; ++y)
    {
        for(int x = 0; x < size_x; ++x)
        {
            float m0, m1, m2;
            float sum_intvel = 0;
            float sum_int = 0;
            float sum_intdvel2 = 0;

            int idx_map = y*size_x + x;

            //
            // Calculate moment 0 for the current spatial position.
            //

            #pragma omp simd
            for(int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y +
                               y*size_x +
                               x;

                // Get intensity of the current slice.
                float int_cur = flxcube[idx_cube];
                int_cur = std::max(0.0f,int_cur);

                // Calculate quantities needed for the moment.
                sum_int += int_cur;
            }
            m0 = sum_int;
            flxmap[idx_map] = m0;

            //
            // Calculate moment 1 for the current spatial position.
            //

            #pragma omp simd
            for(int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y +
                               y*size_x +
                               x;

                // Calculate the velocity for the current slice.
                // Always assume that the central pixel(s) are at zero velocity.
                float zn = (z-size_z/2.0+0.5f);
                zn *= step_z;

                // Get intensity of the current slice.
                float int_cur = flxcube[idx_cube];
                int_cur = std::max(0.0f,int_cur);

                // Calculate quantities needed for the moment.
                sum_intvel += int_cur*zn;
            }
            m1 = 0.0f;
            if(sum_int > 0.0f)
                m1 = sum_intvel/sum_int;
            velmap[idx_map] = m1;

            //
            // Calculate moment 2 for the current spatial position.
            //

            #pragma omp simd
            for(int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y +
                               y*size_x +
                               x;

                // Calculate the velocity for the current slice.
                // Always assume that the central pixel(s) are at zero velocity.
                float zn = (z-size_z/2.0+0.5f);
                zn *= step_z;

                // Get intensity of the current slice.
                float int_cur = flxcube[idx_cube];
                int_cur = std::max(0.0f,int_cur);

                // Calculate quantities needed for the moment.
                sum_intdvel2 += int_cur*(zn-m1)*(zn-m1);
            }
            m2 = 0.0f;
            if(sum_int > 0.0f)
                m2 = sum_intdvel2/sum_int;
            sigmap[idx_map] = std::sqrt(m2);
        }
    }
}

} // namespace kernels_omp
} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
