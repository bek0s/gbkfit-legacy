
#include "gbkfit/models/galaxy_2d_omp/kernels_omp.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
namespace gbkfit {
namespace models {
namespace galaxy_2d {
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

void model_image_3d_evaluate(int profile_flx_id,
                             int profile_vel_id,
                             const float param_vsig,
                             const float param_vsys,
                             const float* params_prj,
                             const float* params_flx,
                             const float* params_vel,
                             int data_size_x,
                             int data_size_y,
                             int data_size_z,
                             int data_size_x_padded,
                             int data_size_y_padded,
                             int data_size_z_padded,
                             int data_padding_x,
                             int data_padding_y,
                             int data_padding_z,
                             float data_step_x,
                             float data_step_y,
                             float data_step_z,
                             int upsampling_x,
                             int upsampling_y,
                             int upsampling_z,
                             float* flxcube_padded)
{
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
    for(int y = 0; y < data_size_y_padded; ++y)
    {
        for(int x = 0; x < data_size_x_padded; ++x)
        {
            float flx_spat, vel_spat, sig_spat, flx_spec;
            float xn, yn, zn, xe, ye, rn, sini, cosi, sinpa, cospa;

            // Transform image coordinates to spatial coordinates.
            xn = x - data_padding_x;
            yn = y - data_padding_y;
            xn += 0.5f;
            yn += 0.5f;
            xn *= data_step_x/upsampling_x; // * upsampling?
            yn *= data_step_y/upsampling_y;

            // Transform spatial coordinates to disk coordinates.
            sini = std::sin(incl);
            cosi = std::cos(incl);
            sinpa = std::sin(pa);
            cospa = std::cos(pa);
            xe = -(xn-xo)*sinpa+(yn-yo)*cospa;
            ye = -(xn-xo)*cospa-(yn-yo)*sinpa;
            rn = std::sqrt((xe*xe)+(ye/cosi)*(ye/cosi));

            // Calculate spatial flux.
            evaluate_model_flx(flx_spat,rn,profile_flx_id,params_flx);

            // Calculate spatial velocity.
            evaluate_model_vel(vel_spat,xe,rn,sini,profile_vel_id,params_vel);
            vel_spat = vel_spat + vsys;

            // Calculate spatial velocity dispersion.
            sig_spat = vsig;


            float vcur;

            //vcur
            // Evaluate a single spectrum.
            #pragma omp simd
            for(int z = 0; z < data_size_z_padded; ++z)
            {

                vcur =  (z - data_padding_z - data_size_z*upsampling_z/2.0+0.5f) * data_step_z/upsampling_z;

            //  std::cout << vcur << " ";

                evaluate_profile_gaussian(flx_spec,vcur,vel_spat,sig_spat);

                // Calculate the right index in the cube.
                int idx = z*data_size_x_padded*data_size_y_padded +
                          y*data_size_x_padded +
                          x;

                // Save flux, we are done! Woohoo!
                flxcube_padded[idx] = flx_spat * flx_spec;
            //  flxcube_padded[idx] = vel_cur;
            //  flxcube_padded[idx] = xn;

                if(x < data_padding_x)
                    flxcube_padded[idx] = 1;
                if(y < data_padding_y)
                    flxcube_padded[idx] = 1;
                if(z < data_padding_z)
                    flxcube_padded[idx] = 1;

                if(x > data_size_x*upsampling_x+data_padding_x)
                    flxcube_padded[idx] = 1;
                if(y > data_size_y*upsampling_y+data_padding_y)
                    flxcube_padded[idx] = 1;
                if(z > data_size_z*upsampling_z+data_padding_z)
                    flxcube_padded[idx] = 1;
            }

        //  std::cout << std::endl;
        }
    }
}

void model_image_3d_copy(const float* flxcube_src,
                         int data_size_x,
                         int data_size_y,
                         int data_size_z,
                         int data_size_x_padded,
                         int data_size_y_padded,
                         int data_size_z_padded,
                         float* flxcube_dst)
{
    // Calculate x, y, z margin. Do not mind integer division by 2.
    // This is correct for both even and odd sizes.
    int data_margin_x = (data_size_x_padded - data_size_x) / 2;
    int data_margin_y = (data_size_y_padded - data_size_y) / 2;
    int data_margin_z = (data_size_z_padded - data_size_z) / 2;

    // Iterate over the indices of the destination flxcube.
    for(int z1 = 0; z1 < data_size_z; ++z1)
    {
        for(int y1 = 0; y1 < data_size_y; ++y1)
        {
            for(int x1 = 0; x1 < data_size_x; ++x1)
            {
                // Calculate the indices of the source flxcube.
                int z0 = z1 + data_margin_z;
                int y0 = y1 + data_margin_y;
                int x0 = x1 + data_margin_x;

                // Calculate final 1d index for source flxcube.
                int idx_src = z0*data_size_x_padded*data_size_y_padded +
                              y0*data_size_x_padded +
                              x0;

                // Calculate final 1d index for destination flxcube.
                int idx_dst = z1*data_size_x*data_size_y +
                              y1*data_size_x +
                              x1;

                // Copy!
                flxcube_dst[idx_dst] = flxcube_src[idx_src];
            }
        }
    }
}

void model_image_3d_downsample_and_copy(const float* flxcube_src,
                                        int data_size_x,
                                        int data_size_y,
                                        int data_size_z,
                                        int data_size_x_padded,
                                        int data_size_y_padded,
                                        int data_size_z_padded,
                                        int data_padding_x,
                                        int data_padding_y,
                                        int data_padding_z,
                                        int downsample_x,
                                        int downsample_y,
                                        int downsample_z,
                                        float* flxcube_dst)
{
    (void)data_size_z_padded;

    // Calculate normalization constant.
    const float norm_constant = 1.0f/(downsample_x*downsample_y*downsample_z);

    #pragma omp parallel for
    for(int z = 0; z < data_size_z; ++z)
    {
        for(int y = 0; y < data_size_y; ++y)
        {
            for(int x = 0; x < data_size_x; ++x)
            {
                float sum = 0;
                int nx = data_padding_x + x * downsample_x;
                int ny = data_padding_y + y * downsample_y;
                int nz = data_padding_z + z * downsample_z;

                int idx_dst = z*data_size_y*data_size_x +
                              y*data_size_x +
                              x;

                for(int dsz = 0; dsz < downsample_z; ++dsz)
                {
                    for(int dsy = 0; dsy < downsample_y; ++dsy)
                    {
                        for(int dsx = 0; dsx < downsample_x; ++dsx)
                        {
                            /*
                            std::cout << dsx << std::endl
                                      << dsy << std::endl
                                      << dsz << std::endl
                                      << "==============" << std::endl;
                                      */
                            int idx_src = (nz+dsz)*data_size_x_padded*data_size_y_padded +
                                          (ny+dsy)*data_size_x_padded +
                                          (nx+dsx);

                            sum += flxcube_src[idx_src];
                        }
                    }
                }
                flxcube_dst[idx_dst] = sum * norm_constant;
            }
        }
    }
}

void model_image_3d_extract_moment_maps(const float* flxcube,
                                        int data_size_x,
                                        int data_size_y,
                                        int data_size_z,
                                        float data_step_x,
                                        float data_step_y,
                                        float data_step_z,
                                        float* flxmap,
                                        float* velmap,
                                        float* sigmap)
{
    (void)data_step_x;
    (void)data_step_y;

//  #pragma omp parallel for
    for(int y = 0; y < data_size_y; ++y)
    {
        for(int x = 0; x < data_size_x; ++x)
        {
            float m0, m1, m2;
            float sum_intvel = 0;
            float sum_int = 0;
            float sum_intdvel2 = 0;

            int idx_map = y*data_size_x + x;

            //
            // Calculate moment 0 for the current spatial position.
            //

        //  #pragma omp simd
            for(int z = 0; z < data_size_z; ++z)
            {
                int idx_cube = z*data_size_x*data_size_y +
                               y*data_size_x +
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

        //  #pragma omp simd
            for(int z = 0; z < data_size_z; ++z)
            {
                int idx_cube = z*data_size_x*data_size_y +
                               y*data_size_x +
                               x;

                // Calculate the velocity for the current slice.
                // Always assume that the central pixel(s) are at zero velocity.
                float zn = (z-data_size_z/2.0+0.5f);
                zn *= data_step_z;

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

        //  #pragma omp simd
            for(int z = 0; z < data_size_z; ++z)
            {
                int idx_cube = z*data_size_x*data_size_y +
                               y*data_size_x +
                               x;

                // Calculate the velocity for the current slice.
                // Always assume that the central pixel(s) are at zero velocity.
                float zn = (z-data_size_z/2.0+0.5f);
                zn *= data_step_z;

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
