#pragma once
#ifndef GBKFIT_MODELS_MODEL01_KERNELS_CUDA_DEVICE_CUH
#define GBKFIT_MODELS_MODEL01_KERNELS_CUDA_DEVICE_CUH

#include <cufft.h>

namespace gbkfit {
namespace models {
namespace model01 {
namespace kernels_cuda_device {

__device__ __constant__ float param_vsig;
__device__ __constant__ float param_vsys;
__device__ __constant__ float params_prj[16];
__device__ __constant__ float params_flx[16];
__device__ __constant__ float params_vel[16];

__device__
void map_index_1d_to_2d(int& out_xidx,
                        int& out_yidx,
                        int idx,
                        int width,
                        int height)
{
    out_yidx = idx/width;
    idx -= out_yidx*width;

    out_xidx = idx/1;
}

__device__
void map_index_1d_to_3d(int& out_xidx,
                        int& out_yidx,
                        int& out_zidx,
                        int idx,
                        int width,
                        int height,
                        int depth)
{
    out_zidx = idx/(width*height);
    idx -= out_zidx*width*height;

    out_yidx = idx/width;
    idx -= out_yidx*width;

    out_xidx = idx/1;
}

__device__
void evaluate_profile_gaussian(float& out, float x, float mu, float sigma)
{
    float a = 1.0f / sigma * sqrtf(2.0f*(float)M_PI);
    out = a * expf(-(x-mu)*(x-mu)/(2*sigma*sigma));
}

__device__
void evaluate_profile_flx_expdisk(float& out, float r, float i0, float r0)
{
    out = i0 * expf(-r/r0);
}

__device__
void evaluate_profile_vel_lramp(float& out, float r, float rt, float vt)
{
    out = r <= rt ? vt * r/rt : vt;
}

__device__
void evaluate_profile_vel_david(float& out, float r, float rt, float vt)
{
    out = vt * (1.0f - expf(-r/rt));
}

__device__
void evaluate_profile_vel_arctan(float& out, float r, float rt, float vt)
{
    out = vt * (2.0f/(float)M_PI) * atanf(r/rt);
}

__device__
void evaluate_profile_vel_epinat(float& out, float r, float rt, float vt, float a, float g)
{
    out = vt * powf(r/rt,g)/(1.0f+powf(r/rt,a));
}

__device__
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

__device__
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
            evaluate_profile_vel_david(vr,r,rt,vt);
        }
        else if (model_id == 3)
        {
            float rt = params[0];
            float vt = params[1];
            evaluate_profile_vel_arctan(vr,r,rt,vt);
        }
        else if (model_id == 4)
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

__global__
void array_3d_fill(int size_x,
                   int size_y,
                   int size_z,
                   float value,
                   float* data)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int grid_size = gridDim.x*blockDim.x;

    const int size = size_x*size_y*size_z;

    int x, y, z;

    while(idx < size)
    {
        // Calculate cube indices.
        map_index_1d_to_3d(x, y, z, idx, size_x, size_y, size_z);

        data[z*size_x*size_y + y*size_x + x] = value;

        idx += grid_size;
    }
}

__global__
void model_image_3d_evaluate(int profile_flx_id,
                             int profile_vel_id,
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
    const float DEG_TO_RAD_F = 0.017453293f;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int grid_size = gridDim.x*blockDim.x;

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

//  const int full_size_u_x = padding_u_x0 + padding_u_x1 + size_u_x;
//  const int full_size_u_y = padding_u_y0 + padding_u_y1 + size_u_y;
//  const int full_size_u_z = padding_u_z0 + padding_u_z1 + size_u_z;
//  const int full_size_u = full_size_u_x*full_size_u_y*full_size_u_z;

//  while(idx < full_size_u)
    while(idx < size_up_x*size_up_y*size_up_z)
    {
        int x, y, z;
        float flx_spat, vel_spat, sig_spat, flx_spec;
        float xn, yn, zn, xe, ye, rn, sini, cosi, sinpa, cospa;

        // Calculate cube indices.
    //  map_index_1d_to_3d(x, y, z, idx, full_size_u_x, full_size_u_y, full_size_u_z);

        map_index_1d_to_3d(x, y, z, idx, size_up_x, size_up_y, size_up_z);

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

        // Evaluate spectrum.
        zn =  (z - padding_u_z0 - size_u_z/2.0+0.5f) * step_u_z;

        evaluate_profile_gaussian(flx_spec,zn,vel_spat,sig_spat);

        // Calculate the right index in the cube.
        int idx_dst = z*size_up_x*size_up_y +
                      y*size_up_x +
                      x;

        // Save flux, we are done! Woohoo!
        flxcube_up[idx_dst] = flx_spat * flx_spec;

        // Proceed to the next pixel.
        idx += grid_size;
    }
}

__global__
void array_complex_multiply_and_scale(cufftComplex* array1,
                                      cufftComplex* array2,
                                      int length, float scale)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Loop until there are no elements left.
    while(idx < length)
    {
        cufftComplex a,b,t;

        a.x = array1[idx].x;
        a.y = array1[idx].y;

        b.x = array2[idx%length].x;
        b.y = array2[idx%length].y;

        // Perform complex multiplication and scalar scale.
        t.x = (a.x*b.x-a.y*b.y)*scale;
        t.y = (a.x*b.y+a.y*b.x)*scale;

        // Store result.
        array1[idx].x = t.x;
        array1[idx].y = t.y;

        // Proceed to the next element.
        idx += gridDim.x * blockDim.x;
    }
}

__global__
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
    (void)size_up_z;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int grid_size = gridDim.x*blockDim.x;
    const int data_size = size_x*size_y*size_z;

    // Calculate normalization constant.
    const float nfactor = 1.0f/(downsample_x*downsample_y*downsample_z);

    //  loop until there are no pixels left
    while(idx < data_size)
    {
        int x,y,z;

        //  calculate pixel's cube indices
        map_index_1d_to_3d(x,y,z,idx,size_x,size_y,size_z);

        // Calculate the indices of the source flxcube.
        int nx = padding_u_x0 + x * downsample_x;
        int ny = padding_u_y0 + y * downsample_y;
        int nz = padding_u_z0 + z * downsample_z;

        // Calculate final 1d index for destination flxcube.
        int idx_dst = z*size_x*size_y +
                      y*size_x +
                      x;

        // Iterate over the "subindices" and calculate their sum.
        float sum = 0;

        for(int dsz = 0; dsz < downsample_z; ++dsz)
        {
            for(int dsy = 0; dsy < downsample_y; ++dsy)
            {
                for(int dsx = 0; dsx < downsample_x; ++dsx)
                {
                    int idx_src = (nz+dsz)*size_up_x*size_up_y +
                                  (ny+dsy)*size_up_x +
                                  (nx+dsx);

                    sum += flxcube_up[idx_src];
                }
            }
        }

        // Normalize and store result.
        flxcube[idx_dst] = sum * nfactor;

        //  proceed to the next pixel
        idx += grid_size;
    }
}

__global__
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

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int grid_size = gridDim.x*blockDim.x;
    int data_size = size_x*size_y;

    //  loop until there are no pixels left
    while(idx < data_size)
    {
        int x, y;

        // Calculate pixel's cube indices
        map_index_1d_to_2d(x,y,idx,size_x,size_y);

        float m0, m1, m2;
        float sum_intvel = 0;
        float sum_int = 0;
        float sum_intdvel2 = 0;

        int idx_map = y*size_x + x;

        //
        // Calculate moment 0 for the current spatial position.
        //

        for(int z = 0; z < size_z; ++z)
        {
            int idx_cube = z*size_x*size_y +
                           y*size_x +
                           x;

            // Get intensity of the current slice.
            float int_cur = flxcube[idx_cube];
            int_cur = fmaxf(0.0f,int_cur);

            // Calculate quantities needed for the moment.
            sum_int += int_cur;
        }
        m0 = sum_int;
        flxmap[idx_map] = m0;

        //
        // Calculate moment 1 for the current spatial position.
        //

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
            int_cur = fmaxf(0.0f,int_cur);

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
            int_cur = fmaxf(0.0f,int_cur);

            // Calculate quantities needed for the moment.
            sum_intdvel2 += int_cur*(zn-m1)*(zn-m1);
        }
        m2 = 0.0f;
        if(sum_int > 0.0f)
            m2 = sum_intdvel2/sum_int;
        sigmap[idx_map] = sqrtf(m2);

        //  Proceed to the next pixel.
        idx += grid_size;
    }
}

} // namespace kernels_cuda_device
} // namespace model01
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_MODEL01_KERNELS_CUDA_DEVICE_CUH
