#pragma once
#ifndef GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_KERNELS_D_CUH
#define GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_KERNELS_D_CUH

namespace gbkfit {
namespace gmodel {
namespace gmodel01 {
namespace kernels_cuda_d {

__device__ __constant__ float params_prj[16];
__device__ __constant__ float params_flx[16];
__device__ __constant__ float params_vel[16];
__device__ __constant__ float param_vsys;
__device__ __constant__ float param_vsig;

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
float evaluate_profile_gaussian(float x, float mu, float sigma)
{
    float a = 1.0f / sigma * sqrtf(2.0f*(float)M_PI);
    return a * expf(-(x-mu)*(x-mu)/(2.0f*sigma*sigma));
}

__device__
float evaluate_profile_flx_exponential(float r, float i0, float r0)
{
    return i0 * expf(-r/r0);
}

__device__
float evaluate_profile_vel_lramp(float r, float rt, float vt)
{
    return r <= rt ? vt * r/rt : vt;
}

__device__
float evaluate_profile_vel_boissier(float r, float rt, float vt)
{
    return vt * (1.0f - expf(-r/rt));
}

__device__
float evaluate_profile_vel_arctan(float r, float rt, float vt)
{
    return vt * (2.0f/(float)M_PI) * atanf(r/rt);
}

__device__
float evaluate_profile_vel_epinat(float r, float rt, float vt, float a, float g)
{
    return vt * powf(r/rt,g)/(1.0f+powf(r/rt,a));
}

__device__
float evaluate_profile_flx(int profile, float r, const float* params)
{
    float flx = 0;

    switch(profile)
    {
        case 1: {
            float i0 = params[0];
            float r0 = params[1];
            flx = evaluate_profile_flx_exponential(r, i0, r0);
            break;
        }
    }

    return flx;
}

__device__
float evaluate_profile_vel(int profile, float r, const float* params)
{
    float vel = 0;

    if (r > 0)
    {
        switch(profile)
        {
            case 1: {
                float rt = params[0];
                float vt = params[1];
                vel = evaluate_profile_vel_lramp(r, rt, vt);
                break;
            }
            case 2: {
                float rt = params[0];
                float vt = params[1];
                vel = evaluate_profile_vel_boissier(r, rt, vt);
                break;
            }
            case 3: {
                float rt = params[0];
                float vt = params[1];
                vel = evaluate_profile_vel_arctan(r, rt, vt);
                break;
            }
            case 4: {
                float rt = params[0];
                float vt = params[1];
                float a = params[2];
                float g = params[3];
                vel = evaluate_profile_vel_epinat(r, rt, vt, a, g);
                break;
            }
        }
    }

    return vel;
}

__global__
void evaluate_model(int profile_flx,
                    int profile_vel,
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
    const float DEG_TO_RAD_F = 0.017453293f;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int grid_size = gridDim.x*blockDim.x;

    //
    // Extract some parameters for convenience
    //

    const float xo = params_prj[0];
    const float yo = params_prj[1];
    const float pa = params_prj[2] * DEG_TO_RAD_F;
    const float incl = params_prj[3] * DEG_TO_RAD_F;
    const float vsig = param_vsig;
    const float vsys = param_vsys;

    //
    // Evaluate cube
    //

    while(idx < data_size_x*data_size_y*data_size_z)
    {
        int x, y, z;
        map_index_1d_to_3d(x, y, z, idx, data_size_x, data_size_y, data_size_z);

        float flx_spat, vel_spat, sig_spat, flx_spec;
        float xn, yn, zn, xd, yd, rd, sini, cosi, sinpa, cospa;

        // Image-to-spatial transform
        xn = data_zero_x + (x+0.5f)*data_step_x;
        yn = data_zero_y + (y+0.5f)*data_step_y;

        // Spatial-to-disk transform
        sini = sinf(incl);
        cosi = cosf(incl);
        sinpa = sinf(pa);
        cospa = cosf(pa);
        xd = -(xn-xo)*sinpa+(yn-yo)*cospa;
        yd = -(xn-xo)*cospa-(yn-yo)*sinpa;
        rd = sqrtf((xd*xd)+(yd/cosi)*(yd/cosi));

        // Calculate flux
        flx_spat = evaluate_profile_flx(profile_flx, rd, params_flx);

        // Calculate rotation velocity
        vel_spat = evaluate_profile_vel(profile_vel, rd, params_vel);

        // Calculate radial velocity
        vel_spat = vsys + (rd > 0 ? vel_spat * sini * xd / rd : vel_spat);

        // Calculate velocity dispersion
        sig_spat = vsig;

        // Evaluate a single spectrum (emission line)
        zn = data_zero_z + z*data_step_z;

        flx_spec = evaluate_profile_gaussian(zn, vel_spat, sig_spat);

        int idx2 = z*data_size_x*data_size_y + y*data_size_x + x;

        data[idx2] = flx_spat * flx_spec;

        // Proceed to the next pixel.
        idx += grid_size;
    }
}

} // namespace kernels_cuda_d
} // namespace gmodel01
} // namespace gmodel
} // namespace gbkfit

#endif // GBKFIT_GMODEL_GMODEL01_GMODEL01_CUDA_KERNELS_D_CUH
