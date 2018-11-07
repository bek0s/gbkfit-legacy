
#include "gbkfit/gmodel/gmodel1/gmodel1_omp_kernels.hpp"

namespace gbkfit {
namespace gmodel {
namespace gmodel1 {
namespace kernels_omp {

float evaluate_profile_gaussian(float x, float mu, float sigma)
{
    float a = 1.0f / (sigma * std::sqrt(2.0f*(float)M_PI));
    return a * std::exp(-(x-mu)*(x-mu)/(2.0f*sigma*sigma));
}

float evaluate_profile_flx_exponential(float r, float i0, float r0)
{
    return i0 * std::exp(-r/r0);
}

float evaluate_profile_vel_lramp(float r, float rt, float vt)
{
    return r <= rt ? vt * r/rt : vt;
}

float evaluate_profile_vel_boissier(float r, float rt, float vt)
{
    return vt * (1.0f - std::exp(-r/rt));
}

float evaluate_profile_vel_arctan(float r, float rt, float vt)
{
    return vt * (2.0f/(float)M_PI) * std::atan(r/rt);
}

float evaluate_profile_vel_epinat(float r, float rt, float vt, float a, float g)
{
    return vt * std::pow(r/rt,g)/(1.0f+std::pow(r/rt,a));
}

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
    const float DEG_TO_RAD_F = 0.017453293f;

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

    #pragma omp parallel for
    for(int x = 0; x < data_size_x; ++x)
    {
        for(int y = 0; y < data_size_y; ++y)
        {
            float flx_spat, vel_spat, sig_spat, flx_spec;
            float xn, yn, zn, xd, yd, rd, sini, cosi, sinpa, cospa;

            // Image-to-spatial transform
            xn = data_zero_x + (x+0.5f)*data_step_x;
            yn = data_zero_y + (y+0.5f)*data_step_y;

            // Spatial-to-disk transform
            sini = std::sin(incl);
            cosi = std::cos(incl);
            sinpa = std::sin(pa);
            cospa = std::cos(pa);
            xd = -(xn-xo)*sinpa+(yn-yo)*cospa;
            yd = -(xn-xo)*cospa-(yn-yo)*sinpa;
            rd = std::sqrt((xd*xd)+(yd/cosi)*(yd/cosi));

            // Calculate flux
            flx_spat = evaluate_profile_flx(flx_profile, rd, params_flx);

            // Calculate rotation velocity
            vel_spat = evaluate_profile_vel(vel_profile, rd, params_vel);

            // Calculate radial velocity
            vel_spat = vsys + (rd > 0 ? vel_spat * sini * xd / rd : vel_spat);

            // Calculate velocity dispersion
            sig_spat = vsig;

            // Evaluate a single spectrum (emission line)
            #pragma omp simd
            for(int z = 0; z < data_size_z; ++z)
            {
                zn = data_zero_z + z*data_step_z;

                flx_spec = evaluate_profile_gaussian(zn, vel_spat, sig_spat);

                int idx = z*data_size_x*data_size_y + y*data_size_x + x;

                data[idx] = flx_spat * flx_spec;
            }
        }
    }
}

} // namespace kernels_omp
} // namespace gmodel1
} // namespace gmodel
} // namespace gbkfit
