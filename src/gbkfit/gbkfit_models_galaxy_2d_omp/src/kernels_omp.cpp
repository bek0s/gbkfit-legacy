
#include "gbkfit/models/galaxy_2d_omp/kernels_omp.hpp"
#include <iostream>
namespace gbkfit {
namespace models {
namespace galaxy_2d {
namespace kernels_omp {


void evaluate_profile_flux_expdisk(float& out, float r, float i0, float r0)
{
    out = i0 * expf(-r/r0);
}

void evaluate_profile_rcur_lramp(float& out, float r, float rt, float vt)
{
    out = r <= rt ? vt * r/rt : vt;
}

void evaluate_profile_rcur_arctan(float& out, float r, float rt, float vt)
{
    out = vt * (2.0f/(float)M_PI) * atanf(r/rt);
}

void evaluate_profile_rcur_epinat(float& out, float r, float rt, float vt, float a, float g)
{
    out = vt * powf(r/rt,g)/(1.0f+powf(r/rt,a));
}

void evaluate_model_flux(float& out, float r, int model_id, const float* params)
{
    out = 0;
    float ir = 0;

    if      (model_id == 1)
    {
        float i0 = params[0];
        float r0 = params[1];
        evaluate_profile_flux_expdisk(ir,r,i0,r0);
    }

    out += ir;
}

void evaluate_model_rcur(float& out, float x, float r, float sini, int model_id, const float* params)
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
            evaluate_profile_rcur_lramp(vr,r,rt,vt);
        }
        else if (model_id == 2)
        {
            float rt = params[0];
            float vt = params[1];
            evaluate_profile_rcur_arctan(vr,r,rt,vt);
        }
        else if (model_id == 3)
        {
            float rt = params[0];
            float vt = params[1];
            float a = params[2];
            float g = params[3];
            evaluate_profile_rcur_epinat(vr,r,rt,vt,a,g);
        }

        costheta = x/r;
        out += vr * sini * costheta;
    }
}

void model_image_2d_evaluate(float* out_flxmap,
                             float* out_velmap,
                             float* out_sigmap,
                             int model_id_flux,
                             int model_id_rcur,
                             int data_size_x,
                             int data_size_y,
                             float step_x,
                             float step_y,
                             const float model_parameter_vsys,
                             const float* model_parameters_proj,
                             int model_parameters_proj_length,
                             const float* model_parameters_flux,
                             int model_parameters_flux_length,
                             const float* model_parameters_rcur,
                             int model_parameters_rcur_length,
                             const float* model_parameters_vsig,
                             int model_parameters_vsig_length)
{
    const float DEG_TO_RAD_F = 0.017453293f;

    //  shut up compiler
    (void)model_parameters_proj_length;
    (void)model_parameters_flux_length;
    (void)model_parameters_rcur_length;
    (void)model_parameters_vsig_length;

    // velocity offset parameter
    const float vsys = model_parameter_vsys;

    // projection parameters
    const float xo = model_parameters_proj[0];
    const float yo = model_parameters_proj[1];
    const float pa = model_parameters_proj[2] * DEG_TO_RAD_F;
    const float incl = model_parameters_proj[3] * DEG_TO_RAD_F;

    // velocity dispersion parameters
    const float vsig = model_parameters_vsig[0];

    //#pragma omp parallel for
    for(int y = 0; y < data_size_y; ++y)
    {
        //#pragma omp simd
        for(int x = 0; x < data_size_x; ++x)
        {
            float flx_spat,vel_spat,sig_spat;
            float xn,yn,xe,ye,rn,sini,cosi,sinpa,cospa;

            // account for margins and sampling (if needed)
            xn = x;
            yn = y;

            // calculate image-to-disk projection parameters
            sini = sinf(incl);
            cosi = cosf(incl);
            sinpa = sinf(pa);
            cospa = cosf(pa);
            xe = -(xn-xo)*sinpa+(yn-yo)*cospa;
            ye = -(xn-xo)*cospa-(yn-yo)*sinpa;
            rn = sqrtf((xe*xe)+(ye/cosi)*(ye/cosi));

            // calculate spatial flux
            evaluate_model_flux(flx_spat,rn,model_id_flux,model_parameters_flux);

            // calculate spatial velocity
            evaluate_model_rcur(vel_spat,xe,rn,sini,model_id_rcur,model_parameters_rcur);
            vel_spat = vel_spat + vsys;

            // valculate spatial velocity dispersion
            sig_spat = vsig;

            // write results to output images
            const int dst_idx = y * data_size_x + x;
            out_flxmap[dst_idx] = flx_spat;
            out_velmap[dst_idx] = vel_spat;
            out_sigmap[dst_idx] = sig_spat;
        }
    }
}


} // namespace kernels_omp
} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit
