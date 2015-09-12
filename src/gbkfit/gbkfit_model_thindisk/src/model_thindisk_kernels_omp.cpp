
#include "gbkfit/model_thindisk/model_thindisk_kernels_omp.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>

#define RAD_TO_DEG_F 57.295779513f
#define DEG_TO_RAD_F  0.017453293f

namespace gbkfit {
namespace model_thindisk {
namespace kernels_omp {

    void evaluate_phot_exp(float& out, float r, float m0, float r0)
    {
        out = m0 * expf(-r/r0);
    }

    void evaluate_rcur_arctan(float& out, float r, float rt, float vt)
    {
        out = vt * (2.0f/(float)M_PI) * atanf(r/rt);
    }

    void evaluate_rcur_epinat(float& out, float r, float rt, float vt, float a, float g)
    {
        out = vt * powf(r/rt,g)/(1+powf(r/rt,a));
    }

    void evaluate_gaussian(float& out, float x, float mu, float sigma)
    {
        float a = 1.0f/(sigma*sqrtf(2.0f*(float)M_PI));
        out = a * expf(-(x-mu)*(x-mu)/(2.0f*sigma*sigma));
    }

    void evaluate_model_phot(float& out, float r, const float* params_phot)
    {
        float m0 = params_phot[0];
        float r0 = params_phot[1];
        evaluate_phot_exp(out,r,m0,r0);
    }

    void evaluate_model_rcur(float& out, int model_id, float r, float sini, float costheta, const float* params_rcur)
    {
        float vr;
        out = 0.0f;

        if      (model_id == 1)
        {
            float rt = params_rcur[0];
            float vt = params_rcur[1];
            evaluate_rcur_arctan(vr,r,rt,vt);
        }
        else if (model_id == 2)
        {
            float rt = params_rcur[0];
            float vt = params_rcur[1];
            float a = params_rcur[2];
            float g = params_rcur[3];
            evaluate_rcur_epinat(vr,r,rt,vt,a,g);
        }

        out += vr * sini * costheta;
    }

    void model_image_3d_evaluate(float* out,
                                 int model_id,
                                 int data_width,
                                 int data_height,
                                 int data_depth,
                                 int data_aligned_width,
                                 int data_aligned_height,
                                 int data_aligned_depth,
                                 int data_margin_width_0,
                                 int data_margin_width_1,
                                 int data_margin_height_0,
                                 int data_margin_height_1,
                                 int data_margin_depth_0,
                                 int data_margin_depth_1,
                                 const float* model_params_proj,
                                 int model_params_proj_length,
                                 const float* model_params_phot,
                                 int model_params_phot_length,
                                 const float* model_params_rcur,
                                 int model_params_rcur_length,
                                 const float* model_params_vsig,
                                 int model_params_vsig_length,
                                 const float* cube_sampling,
                                 int cube_sampling_length)
    {
        // unused parameters
        (void)model_params_proj_length;
        (void)model_params_phot_length;
        (void)model_params_rcur_length;
        (void)model_params_vsig_length;
        (void)data_aligned_depth;
        (void)cube_sampling_length;

        // projection parameters
        const float xo = model_params_proj[0];
        const float yo = model_params_proj[1];
        const float pa = model_params_proj[2] * DEG_TO_RAD_F;
        const float incl = model_params_proj[3] * DEG_TO_RAD_F;

        // velocity dispersion parameters
        const float vel_sigma = model_params_vsig[0];

        // cube sampling
        const float step_x = cube_sampling[0];
        const float step_y = cube_sampling[1];
        const float step_z = cube_sampling[2];


        // loop until there are no pixels left
        //#pragma omp parallel for
        for(int y = 0; y < data_height+data_margin_height_0+data_margin_height_1; ++y)
        {
            for(int x = 0; x < data_width+data_margin_width_0+data_margin_width_1; ++x)
            {
                float vel_spat,int_spat,vel_spec,int_spec;
                float xn,yn,xe,ye,rn,sini,cosi,sinpa,cospa,costheta;

                // account for margins and sampling
                xn = x - data_margin_width_0;
                yn = y - data_margin_height_0;
                xn = xn * step_x;
                yn = yn * step_y;

                // calculate image-to-disk projection parameters
                sini = sinf(incl);
                cosi = cosf(incl);
                sinpa = sinf(pa);
                cospa = cosf(pa);
                xe = -(xn-xo)*sinpa+(yn-yo)*cospa;
                ye = -(xn-xo)*cospa-(yn-yo)*sinpa;
                rn = sqrtf((xe*xe)+(ye/cosi)*(ye/cosi));

                // calculate spatial velocity
                vel_spat = 0;
                if(rn > 0) {
                    costheta = xe/rn;
                    evaluate_model_rcur(vel_spat,model_id,rn,sini,costheta,model_params_rcur);
                }

                // calculate spatial intensity
                evaluate_model_phot(int_spat,rn,model_params_phot);

                // evaluate the "velocity spectrum"
                for(int z = 0; z < data_depth+data_margin_depth_0+data_margin_depth_1; ++z)
                {
                    float zn;

                    //  account for margins
                    zn = z - data_margin_depth_0;

                    // calculate the velocity
                    vel_spec = (zn-data_depth/2) * step_z;

                    // calculate the intensity
                    evaluate_gaussian(int_spec,vel_spec,vel_spat,vel_sigma);

                    // calculate and store the final intensity for the current slice
                    out[z*data_aligned_width*data_aligned_height+y*data_aligned_width+x] = 100.0f * int_spat * int_spec;
                //  out[z*data_aligned_width*data_aligned_height+y*data_aligned_width+x] = 1;
                //  out[z*data_aligned_width*data_aligned_height+y*data_aligned_width+x] = zn;
                //  out[z*data_aligned_width*data_aligned_height+y*data_aligned_width+x] = vel_spec;
                }
            }
        }
    }

    void array_complex_multiply_and_scale(std::complex<float>* inout_array1,
                                          const std::complex<float>* array2,
                                          int length,
                                          int batch,
                                          float scale)
    {
        int idx;

        //  loop until there are no elements left
        //#pragma omp parallel for
        for(idx = 0; idx < length*batch; ++idx)
        {
            std::complex<float> a(inout_array1[idx].real(),
                                  inout_array1[idx].imag());

            std::complex<float> b(array2[idx%length].real(),
                                  array2[idx%length].imag());

            //  perform complex multiplication and scalar scale
            std::complex<float> t((a.real()*b.real()-a.imag()*b.imag())*scale,
                                  (a.real()*b.imag()+a.imag()*b.real())*scale);

            //  store result
            inout_array1[idx].real(t.real());
            inout_array1[idx].imag(t.imag());
        }
    }

    void model_image_3d_convolve_fft(float* inout_img,
                                     std::complex<float>* img_fft,
                                     const std::complex<float>* krl_fft,
                                     int width,
                                     int height,
                                     int depth,
                                     int batch,
                                     fftwf_plan plan_r2c,
                                     fftwf_plan plan_c2r)
    {
        //  perform real-to-complex transform
        fftwf_execute_dft_r2c(plan_r2c,inout_img,reinterpret_cast<fftwf_complex*>(img_fft));

        int length = depth*height*(width/2+1);
        float norm_factor = 1.0f/(width*height*depth);

        array_complex_multiply_and_scale(img_fft,krl_fft,length,batch,norm_factor);

        //  perform complex-to-real transform
        fftwf_execute_dft_c2r(plan_c2r,reinterpret_cast<fftwf_complex*>(img_fft),inout_img);
    }

    void model_image_3d_downsample(float* data_dst,
                                   float* data_src,
                                   int data_downsampled_width,
                                   int data_downsampled_height,
                                   int data_downsampled_depth,
                                   int data_aligned_width,
                                   int data_aligned_height,
                                   int data_aligned_depth,
                                   int data_margin_width_0,
                                   int data_margin_height_0,
                                   int data_margin_depth_0,
                                   int downsample_x,
                                   int downsample_y,
                                   int downsample_z)
    {
        // unused parameters
        (void)data_aligned_depth;

        // calculate normalization constant
        const float norm_constant = 1.0f/(downsample_x*downsample_y*downsample_z);

        //  loop until there are no pixels left
        //#pragma omp parallel for
        for(int z = 0; z < data_downsampled_depth; ++z)
        {
            for(int y = 0; y < data_downsampled_height; ++y)
            {
                for(int x = 0; x < data_downsampled_width; ++x)
                {
                    float sum = 0;
                    int nx = data_margin_width_0 + x * downsample_x;
                    int ny = data_margin_height_0 + y * downsample_y;
                    int nz = data_margin_depth_0 + z * downsample_z;
                    int idx_dst = z*data_downsampled_height*data_downsampled_width + y*data_downsampled_width + x;

                    for(int dsz = 0; dsz < downsample_z; ++dsz)
                    {
                        for(int dsy = 0; dsy < downsample_y; ++dsy)
                        {
                            for(int dsx = 0; dsx < downsample_x; ++dsx)
                            {
                                int idx_src = (nz+dsz)*data_aligned_height*data_aligned_width + (ny+dsy)*data_aligned_width + (nx+dsx);
                                sum += data_src[idx_src];
                            }
                        }
                    }
                    data_dst[idx_dst] = sum * norm_constant;
                }
            }
        }
    }

    void model_image_3d_extract_moment_maps(float* out_velmap,
                                            float* out_sigmap,
                                            const float* cube,
                                            int data_width,
                                            int data_height,
                                            int data_depth,
                                            const float* cube_sampling,
                                            int cube_sampling_length,
                                            float velmap_offset,
                                            float sigmap_offset)
    {
        //  unused parameters
        (void)cube_sampling_length;

        //  cube sampling
        const float step_z = cube_sampling[2];

        //  loop until there are no pixels left
        //#pragma omp parallel for
        for(int y = 0; y < data_height; ++y)
        {
            for(int x = 0; x < data_width; ++x)
            {
                float m1;
                float m2;
                float sum_intvel = 0;
                float sum_int = 0;
                float sum_intdvel2 = 0;

                //  calculate the mean of the "velocity spectrum" for the current spatial position
                for(int z = 0; z < data_depth; ++z)
                {
                    //  calculate the velocity for the current slice
                    float vel_cur = (z-data_depth/2) * step_z;

                    //  get intensity of the current slice
                    float int_cur = cube[z*data_width*data_height+y*data_width+x];
                    int_cur = std::max(0.0f,int_cur);

                    //  calculate quantities needed for the first moment
                    sum_intvel += int_cur*vel_cur;
                    sum_int += int_cur;
                }
                m1 = 0.0f;
                if(sum_int > 0.0f)
                    m1 = sum_intvel/sum_int;
                out_velmap[y*data_width+x] = velmap_offset + m1;

                //  calculate the variance of the "velocity spectrum" for the current spatial position
                for(int z = 0; z < data_depth; ++z)
                {
                    //  calculate the velocity for the current slice
                    float vel_cur = (z-data_depth/2) * step_z;

                    //  get intensity of the current slice
                    float int_cur = cube[z*data_width*data_height+y*data_width+x];
                    int_cur = std::max(0.0f,int_cur);

                    //  calculate quantities needed for the second moment
                    sum_intdvel2 += int_cur*(vel_cur-m1)*(vel_cur-m1);
                }
                m2 = 0.0f;
                if(sum_int > 0.0f)
                    m2 = sum_intdvel2/sum_int;
                out_sigmap[y*data_width+x] = sigmap_offset + sqrtf(m2);
            }
        }
    }

} // namespace kernels_omp
} // namespace model_thindisk
} // namespace gbkfit
