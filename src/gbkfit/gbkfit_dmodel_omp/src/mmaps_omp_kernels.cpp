
#include "gbkfit/dmodel/mmaps/mmaps_omp_kernels.hpp"

#include "mpfit.h"

namespace gbkfit {
namespace dmodel {
namespace mmaps {
namespace kernels_omp {

void evaluate_model_gaussian(float& out, float x, float a, float b, float c, float d)
{
    out = a * expf(-(x-b)*(x-b)/(2*c*c)) + d;
}

struct LMUserData
{
    const float* data;
    int size_x;
    int size_y;
    int size_z;
    int pos_x;
    int pos_y;
    float zero_z;
    float step_z;
};

int mpfit_callback(int num_measurements,
                   int num_params,
                   double* params,
                   double* measurements,
                   double** derivatives,
                   void* adata)
{
    float a = params[0];
    float b = params[1];
    float c = params[2];

    LMUserData* user_data = (LMUserData*)adata;
    const float* data = (const float*)user_data->data;
    int size_x = user_data->size_x;
    int size_y = user_data->size_y;
    int size_z = user_data->size_z;
    int pos_x = user_data->pos_x;
    int pos_y = user_data->pos_y;
    float zero_z = user_data->zero_z;
    float step_z = user_data->step_z;


    #pragma omp simd
    for(int i = 0; i < num_measurements; ++i)
    {
        float x = zero_z + i*step_z;
        float gauss;
        evaluate_model_gaussian(gauss, x, a, b, c, 0);
        int j = pos_y*size_x + pos_x + size_x*size_y*i;
        measurements[i] = data[j] - gauss;
    }

    if(derivatives)
    {
        for (int i = 0; i < num_measurements; i++)
        {
            float x = zero_z + i*step_z;
            if(derivatives[0]) derivatives[0][i] = -    expf(-(x-b)*(x-b)/(2*c*c));
            if(derivatives[1]) derivatives[1][i] = -a * expf(-(x-b)*(x-b)/(2*c*c)) * (x-b)/(c*c);
            if(derivatives[2]) derivatives[2][i] = -a * expf(-(x-b)*(x-b)/(2*c*c)) * (x-b)*(x-b)/(c*c*c);;
        }
    }

    return 0;

}


void extract_maps_gfit(const float* cube,
                       int size_x,
                       int size_y,
                       int size_z,
                       float zero_z,
                       float step_z,
                       float *mom0,
                       float *mom1,
                       float *mom2)
{

    mp_par params_info[3];


    std::memset(params_info, 0, sizeof(params_info));

    params_info[0].step = 0.00001;
    params_info[0].side = 3;
    params_info[0].fixed = 0;
    params_info[0].limited[0] = 1;
    params_info[0].limited[1] = 1;
    params_info[0].limits[0] = 0;
    params_info[0].limits[1] = 5.0;
    params_info[0].parname = 0;

    params_info[1].step = 0.01;
    params_info[1].side = 3;
    params_info[1].fixed = 0;
    params_info[1].limited[0] = 0;
    params_info[1].limited[1] = 0;
    params_info[1].limits[0] = -400;
    params_info[1].limits[1] = +400.0;
    params_info[1].parname = 0;

    params_info[2].step = 0.01;
    params_info[2].side = 3;
    params_info[2].fixed = 0;
    params_info[2].limited[0] = 1;
    params_info[2].limited[1] = 0;
    params_info[2].limits[0] = 0;
    params_info[2].limits[1] = 100.0;
    params_info[2].parname = 0;

    #pragma omp parallel for
    for(int y = 0; y < size_y; ++y)
    {
        for(int x = 0; x < size_x; ++x)
        {
            float flx_sum = 0;
            #pragma omp simd
            for (int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y + y*size_x + x;
                float flx = std::max(0.0f, cube[idx_cube]);
                flx_sum += flx;
            }
            mom0[size_x*y + x] = flx_sum;

            double params[] = {1.0, 0.0, 50.0, 0.0};

            LMUserData lm_user_data;
            lm_user_data.data = cube;
            lm_user_data.size_x = size_x;
            lm_user_data.size_y = size_y;
            lm_user_data.size_z = size_z;
            lm_user_data.pos_x = x;
            lm_user_data.pos_y = y;
            lm_user_data.zero_z = zero_z;
            lm_user_data.step_z = step_z;

            mpfit(mpfit_callback, size_z, 3, params, params_info, nullptr, &lm_user_data, nullptr);

            mom1[size_x*y + x] = params[1];
            mom2[size_x*y + x] = fabsf(params[2]);

        }
    }

}

void extract_maps_mmnt(const float* cube,
                       int size_x,
                       int size_y,
                       int size_z,
                       float zero_z,
                       float step_z,
                       float* mom0,
                       float* mom1,
                       float* mom2)
{
    #pragma omp parallel for
    for (int x = 0; x < size_x; ++x)
    {
        for (int y = 0; y < size_y; ++y)
        {
            float m0, m1, m2;
            float flx_sum = 0;
            float flxvel_sum = 0;
            float flxdvel2_sum = 0;
            int idx_map = y*size_x + x;

            //
            // Calculate moment 0 for the current spatial position.
            //

            #pragma omp simd
            for (int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y + y*size_x + x;
                float flx = std::max(0.0f, cube[idx_cube]);
                flx_sum += flx;
            }
            m0 = flx_sum;
            mom0[idx_map] = m0;

            //
            // Calculate moment 1 for the current spatial position.
            //

            #pragma omp simd
            for (int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y + y*size_x + x;
                float flx = std::max(0.0f, cube[idx_cube]);
                float vel = zero_z + z*step_z;
                flxvel_sum += flx*vel;
            }
            m1 = flx_sum > 0 ? flxvel_sum/flx_sum : 0;
            m1 = m1 - zero_z - size_z/2*step_z; // ...
            mom1[idx_map] = m1;

            //
            // Calculate moment 2 for the current spatial position.
            //

            #pragma omp simd
            for (int z = 0; z < size_z; ++z)
            {
                int idx_cube = z*size_x*size_y + y*size_x + x;
                float flx = std::max(0.0f, cube[idx_cube]);
                float vel = zero_z + z*step_z;
                flxdvel2_sum += flx*(vel-m1)*(vel-m1);
            }
            m2 = flx_sum > 0 ? std::sqrt(flxdvel2_sum/flx_sum) : 0;
            mom2[idx_map] = m2;
        }
    }
}

} // namespace kernels_omp
} // namespace mmaps
} // namespace dmodel
} // namespace gbkfit
