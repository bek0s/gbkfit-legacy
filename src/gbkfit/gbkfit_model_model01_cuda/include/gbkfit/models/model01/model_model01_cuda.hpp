#pragma once
#ifndef GBKFIT_MODELS_MODEL01_MODEL_MODEL01_CUDA_HPP
#define GBKFIT_MODELS_MODEL01_MODEL_MODEL01_CUDA_HPP

#include "gbkfit/models/model01/model_model01.hpp"
#include <cufft.h>
#include "gbkfit/cuda/ndarray.hpp"

namespace gbkfit {
namespace models {
namespace model01 {

class model_model01_cuda : public model_model01
{

private:

    cuda::NDArray* m_data_flxcube_up;
    cuda::NDArray* m_data_flxcube_up_fft;

    cuda::NDArray* m_data_psfcube;
    cuda::NDArray* m_data_psfcube_u;
    cuda::NDArray* m_data_psfcube_up;
    cuda::NDArray* m_data_psfcube_up_fft;

    cuda::NDArray* m_data_flxcube;
    cuda::NDArray* m_data_flxmap;
    cuda::NDArray* m_data_velmap;
    cuda::NDArray* m_data_sigmap;

    std::map<std::string,NDArray*> m_data_map;

    int m_upsampling_x;
    int m_upsampling_y;
    int m_upsampling_z;

    int m_size_x;
    int m_size_y;
    int m_size_z;
    int m_size_u_x;
    int m_size_u_y;
    int m_size_u_z;
    int m_size_up_x;
    int m_size_up_y;
    int m_size_up_z;

    float m_step_x;
    float m_step_y;
    float m_step_z;
    float m_step_u_x;
    float m_step_u_y;
    float m_step_u_z;

    int m_psf_size_u_x;
    int m_psf_size_u_y;
    int m_psf_size_u_z;

    cufftHandle m_fft_plan_flxcube_r2c;
    cufftHandle m_fft_plan_flxcube_c2r;
    cufftHandle m_fft_plan_psfcube_r2c;

public:

    model_model01_cuda(profile_flx_type profile_flux, profile_vel_type profile_vel);

    ~model_model01_cuda();

    void initialize(int size_x, int size_y, int size_z, instrument* instrument) final;

    const std::string& get_type_name(void) const final;

    const std::map<std::string,NDArray*>& get_data(void) const final;

private:

    const std::map<std::string,NDArray*>& evaluate(int profile_flx_id,
                                                   int profile_vel_id,
                                                   const std::vector<float>& params_prj,
                                                   const std::vector<float>& params_flx,
                                                   const std::vector<float>& params_vel,
                                                   const float param_vsys,
                                                   const float param_vsig) final;
}; // class model_model01_cuda


class model_factory_model01_cuda : public model_factory_model01
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    model_factory_model01_cuda(void);

    ~model_factory_model01_cuda();

    const std::string& get_type_name(void) const final;

    Model* create_model(const std::string& info) const final;

}; // class model_factory_model01_cuda

} // namespace model01
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_MODEL01_MODEL_MODEL01_CUDA_HPP
