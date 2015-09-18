#pragma once
#ifndef GBKFIT_MODELS_MODEL01_MODEL_MODEL01_OMP_HPP
#define GBKFIT_MODELS_MODEL01_MODEL_MODEL01_OMP_HPP

#include "gbkfit/models/model01/model_model01.hpp"
#include <fftw3.h>

namespace gbkfit {
namespace models {
namespace model01 {

class model_model01_omp : public model_model01
{

private:

    ndarray_host* m_data_flxcube_up;
    ndarray_host* m_data_flxcube_up_fft;

    ndarray_host* m_data_psfcube;
    ndarray_host* m_data_psfcube_u;
    ndarray_host* m_data_psfcube_up;
    ndarray_host* m_data_psfcube_up_fft;

    ndarray_host* m_data_flxcube;
    ndarray_host* m_data_flxmap;
    ndarray_host* m_data_velmap;
    ndarray_host* m_data_sigmap;

    std::map<std::string,ndarray*> m_data_map;

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

    fftwf_plan m_fft_plan_flxcube_r2c;
    fftwf_plan m_fft_plan_flxcube_c2r;
    fftwf_plan m_fft_plan_psfcube_r2c;

public:

    model_model01_omp(profile_flx_type profile_flx, profile_vel_type profile_);

    ~model_model01_omp();

    void initialize(int size_x, int size_y, int size_z, instrument* instrument) final;

    const std::string& get_type_name(void) const final;

    const std::map<std::string,ndarray*>& get_data(void) const final;

private:

    const std::map<std::string,ndarray*>& evaluate(int profile_flx_id,
                                                   int profile_vel_id,
                                                   const std::vector<float>& params_prj,
                                                   const std::vector<float>& params_flx,
                                                   const std::vector<float>& params_vel,
                                                   const float param_vsys,
                                                   const float param_vsig) final;

}; // class model_model01_omp

class model_factory_model01_omp : public model_factory_model01
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    model_factory_model01_omp(void);

    ~model_factory_model01_omp();

    const std::string& get_type_name(void) const final;

    model* create_model(const std::string& info) const final;

}; // class model_factory_model01

} // namespace model01
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_MODEL01_MODEL_MODEL01_OMP_HPP
