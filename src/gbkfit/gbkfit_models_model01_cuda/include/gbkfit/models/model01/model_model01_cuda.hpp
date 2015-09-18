#pragma once
#ifndef GBKFIT_MODELS_MODEL01_MODEL_MODEL01_CUDA_HPP
#define GBKFIT_MODELS_MODEL01_MODEL_MODEL01_CUDA_HPP

#include "gbkfit/models/model01/model_model01.hpp"

namespace gbkfit {
namespace models {
namespace model01 {

class model_model01_cuda : public model_model01
{

private:

    ndarray* m_model_velmap;
    ndarray* m_model_sigmap;
    std::map<std::string,ndarray*> m_model_data_list;

public:

    model_model01_cuda(profile_flx_type profile_flux, profile_vel_type profile_vel);

    ~model_model01_cuda();

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
}; // class model_model01_cuda


class model_factory_model01_cuda : public model_factory_model01
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    model_factory_model01_cuda(void);

    ~model_factory_model01_cuda();

    const std::string& get_type_name(void) const final;

    model* create_model(const std::string& info) const final;

}; // class model_factory_model01_cuda


} // namespace model01
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_MODEL01_MODEL_MODEL01_CUDA_HPP
