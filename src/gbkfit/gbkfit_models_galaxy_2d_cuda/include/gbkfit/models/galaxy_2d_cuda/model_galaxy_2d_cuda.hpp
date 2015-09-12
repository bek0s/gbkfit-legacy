#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_CUDA_HPP
#define GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_CUDA_HPP

#include "gbkfit/models/galaxy_2d/model_factory_galaxy_2d.hpp"
#include "gbkfit/models/galaxy_2d/model_galaxy_2d.hpp"

namespace gbkfit {
namespace models {
namespace galaxy_2d {

class model_galaxy_2d_cuda : public model_galaxy_2d
{

private:

    ndarray* m_model_velmap;
    ndarray* m_model_sigmap;
    std::map<std::string,ndarray*> m_model_data_list;

public:

    model_galaxy_2d_cuda(profile_flx_type profile_flux, profile_vel_type profile_vel);

    ~model_galaxy_2d_cuda();

    void initialize(int size_x, int size_y, int size_z, instrument* instrument) final {}

    const std::string& get_type_name(void) const final;

    const std::map<std::string,ndarray*>& get_data(void) const final;

private:

    const std::map<std::string,ndarray*>& evaluate(int profile_flx_id,
                                                   int profile_vel_id,
                                                   const float param_vsig,
                                                   const float param_vsys,
                                                   const std::vector<float>& params_prj,
                                                   const std::vector<float>& params_flx,
                                                   const std::vector<float>& params_vel) final;
}; // class model_galaxy_2d_cuda


//!
//! \brief The model_factory_galaxy_2d_cuda class
//!
class model_factory_galaxy_2d_cuda : public model_factory_galaxy_2d
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    model_factory_galaxy_2d_cuda(void);

    ~model_factory_galaxy_2d_cuda();

    const std::string& get_type_name(void) const final;

    model* create_model(const std::string& info) const final;

}; // class model_factory_galaxy_2d


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_CUDA_HPP
