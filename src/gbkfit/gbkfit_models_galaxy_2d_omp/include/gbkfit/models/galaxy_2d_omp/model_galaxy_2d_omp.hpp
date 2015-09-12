#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_OMP_HPP
#define GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_OMP_HPP

#include "gbkfit/models/galaxy_2d/model_factory_galaxy_2d.hpp"
#include "gbkfit/models/galaxy_2d/model_galaxy_2d.hpp"

namespace gbkfit {
namespace models {
namespace galaxy_2d {


class model_galaxy_2d_omp : public model_galaxy_2d
{

private:

    ndarray_host* m_data_x;
    ndarray_host* m_data_y;
    ndarray_host* m_data_z;


    ndarray_host* m_data_flxcube;
    ndarray_host* m_data_flxcube_padded;
    ndarray_host* m_data_flxmap;
    ndarray_host* m_data_velmap;
    ndarray_host* m_data_sigmap;

    std::map<std::string,ndarray*> m_data_map;

    int m_psf_size_x;
    int m_psf_size_y;
    int m_psf_size_z;

    int m_upsampling_x;
    int m_upsampling_y;
    int m_upsampling_z;

    instrument* m_instrument;

public:

    model_galaxy_2d_omp(profile_flx_type profile_flx, profile_vel_type profile_);

    ~model_galaxy_2d_omp();

    void initialize(int size_x, int size_y, int size_z, instrument* instrument) final;

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

}; // class model_galaxy_2d_omp


//!
//! \brief The model_factory_galaxy_2d_omp class
//!
class model_factory_galaxy_2d_omp : public model_factory_galaxy_2d
{

public:

    static const std::string FACTORY_TYPE_NAME;

public:

    model_factory_galaxy_2d_omp(void);

    ~model_factory_galaxy_2d_omp();

    const std::string& get_type_name(void) const final;

    model* create_model(const std::string& info) const final;

}; // class model_factory_galaxy_2d


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_MODEL_GALAXY_2D_OMP_HPP
