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

    ndarray* m_model_flxmap;
    ndarray* m_model_velmap;
    ndarray* m_model_sigmap;
    std::map<std::string,ndarray*> m_model_data_list;

public:

    model_galaxy_2d_omp(int width, int height, float step_x, float step_y, int upsampling_x, int upsampling_y,
                        profile_flux_type profile_flux, profile_rcur_type profile_rcur);

    ~model_galaxy_2d_omp();

    const std::string& get_type_name(void) const final;

    const std::map<std::string,ndarray*>& get_data(void) const final;

private:

    const std::map<std::string,ndarray*>& evaluate(int model_flux_id,
                                                   int model_rcur_id,
                                                   const float parameter_vsys,
                                                   const std::vector<float>& parameters_proj,
                                                   const std::vector<float>& parameters_flux,
                                                   const std::vector<float>& parameters_rcur,
                                                   const std::vector<float>& parameters_vsig) final;

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
