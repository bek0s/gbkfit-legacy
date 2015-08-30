#pragma once
#ifndef GBKFIT_MODELS_GALAXY_2D_MODEL_FACTORY_GALAXY_2D_HPP
#define GBKFIT_MODELS_GALAXY_2D_MODEL_FACTORY_GALAXY_2D_HPP

#include "gbkfit/model.hpp"

namespace gbkfit {
namespace models {
namespace galaxy_2d {

//!
//! \brief The model_factory_galaxy_2d class
//!
class model_factory_galaxy_2d : public model_factory
{

public:

    model_factory_galaxy_2d(void);

    ~model_factory_galaxy_2d();

protected:

    void read_parameters(std::stringstream& info) const;

}; // class model_factory_galaxy_2d


} // namespace galaxy_2d
} // namespace models
} // namespace gbkfit

#endif // GBKFIT_MODELS_GALAXY_2D_MODEL_FACTORY_GALAXY_2D_HPP
